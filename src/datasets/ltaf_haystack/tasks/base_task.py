# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Base task generator for LTAF-Haystack (natural-only).

Subclasses implement :meth:`_generate` which consumes an
:class:`LTAFRecordingSample` (a real windowed slice produced by
:class:`RecordingSampler`) and returns a :class:`LTAFGeneratedSample`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.core.ltaf_prompt_templates import (
    LTAFPromptTemplateBank,
)
from src.datasets.ltaf_haystack.core.recording_sampler import RecordingSampler
from src.datasets.ltaf_haystack.core.seed_manager import LTAFSeedManager


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class LTAFBaseTaskGenerator(ABC):
    """Abstract base for LTAF-Haystack task generators (natural-only)."""

    def __init__(
        self,
        recording_sampler: RecordingSampler,
        template_bank: LTAFPromptTemplateBank,
        seed_manager: LTAFSeedManager,
        label_class: str = "rhythms",
    ):
        self.recording_sampler = recording_sampler
        self.template_bank = template_bank
        self.seed_manager = seed_manager
        self.label_class = label_class
        self.source_hz = int(recording_sampler._source_hz)

    # ------------------------------------------------------------------
    # Task identity / context gating
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def task_name(self) -> str: ...

    @property
    @abstractmethod
    def answer_type(self) -> str: ...

    @classmethod
    def supports_context_length(
        cls, label_class: str, context_length_s: Optional[float]
    ) -> bool:
        # LTAF always operates on fixed-length windows; whole-recording
        # (context_length_s=None) is not a supported mode.
        return context_length_s is not None

    # ------------------------------------------------------------------
    # Core entry point
    # ------------------------------------------------------------------
    def generate_sample(
        self,
        recording: LTAFRecordingSample,
        rng: np.random.Generator,
    ) -> LTAFGeneratedSample:
        return self._generate(recording, rng)

    @abstractmethod
    def _generate(
        self,
        recording: LTAFRecordingSample,
        rng: np.random.Generator,
    ) -> LTAFGeneratedSample: ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _materialize_signals(self, recording: LTAFRecordingSample) -> np.ndarray:
        if recording.signals is not None:
            return recording.signals
        return self.recording_sampler.load_signals(recording)

    def _build_sample(
        self,
        recording: LTAFRecordingSample,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LTAFGeneratedSample:
        signals = self._materialize_signals(recording)
        ctx_samples = int(signals.shape[0])
        return LTAFGeneratedSample(
            task_type=self.task_name,
            question=question,
            answer=str(answer),
            answer_type=self.answer_type,
            signals=np.asarray(signals, dtype=np.float32),
            context_length_samples=ctx_samples,
            record_id=str(recording.record_id),
            window_start_ms=int(recording.window_start_ms),
            window_end_ms=int(recording.window_end_ms),
            metadata=metadata or {},
            is_valid=True,
            invalid_reason=None,
            source_hz=int(recording.source_hz),
        )

    def _create_invalid_sample(
        self,
        reason: str,
        recording: Optional[LTAFRecordingSample] = None,
    ) -> LTAFGeneratedSample:
        if recording is not None:
            ctx = int(recording.duration_samples)
            record_id = recording.record_id
            win_start = int(recording.window_start_ms)
            win_end = int(recording.window_end_ms)
            hz = int(recording.source_hz)
        else:
            ctx = 0
            record_id = ""
            win_start = 0
            win_end = 0
            hz = self.source_hz
        return LTAFGeneratedSample(
            task_type=self.task_name,
            question="",
            answer="",
            answer_type=self.answer_type,
            signals=np.zeros((max(ctx, 0), 2), dtype=np.float32),
            context_length_samples=ctx,
            record_id=str(record_id),
            window_start_ms=win_start,
            window_end_ms=win_end,
            metadata={},
            is_valid=False,
            invalid_reason=str(reason),
            source_hz=hz,
        )

    @staticmethod
    def _ms_to_timestamp(ms: int) -> str:
        total_s = ms / 1000.0
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = int(total_s % 60)
        ms_part = int(ms % 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms_part:03d}"

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------
    def _dedup_key(self, sample: LTAFGeneratedSample) -> Tuple:
        return (sample.record_id, sample.window_start_ms, sample.question)

    def generate_dataset(
        self,
        n_samples: int,
        split: str,
        n_jobs: int = 1,
        retry_factor: float = 10.0,
        verbose: bool = True,
    ) -> List[LTAFGeneratedSample]:
        max_seeds = int(n_samples * retry_factor)
        ctx_key = str(self.recording_sampler.window_index.context_length_s)
        sample_seeds = self.seed_manager.get_sample_seeds(
            task=self.task_name,
            context_length=ctx_key,
            split=split,
            n_samples=max_seeds,
        )

        from tqdm import tqdm

        out: List[LTAFGeneratedSample] = []
        seen: set = set()
        failures = 0
        duplicates = 0
        pbar = tqdm(total=n_samples, desc=f"  {split}", disable=not verbose)
        for seed in sample_seeds:
            if len(out) >= n_samples:
                break
            rng = np.random.default_rng(seed)
            try:
                recording = self.recording_sampler.sample_recording(rng)
            except ValueError:
                failures += 1
                continue
            sample = self.generate_sample(recording, rng)
            if not sample.is_valid:
                failures += 1
                pbar.set_postfix(failures=failures, dups=duplicates)
                continue
            key = self._dedup_key(sample)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            out.append(sample)
            pbar.update(1)
        pbar.close()

        if verbose and len(out) < n_samples:
            print(
                f"  WARNING: {len(out)}/{n_samples} samples for "
                f"task={self.task_name}, split={split} "
                f"({failures} invalid, {duplicates} dups)"
            )
        return out

    def save_dataset(
        self,
        samples: List[LTAFGeneratedSample],
        split: str,
        output_dir: Path,
        context_dir: str,
    ) -> Path:
        import polars as pl

        task_dir = Path(output_dir) / context_dir / self.task_name / split
        task_dir.mkdir(parents=True, exist_ok=True)

        output_path = task_dir / "data.parquet"
        if not samples:
            return output_path

        df = pl.DataFrame(
            {
                "task_type": [s.task_type for s in samples],
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "answer_type": [s.answer_type for s in samples],
                "context_length_samples": [s.context_length_samples for s in samples],
                "record_id": [s.record_id for s in samples],
                "window_start_ms": [s.window_start_ms for s in samples],
                "window_end_ms": [s.window_end_ms for s in samples],
                "metadata": [json.dumps(s.metadata) for s in samples],
                "is_valid": [bool(s.is_valid) for s in samples],
                "invalid_reason": [s.invalid_reason or "" for s in samples],
                "source_hz": [s.source_hz for s in samples],
            }
        )
        df.write_parquet(output_path)
        return output_path


__all__ = ["LTAFBaseTaskGenerator", "_ordinal"]
