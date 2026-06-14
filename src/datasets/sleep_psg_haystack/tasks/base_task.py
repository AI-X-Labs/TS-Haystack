# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Base task generator for the natural annotation-based Sleep PSG benchmark.

Generates questions from real sleep stage and arousal annotations.
No needle insertion — recordings are used unmodified.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.recording_sampler import RecordingSampler
from src.datasets.sleep_psg_haystack.core.prompt_templates import (
    SleepPromptTemplateBank,
    ordinal,
)
from src.datasets.sleep_psg_haystack.loader import SOURCE_HZ

from src.datasets.capture24_haystack.core.data_structures import BoutRecord
from src.datasets.capture24_haystack.core.seed_manager import SeedManager


class PSGBaseTaskGenerator(ABC):
    """
    Abstract base class for annotation-based PSG task generators.

    Each subclass generates questions from natural annotations in a
    full recording. No signal modification occurs.
    """

    def __init__(
        self,
        recording_sampler: RecordingSampler,
        template_bank: SleepPromptTemplateBank,
        seed_manager: SeedManager,
        label_class: str = "sleep_stages",
    ):
        self.recording_sampler = recording_sampler
        self.template_bank = template_bank
        self.seed_manager = seed_manager
        self.label_class = label_class

    @property
    @abstractmethod
    def task_name(self) -> str: ...

    @property
    @abstractmethod
    def answer_type(self) -> str: ...

    @abstractmethod
    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample: ...

    @classmethod
    def supports_context_length(
        cls, label_class: str, context_length_s: Optional[float],
    ) -> bool:
        """
        Whether this task should be generated for a given (label_class, context).

        Default: applicable everywhere. Override (e.g. PSGExistenceTaskGenerator)
        to gate by context length when label balance would degenerate.
        """
        return True

    # =========================================================================
    # Annotation Helpers
    # =========================================================================

    @staticmethod
    def _window_fields(recording: PSGRecordingSample) -> Dict[str, Any]:
        """Extract window metadata for a generated sample's parquet row."""
        return {
            "context_length_s": recording.context_length_s,
            "window_start_ms": recording.window_start_ms,
            "window_end_ms": recording.window_start_ms + recording.recording_duration_ms,
        }

    @staticmethod
    def _ms_to_timestamp(ms: int) -> str:
        """Convert milliseconds to HH:MM:SS.mmm format."""
        total_s = ms / 1000
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = int(total_s % 60)
        ms_part = int(ms % 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms_part:03d}"

    @staticmethod
    def _bout_time_range(bout: BoutRecord) -> Tuple[str, str]:
        """Return (start_timestamp, end_timestamp) for a bout."""
        return (
            PSGBaseTaskGenerator._ms_to_timestamp(bout.start_ms),
            PSGBaseTaskGenerator._ms_to_timestamp(bout.end_ms),
        )

    @staticmethod
    def _get_sorted_timeline(recording: PSGRecordingSample, label_class: str) -> List[BoutRecord]:
        """Get the full timeline sorted by start time for a given label class."""
        if label_class == "sleep_stages":
            return sorted(recording.sleep_stages_timeline, key=lambda b: b.start_ms)
        else:
            return sorted(recording.arousals_timeline, key=lambda b: b.start_ms)

    def _create_invalid_sample(
        self, reason: str, recording: Optional[PSGRecordingSample] = None,
    ) -> PSGGeneratedSample:
        ctx_s = recording.context_length_s if recording else None
        win_start = recording.window_start_ms if recording else 0
        win_end = (win_start + recording.recording_duration_ms) if recording else 0
        return PSGGeneratedSample(
            subject_id=recording.subject_id if recording else "",
            recording_duration_ms=recording.recording_duration_ms if recording else 0,
            label_class=self.label_class,
            task_type=self.task_name,
            context_length_s=ctx_s,
            window_start_ms=win_start,
            window_end_ms=win_end,
            question="",
            answer="",
            answer_type=self.answer_type,
            is_valid=False,
            validation_notes=reason,
        )

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def _dedup_key(self, sample: PSGGeneratedSample) -> Tuple:
        """
        Hashable key used to reject duplicate samples within a single
        generate_dataset call. Subclasses may override to use a stricter
        semantic key (e.g. ignoring template-rendering variation).

        Default rejects bit-for-bit collisions: same subject + same window +
        same exact question text. This is enough to eliminate the wasted-budget
        duplicates produced by tasks with small candidate spaces (notably
        comparison and existence at long context lengths) without preventing
        the same logical target from appearing under different question
        templates.
        """
        return (sample.subject_id, sample.window_start_ms, sample.question)

    def generate_dataset(
        self, n_samples: int, split: str,
        n_jobs: int = 1, retry_factor: float = 10.0, verbose: bool = True,
    ) -> List[PSGGeneratedSample]:
        max_seeds = int(n_samples * retry_factor)
        # Key seeds by the actual context length so different windows produce
        # independent sample streams.
        if self.recording_sampler.window_index is not None:
            ctx_key = self.recording_sampler.window_index.context_length_s
            if ctx_key is None:
                ctx_key = "full"
        else:
            ctx_key = "full"
        sample_seeds = self.seed_manager.get_sample_seeds(
            task=self.task_name,
            context_length=str(ctx_key),
            split=split,
            n_samples=max_seeds,
        )

        from tqdm import tqdm

        samples = []
        seen_keys: set = set()
        failures = 0
        duplicates = 0
        pbar = tqdm(total=n_samples, desc=f"  {split}", disable=not verbose)
        for seed in sample_seeds:
            if len(samples) >= n_samples:
                break
            rng = np.random.default_rng(seed)
            recording = self.recording_sampler.sample_recording(rng)
            sample = self.generate_sample(recording, rng)
            if not sample.is_valid:
                failures += 1
                pbar.set_postfix(failures=failures, dups=duplicates)
                continue
            key = self._dedup_key(sample)
            if key in seen_keys:
                duplicates += 1
                pbar.set_postfix(failures=failures, dups=duplicates)
                continue
            seen_keys.add(key)
            samples.append(sample)
            pbar.update(1)
        pbar.close()

        if len(samples) < n_samples:
            print(
                f"WARNING: Only {len(samples)}/{n_samples} after "
                f"{len(samples) + failures + duplicates} attempts "
                f"({failures} failures, {duplicates} dup-rejects)"
            )
        elif duplicates and verbose:
            print(f"  ({duplicates} duplicate samples rejected)")
        return samples

    def save_dataset(
        self, samples: List[PSGGeneratedSample], split: str,
        output_dir: Optional[Path] = None,
        context_dir: Optional[str] = None,
    ) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if output_dir is None:
            from src.datasets.sleep_psg_haystack.loader import SLEEP_PSG_DATA_DIR
            output_dir = SLEEP_PSG_DATA_DIR / "ts_haystack" / self.label_class / "tasks"

        if context_dir is None:
            context_dir = "full"
        task_dir = output_dir / context_dir / self.task_name / split
        task_dir.mkdir(parents=True, exist_ok=True)

        rows = [s.to_dict() for s in samples]
        if not rows:
            return task_dir / "data.parquet"

        arrays = {k: pa.array([r[k] for r in rows]) for k in rows[0]}
        table = pa.table(arrays)
        output_path = task_dir / "data.parquet"
        pq.write_table(table, output_path)
        return output_path
