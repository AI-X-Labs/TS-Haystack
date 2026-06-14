"""QADataset implementation for UK-DALE-Haystack generated samples.

Shards are metadata-only -- the additive mains is reconstructed at sample
load time from ``background_house_id``, ``background_start_ns/end_ns`` and
``needles_json`` via ``plot_generator.reconstruct_sample_signal``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Tuple

import numpy as np
from datasets import Dataset

from src.datasets.qa_base import QADataset
from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.plot_generator import reconstruct_sample_signal
from src.datasets.uk_dale_haystack.qa_loader import load_uk_dale_haystack_splits
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer as evaluate_ts_haystack_answer,
    extract_final_answer,
)
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt


# Per-answer-type format guidance. The time_range / timestamp formats must
# match prompt_templates.fmt_time_range exactly so the answer the model
# learns to emit is byte-identical to the ground truth in the parquet.
ANSWER_FORMAT_GUIDANCE = {
    "boolean": "Answer with 'Yes' or 'No'.",
    "integer": "Answer with a single non-negative integer.",
    "category": (
        "Answer with the canonical appliance label, exactly one of: "
        + ", ".join(V1_VOCAB)
        + "."
    ),
    "time_range": (
        "Answer with a window-relative time range in the format "
        "'HH:MM:SS - HH:MM:SS' where 00:00:00 is the start of the trace."
    ),
    "timestamp": (
        "Answer with a window-relative timestamp in the format 'HH:MM:SS' "
        "where 00:00:00 is the start of the trace."
    ),
}


def _format_duration(total_s: float) -> str:
    s = max(0, int(round(total_s)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _format_hms(total_s: float) -> str:
    """HH:MM:SS form, matching prompt_templates.fmt_time_range."""
    s = max(0, int(round(total_s)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class UKDaleHaystackQADataset(QADataset):
    """Runtime dataset for UK-DALE-Haystack synthetic QA samples."""

    IOU_THRESHOLD = 0.25
    # UK-DALE samples on a regular 6 s grid; tolerate one grid step for
    # single-timestamp answers (no current task uses 'timestamp' but keep
    # the knob for parity with the LTAF dataset).
    TIMESTAMP_TOLERANCE_S = 6.0

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        tasks: list[str] | None = None,
        context_lengths_seconds: list[float | str] | None = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
        base_dir: str | Path | None = None,
    ):
        self.tasks = tasks or ["all"]
        self.context_lengths_seconds = context_lengths_seconds or ["all"]
        self.base_dir = Path(base_dir) if base_dir is not None else None
        super().__init__(
            split,
            EOS_TOKEN,
            format_sample_str,
            time_series_format_function,
            lazy_loading,
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_uk_dale_haystack_splits(
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
            base_dir=self.base_dir,
        )

    def _dataset_cache_key(self) -> tuple[Any, ...]:
        tasks_key = tuple(sorted(str(task).strip() for task in self.tasks))
        context_key = tuple(
            sorted(self._normalize_context_key(v) for v in self.context_lengths_seconds)
        )
        return (
            tasks_key,
            context_key,
            str(self.base_dir) if self.base_dir is not None else None,
        )

    @staticmethod
    def _normalize_context_key(value: float | str) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):g}"
        raw = str(value).strip().lower()
        if not raw or raw == "all":
            return "all"
        if raw.endswith("s"):
            raw = raw[:-1]
        raw = raw.replace("_", ".")
        try:
            return f"{float(raw):g}"
        except ValueError:
            return raw

    def _get_answer(self, row) -> str:
        return str(row.get("answer", ""))

    def _get_pre_prompt(self, row) -> str:
        question = str(row.get("question", "Analyze the mains power trace and answer the question."))
        ctx_s = float(row.get("context_length_s", 0) or 0)
        dt_s = float(row.get("dt_s", 6.0) or 6.0)
        duration = _format_duration(ctx_s) if ctx_s > 0 else "the full window"
        end_hms = _format_hms(ctx_s) if ctx_s > 0 else "the end of the trace"
        return (
            "You are given a long-context UK domestic mains active-power trace "
            f"(watts, sampled every {dt_s:g} s).\n"
            f"This window spans {duration} of the recording, from 00:00:00 to {end_hms}. "
            "Time references in the question and answer are window-relative offsets "
            "from 00:00:00, not wall-clock times.\n\n"
            f"Question: {question}"
        )

    def _get_post_prompt(self, row) -> str:
        answer_type = str(row.get("answer_type", "category")).strip().lower()
        guidance = ANSWER_FORMAT_GUIDANCE.get(answer_type, "Provide your answer.")
        return (
            "\nInstructions:\n"
            "- Analyze the mains active-power trace carefully.\n"
            "- Think step-by-step about what the appliance signatures indicate.\n"
            f"- {guidance}\n"
            '- End your response with "Answer: <your answer>"'
        )

    def _get_text_time_series_prompt_list(self, row) -> list[TextTimeSeriesPrompt]:
        mains_w = reconstruct_sample_signal(row)
        return [TextTimeSeriesPrompt("mains active power (W)", mains_w.tolist())]

    @property
    def category_key(self) -> str:
        return "task_type"

    def _attach_metadata(self, sample: dict, row: dict) -> dict:
        if "task_type" in row:
            sample["task_type"] = row["task_type"]
        if "answer_type" in row:
            sample["answer_type"] = row["answer_type"]
        if "context_length_s" in row:
            sample["context_length_s"] = float(row["context_length_s"])
        if "dt_s" in row:
            sample["dt_s"] = float(row["dt_s"])
        if "question" in row:
            sample["question"] = str(row["question"])
        return sample

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        return self._attach_metadata(sample, row)

    def _format_sample_str(
        self,
        time_series_format_function: Callable[[np.ndarray], str] | None,
        row,
    ):
        sample = super()._format_sample_str(time_series_format_function, row)
        return self._attach_metadata(sample, row)

    def extract_answer(self, prediction: str, sample: dict) -> str:
        answer_type = str(sample.get("answer_type", "category")).strip().lower()
        if "answer:" in prediction.lower():
            return extract_final_answer(prediction, answer_type)
        return prediction.strip()

    def evaluate_answer(self, prediction: str, sample: dict) -> dict:
        answer_type = str(sample.get("answer_type", "category")).strip().lower()
        ground_truth = str(sample.get("direct_answer") or sample.get("answer", ""))

        result = evaluate_ts_haystack_answer(
            ground_truth=ground_truth,
            prediction=prediction,
            answer_type=answer_type,
            iou_threshold=self.IOU_THRESHOLD,
            timestamp_tolerance_s=self.TIMESTAMP_TOLERANCE_S,
        )
        return {
            "correct": bool(result.get("correct", False)),
            "iou": result.get("iou"),
            "timestamp_error_s": result.get("timestamp_error_s"),
            "normalized_gt": result.get("normalized_gt"),
            "normalized_pred": result.get("normalized_pred"),
        }
