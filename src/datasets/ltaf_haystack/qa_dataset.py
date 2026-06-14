# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""QADataset implementation for LTAF-Haystack generated samples."""

from pathlib import Path
from typing import Any, Callable, Literal, Tuple

import numpy as np
from datasets import Dataset

from src.datasets.ltaf_haystack.loader import SOURCE_HZ, load_window_ms
from src.datasets.ltaf_haystack.qa_loader import load_ltaf_haystack_splits
from src.datasets.qa_base import QADataset
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer as evaluate_ts_haystack_answer,
    extract_final_answer,
)


class LTAFHaystackQADataset(QADataset):
    """Runtime dataset for LTAF-Haystack synthetic QA samples."""

    IOU_THRESHOLD = 0.25
    # Tolerance in seconds for single-timestamp answers (anomaly_localization).
    # LTAF is 128 Hz; typical beat spacing at HR 60-100 BPM is 0.6-1.0 s, so
    # 0.5 s requires the prediction within half a beat cycle.
    TIMESTAMP_TOLERANCE_S = 0.5

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        tasks: list[str] | None = None,
        context_lengths_seconds: list[float | str] | None = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
        use_cot: bool = False,
        base_dir: str | Path | None = None,
    ):
        self.tasks = tasks or ["all"]
        self.context_lengths_seconds = context_lengths_seconds or ["all"]
        self.use_cot = bool(use_cot)
        self.base_dir = Path(base_dir) if base_dir is not None else None
        super().__init__(
            split,
            EOS_TOKEN,
            format_sample_str,
            time_series_format_function,
            lazy_loading,
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        train, val, test = load_ltaf_haystack_splits(
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
            use_cot=self.use_cot,
            base_dir=self.base_dir,
        )
        return train, val, test

    def _dataset_cache_key(self) -> tuple[Any, ...]:
        tasks_key = tuple(sorted(str(task).strip() for task in self.tasks))
        context_key = tuple(
            sorted(self._normalize_context_key(v) for v in self.context_lengths_seconds)
        )
        return (
            tasks_key,
            context_key,
            self.use_cot,
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
        if row.get("pre_prompt"):
            return str(row["pre_prompt"])
        question = str(row.get("question", "Analyze the ECG and answer the question."))
        return f"You are given long-context two-lead ECG data.\\n\\nQuestion: {question}"

    def _get_post_prompt(self, row) -> str:
        if row.get("post_prompt"):
            return str(row["post_prompt"])
        return "\\nProvide a concise final answer."

    def _get_text_time_series_prompt_list(self, row) -> list[TextTimeSeriesPrompt]:
        signals = load_window_ms(
            record_id=str(row["record_id"]),
            window_start_ms=int(row["window_start_ms"]),
            window_end_ms=int(row["window_end_ms"]),
            source_hz=int(row.get("source_hz", SOURCE_HZ)),
        )
        return [
            TextTimeSeriesPrompt("ECG lead 1", signals[:, 0].tolist()),
            TextTimeSeriesPrompt("ECG lead 2", signals[:, 1].tolist()),
        ]

    @property
    def category_key(self) -> str:
        return "task_type"

    def _attach_metadata(self, sample: dict, row: dict) -> dict:
        if "task_type" in row:
            sample["task_type"] = row["task_type"]
        if "answer_type" in row:
            sample["answer_type"] = row["answer_type"]
        if "context_length_samples" in row:
            sample["context_length_samples"] = int(row["context_length_samples"])
        if "source_hz" in row:
            sample["source_hz"] = int(row["source_hz"])
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
