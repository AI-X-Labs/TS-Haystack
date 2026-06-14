# SPDX-License-Identifier: CC-BY-NC-4.0

"""
QADataset wrapper for the Sleep PSG TS-Haystack benchmark.

Wraps the parquet metadata produced by the sleep PSG haystack generator and
loads 13-channel PSG signals on demand via :func:`loader.load_window`. Output
matches the contract of :class:`TSHaystackQADataset`: a list of
:class:`TextTimeSeriesPrompt`, plus pre/post prompts and an answer.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from src.datasets.qa_base import QADataset
from src.datasets.sleep_psg_haystack.loader import (
    CHANNEL_NAMES,
    EFFECTIVE_HZ,
    load_window,
)
from src.datasets.sleep_psg_haystack.qa_loader import (
    ALL_TASKS,
    load_sleep_psg_haystack_splits,
)
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    extract_final_answer,
    evaluate_answer as evaluate_answer_fn,
)
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt


SLEEP_STAGE_CLASSES = ["Wake", "N1", "N2", "N3", "REM"]
AROUSAL_CLASSES = [
    "rera",
    "hypopnea",
    "obstructive_apnea",
    "central_apnea",
    "mixed_apnea",
]


ANSWER_FORMAT_GUIDANCE = {
    "boolean": "Answer with 'Yes' or 'No'.",
    "integer": "Answer with a number.",
    "category": "Answer with the class name.",
    "time_range": "Answer with the time range in the format 'HH:MM:SS to HH:MM:SS'.",
    "timestamp": "Answer with the time in the format 'HH:MM:SS'.",
}


def _format_duration(ms: int) -> str:
    s = max(0, int(round(ms / 1000)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class SleepPSGHaystackQADataset(QADataset):
    """
    QADataset for the 13-channel sleep PSG haystack.

    Args:
        split: "train", "validation", or "test".
        EOS_TOKEN: end-of-sequence token to append to answers.
        label_class: "sleep_stages" or "arousals".
        tasks: list of task names; ``["all"]`` resolves to :data:`ALL_TASKS`.
        context_lengths_seconds: list of context lengths (e.g. ``[900, 3600]``)
            or ``["all"]``.
        context_length_seconds: convenience scalar passed by curriculum YAML —
            normalized into a single-element list. Mutually exclusive with
            ``context_lengths_seconds``.
    """

    _cached_config: Optional[tuple] = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        label_class: str,
        tasks: Optional[List[str]] = None,
        context_lengths_seconds: Optional[List[Union[int, float, str]]] = None,
        context_length_seconds: Optional[Union[int, float, str]] = None,
        format_sample_str: bool = False,
        time_series_format_function: Optional[Callable[[np.ndarray], str]] = None,
    ):
        if label_class not in ("sleep_stages", "arousals"):
            raise ValueError(
                f"label_class must be 'sleep_stages' or 'arousals', got {label_class!r}"
            )
        self.label_class = label_class

        if context_lengths_seconds is not None and context_length_seconds is not None:
            raise ValueError(
                "Pass either context_lengths_seconds (list) or "
                "context_length_seconds (scalar), not both."
            )
        if context_length_seconds is not None:
            context_lengths_seconds = [context_length_seconds]
        if context_lengths_seconds is None:
            context_lengths_seconds = ["all"]

        if tasks is None:
            tasks = ["all"]
        if "all" in tasks:
            self.tasks = list(ALL_TASKS)
        else:
            self.tasks = list(tasks)

        self.context_lengths_seconds = list(context_lengths_seconds)

        self._check_cache_config()

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _check_cache_config(self) -> None:
        cfg = (
            self.label_class,
            tuple(sorted(self.tasks)),
            tuple(sorted(str(c) for c in self.context_lengths_seconds)),
        )
        cls = self.__class__
        if cls._cached_config is None:
            cls._cached_config = cfg
            return
        if cls._cached_config != cfg:
            # Invalidate the QADataset base class cache so we re-load.
            warnings.warn(
                f"SleepPSGHaystackQADataset config changed "
                f"(was {cls._cached_config}, now {cfg}); reloading splits.",
                UserWarning,
                stacklevel=3,
            )
            for attr in ("_raw_loaded", "_raw_train", "_raw_val", "_raw_test"):
                if hasattr(cls, attr):
                    delattr(cls, attr)
            cls._cached_config = cfg

    # ------------------------------------------------------------------
    # QADataset hooks
    # ------------------------------------------------------------------

    def _load_splits(self):
        return load_sleep_psg_haystack_splits(
            label_class=self.label_class,
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
        )

    def _get_answer(self, row) -> str:
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        ws = int(row["window_start_ms"])
        we = int(row["window_end_ms"])
        duration = _format_duration(we - ws)
        if self.label_class == "sleep_stages":
            class_str = ", ".join(SLEEP_STAGE_CLASSES)
            class_label = "sleep stages"
        else:
            class_str = ", ".join(AROUSAL_CLASSES)
            class_label = "arousal/respiratory event types"
        channel_str = ", ".join(CHANNEL_NAMES)
        return (
            f"You are given a 13-channel polysomnography (PSG) recording sampled at "
            f"{EFFECTIVE_HZ} Hz. The channels, in order, are: {channel_str}.\n"
            f"This window spans {duration} of the recording. "
            f"Possible {class_label}: {class_str}.\n\n"
            f"Question: {row['question']}"
        )

    def _get_post_prompt(self, row) -> str:
        answer_type = row.get("answer_type", "category")
        guidance = ANSWER_FORMAT_GUIDANCE.get(answer_type, "Provide your answer.")
        return (
            f"\nInstructions:\n"
            f"- Analyze the polysomnography signals carefully across all 13 channels.\n"
            f"- Think step-by-step about what the signal patterns indicate.\n"
            f"- {guidance}\n"
            f'- End your response with "Answer: <your answer>"'
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        signal = load_window(
            row["subject_id"],
            int(row["window_start_ms"]),
            int(row["window_end_ms"]),
        )  # (13, L) at EFFECTIVE_HZ
        prompts = []
        for i, name in enumerate(CHANNEL_NAMES):
            prompts.append(
                TextTimeSeriesPrompt(
                    f"The following is the {name} channel of a 13-channel "
                    f"polysomnography recording sampled at {EFFECTIVE_HZ} Hz.",
                    signal[i],
                )
            )
        return prompts

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def category_key(self) -> str:
        return "task_type"

    def get_ground_truth(self, sample: dict) -> str:
        gt = sample.get("direct_answer") or sample.get("answer", "")
        return gt.replace(self.EOS_TOKEN, "").strip()

    def extract_answer(self, prediction: str, sample: dict) -> str:
        answer_type = sample.get("answer_type", "category")
        return extract_final_answer(prediction, answer_type)

    def evaluate_answer(self, prediction: str, sample: dict) -> dict:
        ground_truth = sample.get("direct_answer") or sample.get("answer", "")
        ground_truth = ground_truth.replace(self.EOS_TOKEN, "").strip()
        answer_type = sample.get("answer_type", "category")
        return evaluate_answer_fn(ground_truth, prediction, answer_type, iou_threshold=0.25)

    # ------------------------------------------------------------------
    # Sample formatting
    # ------------------------------------------------------------------

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["subject_id"] = row.get("subject_id", "")
        sample["window_start_ms"] = int(row.get("window_start_ms", 0))
        sample["window_end_ms"] = int(row.get("window_end_ms", 0))
        sample["task_type"] = row.get("task_type", "unknown")
        sample["answer_type"] = row.get("answer_type", "unknown")
        sample["context_length_s"] = row.get("context_length_s", 0)
        sample["label_class"] = row.get("label_class", self.label_class)
        sample["metadata"] = row.get("metadata", "{}")
        sample["question"] = row.get("question", "")
        return sample

    def get_tasks(self) -> List[str]:
        return list(self.tasks)

    def get_context_lengths(self) -> List[Union[int, float, str]]:
        return list(self.context_lengths_seconds)
