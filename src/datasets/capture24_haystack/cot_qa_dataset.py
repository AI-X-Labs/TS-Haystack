# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
TS-Haystack Chain-of-Thought QADataset implementation for OpenTSLM training.

This module provides a QADataset subclass that formats TS-Haystack benchmark
data WITH chain-of-thought rationales for training models to reason step-by-step.

Usage:
    from src.datasets.capture24_haystack import TSHaystackCoTQADataset

    # Single task with CoT rationales
    dataset = TSHaystackCoTQADataset(
        split="train",
        EOS_TOKEN=model.get_eos_token(),
        tasks=["existence"],
        context_lengths_seconds=[100],
    )

    # The answer includes the full chain-of-thought rationale
    sample = dataset[0]
    print(sample["answer"])  # Full rationale ending with "Answer: ..."
    print(sample["direct_answer"])  # Just the final answer
"""

from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.datasets.qa_base import QADataset
from src.datasets.capture24_haystack.qa_loader import (
    ALL_CONTEXT_LENGTHS,
    ALL_TASKS,
    load_ts_haystack_splits,
)
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    extract_final_answer,
    evaluate_answer as evaluate_answer_fn,
)


# Axis labels matching HAR/Capture24 format
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]


class TSHaystackCoTQADataset(QADataset):
    """
    QADataset subclass for TS-Haystack benchmark with chain-of-thought rationales.

    This class loads TS-Haystack benchmark data that includes LLM-generated
    chain-of-thought rationales. The model is trained to produce the full
    reasoning process, not just the final answer.

    Note on caching:
        The QADataset base class uses class-level caching. Once data is loaded
        for a specific configuration, it's cached for all subsequent instances.
        If you need different configurations, restart the Python session.

    Args:
        split: Dataset split to load ("train", "test", or "validation")
        EOS_TOKEN: End-of-sequence token appended to answers. Required.
                   Passed by the training script from ``model.get_eos_token()``.
        tasks: List of tasks to load (e.g., ["existence", "localization"])
               Use ["all"] to load all tasks
        context_lengths_seconds: List of context lengths in seconds
                                 (e.g., [100] for 10000 samples at 100Hz)
        cot_small: If True, use the cot_small dataset variant (smaller dataset)
        format_sample_str: If True, format samples as strings (default: False)
        time_series_format_function: Optional function to format time series as strings
        lazy_loading: If True (default), samples are formatted on-demand to save memory.
                      Set to False for legacy behavior (pre-format all samples).

    Example:
        >>> dataset = TSHaystackCoTQADataset(
        ...     split="train",
        ...     EOS_TOKEN="<|endofchunk|>",
        ...     tasks=["existence", "counting"],
        ...     context_lengths_seconds=[100],
        ... )
        >>> sample = dataset[0]
        >>> print(sample["answer"])  # Full rationale
        >>> print(sample["direct_answer"])  # Just the final answer
    """

    # Class-level storage for configuration caching
    # Note: This is separate from TSHaystackQADataset's cache
    _cached_config: Optional[tuple] = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        tasks: List[str] = None,
        context_lengths_seconds: List[Union[str, float, int]] = None,
        cot_small: bool = False,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
    ):
        # Set defaults - use "all" for both tasks and context lengths
        if tasks is None:
            tasks = ["all"]
        if context_lengths_seconds is None:
            context_lengths_seconds = ["all"]

        # Resolve "all" tasks
        if "all" in tasks:
            self.tasks = list(ALL_TASKS)
        else:
            self.tasks = list(tasks)

        # Store context lengths (may be ["all"] - resolved during loading)
        self.context_lengths_seconds = context_lengths_seconds

        # Store cot_small flag
        self.cot_small = cot_small

        # Check for configuration mismatch with cached data
        self._check_cache_config()

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function, lazy_loading
        )

    def _check_cache_config(self) -> None:
        """
        Check if cached data exists with different configuration.
        """
        current_config = (
            tuple(sorted(self.tasks)),
            tuple(sorted(self.context_lengths_seconds)),
        )

        if self.__class__._cached_config is None:
            self.__class__._cached_config = current_config
        elif self.__class__._cached_config != current_config:
            warnings.warn(
                f"TSHaystackCoTQADataset was previously loaded with config "
                f"(tasks={self.__class__._cached_config[0]}, "
                f"context_lengths_seconds={self.__class__._cached_config[1]}). "
                f"Now instantiated with (tasks={current_config[0]}, "
                f"context_lengths_seconds={current_config[1]}). "
                f"Using cached data from original config. "
                f"Restart Python session for different configuration.",
                UserWarning,
                stacklevel=3,
            )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the TS-Haystack dataset splits with CoT rationales.

        Returns:
            Tuple of (train, validation, test) HuggingFace Dataset objects
        """
        return load_ts_haystack_splits(
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
            use_cot=True,  # Load from cot/ directory
            cot_small=self.cot_small,  # Use cot_small/ if enabled
        )

    def _get_answer(self, row) -> str:
        """
        Return the full chain-of-thought rationale as the answer.

        The rationale should end with "Answer: <answer>".
        """
        rationale = row.get("rationale", "")
        if rationale:
            return rationale
        # Fallback to direct answer if no rationale available
        return row.get("answer", "")

    def _get_pre_prompt(self, row) -> str:
        """
        Return the instruction text before the time series.

        Includes recording time context and the question.
        """
        time_start = row.get("recording_time_start", "unknown")
        time_end = row.get("recording_time_end", "unknown")

        return (
            f"You are given accelerometer data in all three dimensions from a "
            f"wrist-worn sensor. The recording spans from {time_start} to {time_end}.\n\n"
            f"Question: {row['question']}"
        )

    def _get_post_prompt(self, row) -> str:
        """
        Return the instruction text after the time series.

        Guides the model to produce chain-of-thought reasoning.
        """
        return (
            "\nInstructions:\n"
            "- Analyze the accelerometer data carefully.\n"
            "- Think step-by-step about what the signal patterns indicate.\n"
            "- Write your reasoning as a natural paragraph.\n"
            '- End your response with "Answer: <your answer>"'
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into TextTimeSeriesPrompt objects.

        Args:
            row: Dataset row containing x_axis, y_axis, z_axis

        Returns:
            List of TextTimeSeriesPrompt objects, one per axis
        """
        # Stack numpy arrays first, then convert to tensor (much faster)
        series = torch.as_tensor(
            np.stack([
                np.asarray(row["x_axis"]),
                np.asarray(row["y_axis"]),
                np.asarray(row["z_axis"]),
            ]),
            dtype=torch.float32,
        )

        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIES_LABELS, series.tolist()
            )
        ]

    # ------------------------------------------------------------------
    # Evaluation overrides
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

    def get_tasks(self) -> List[str]:
        """Return the list of tasks loaded."""
        return list(self.tasks)

    def get_context_lengths(self) -> List[int]:
        """Return the list of context lengths (in seconds)."""
        return list(self.context_lengths_seconds)

    def _format_sample(self, row):
        """
        Format a sample for training, including metadata and direct answer.

        The key difference from TSHaystackQADataset is:
        - "answer" contains the full rationale
        - "direct_answer" contains just the final answer (for evaluation)

        Args:
            row: Dataset row

        Returns:
            Dictionary with formatted prompt, rationale, and metadata
        """
        sample = super()._format_sample(row)

        # Add raw data
        sample["x_axis"] = row["x_axis"]
        sample["y_axis"] = row["y_axis"]
        sample["z_axis"] = row["z_axis"]

        # Add metadata
        sample["task_type"] = row.get("task_type", "unknown")
        sample["answer_type"] = row.get("answer_type", "unknown")
        sample["context_length_samples"] = row.get("context_length_samples", 0)
        sample["question"] = row.get("question", "")

        # Keep the direct answer for evaluation
        sample["direct_answer"] = row.get("answer", "")

        # Keep needles and difficulty_config for analysis
        sample["needles"] = row.get("needles", "[]")
        sample["difficulty_config"] = row.get("difficulty_config", "{}")

        return sample


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack CoT QADataset Demo")
    print("=" * 60)

    # Configuration
    tasks = ["existence"]
    context_lengths = [100]

    print(f"\nConfiguration:")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths (seconds): {context_lengths}")

    # Create datasets
    print("\nCreating CoT datasets...")
    try:
        dataset = TSHaystackCoTQADataset(
            split="train",
            EOS_TOKEN="",
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )

        print(f"\nDataset size: {len(dataset)}")
        print(f"Loaded tasks: {dataset.get_tasks()}")
        print(f"Context lengths: {dataset.get_context_lengths()}")

        # Test sample access
        if len(dataset) > 0:
            print("\n" + "=" * 50)
            print("Sample from training set:")
            sample = dataset[0]
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Task type: {sample.get('task_type', 'N/A')}")
            print(f"  Question: {sample.get('question', 'N/A')}")
            print(f"  Direct answer: {sample.get('direct_answer', 'N/A')}")

            # Show rationale preview
            answer = sample.get("answer", "")
            if len(answer) > 200:
                print(f"  Rationale (truncated): {answer[:200]}...")
            else:
                print(f"  Rationale: {answer}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure TS-Haystack CoT datasets have been generated first.")
        print("The CoT datasets require:")
        print("  1. Base datasets: run generate_ts_haystack_dataset.py")
        print("  2. CoT rationales: run generate_ts_haystack_cot.py")

    print("\n" + "=" * 60)
    print("CoT QADataset demo complete!")
    print("=" * 60)
