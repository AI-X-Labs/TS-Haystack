# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
TS-Haystack Oracle QADataset implementation for OpenTSLM training.

This module provides a QADataset subclass that includes ground-truth activity
segmentation in the prompts. This "oracle mode" isolates the LLM's language
reasoning capacity from time series perception - if the oracle achieves high
accuracy, it proves the questions are answerable given perfect perception.

Usage:
    from src.datasets.capture24_haystack import TSHaystackOracleQADataset

    # Oracle mode includes ground-truth activity timeline in prompts
    dataset = TSHaystackOracleQADataset(
        split="train",
        tasks=["existence"],
        context_lengths_seconds=[100],
    )

    # The pre-prompt will include the activity timeline
    sample = dataset[0]
    print(sample["prompt"])  # Contains oracle timeline
"""

from typing import Callable, List, Literal, Optional, Union

import numpy as np

from src.datasets.capture24_haystack.cot_qa_dataset import (
    TSHaystackCoTQADataset,
)
from src.datasets.capture24_haystack.utils.oracle_utils import (
    format_oracle_timeline,
)


class TSHaystackOracleQADataset(TSHaystackCoTQADataset):
    """
    QADataset subclass for TS-Haystack benchmark with oracle mode.

    This class extends TSHaystackCoTQADataset by prepending ground-truth
    activity segmentation to each prompt. The oracle timeline shows exactly
    when each activity occurred and which segments were inserted needles.

    This is useful for:
    1. Validating that questions are answerable given perfect perception
    2. Testing the LLM backbone's reasoning capacity in isolation
    3. Establishing an upper bound on task performance

    Note on caching:
        This class uses a separate cache from TSHaystackCoTQADataset to avoid
        conflicts when switching between oracle and non-oracle modes.

    Args:
        split: Dataset split to load ("train", "test", or "validation")
        EOS_TOKEN: End-of-sequence token appended to answers. Required.
                   Passed by the training script from ``model.get_eos_token()``.
        tasks: List of tasks to load (e.g., ["existence", "localization"])
               Use ["all"] to load all tasks
        context_lengths_seconds: List of context lengths in seconds
        cot_small: If True, use the cot_small dataset variant (smaller dataset)
        format_sample_str: If True, format samples as strings (default: False)
        time_series_format_function: Optional function to format time series
        lazy_loading: If True (default), samples are formatted on-demand to save memory.

    Example:
        >>> dataset = TSHaystackOracleQADataset(
        ...     split="train",
        ...     tasks=["existence", "counting"],
        ...     context_lengths_seconds=[100],
        ... )
        >>> sample = dataset[0]
        >>> # The prompt includes the oracle timeline before the question
        >>> print("Activity Timeline" in sample["prompt"])  # True
    """

    # Separate cache for oracle mode to avoid conflicts
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
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            tasks=tasks,
            context_lengths_seconds=context_lengths_seconds,
            cot_small=cot_small,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
            lazy_loading=lazy_loading,
        )

    def _get_pre_prompt(self, row) -> str:
        """
        Return the instruction text before the time series, including oracle timeline.

        The oracle timeline is prepended to the standard pre-prompt, providing
        ground-truth activity segmentation with timestamps and [inserted] markers.
        """
        # Get the standard pre-prompt from parent class
        standard_pre_prompt = super()._get_pre_prompt(row)

        # Build the oracle timeline
        oracle_timeline = format_oracle_timeline(
            needles=row.get("needles", "[]"),
            difficulty_config=row.get("difficulty_config", "{}"),
            recording_time_start=row.get("recording_time_start", "unknown"),
            recording_time_end=row.get("recording_time_end", "unknown"),
            context_length_samples=row.get("context_length_samples", 0),
        )

        # Prepend the oracle timeline with a separator
        return f"{oracle_timeline}\n\n{standard_pre_prompt}"


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack Oracle QADataset Demo")
    print("=" * 60)

    # Configuration
    tasks = ["existence"]
    context_lengths = [100]

    print(f"\nConfiguration:")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths (seconds): {context_lengths}")

    # Create datasets
    print("\nCreating Oracle dataset...")
    try:
        dataset = TSHaystackOracleQADataset(
            split="train",
            EOS_TOKEN="",
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )

        print(f"\nDataset size: {len(dataset)}")
        print(f"Loaded tasks: {dataset.get_tasks()}")

        # Test sample access
        if len(dataset) > 0:
            print("\n" + "=" * 50)
            print("Sample from training set:")
            sample = dataset[0]
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Task type: {sample.get('task_type', 'N/A')}")
            print(f"  Question: {sample.get('question', 'N/A')}")
            print(f"  Direct answer: {sample.get('direct_answer', 'N/A')}")

            # Show prompt preview (should include oracle timeline)
            prompt = sample.get("prompt", "")
            print("\n  Prompt preview (first 500 chars):")
            print("  " + "-" * 40)
            preview = prompt[:500].replace("\n", "\n  ")
            print(f"  {preview}...")

            # Check if oracle timeline is present
            has_timeline = "Activity Timeline (Ground Truth):" in prompt
            print("\n  " + "-" * 40)
            print(f"  Oracle timeline present: {has_timeline}")

    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        print("\nMake sure TS-Haystack CoT datasets have been generated first.")

    print("\n" + "=" * 60)
    print("Oracle QADataset demo complete!")
    print("=" * 60)
