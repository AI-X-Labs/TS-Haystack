# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Capture-24 QADataset implementation for OpenTSLM training.

This module provides a QADataset subclass that formats Capture-24 accelerometer
classification data for use with the OpenTSLM Flamingo training pipeline.
"""

import re
import warnings
from typing import Callable, List, Literal, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.datasets.qa_base import QADataset
from src.datasets.capture24.qa_loader import (
    get_label_list,
    load_capture24_classification_splits,
)
from src.datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

# Axis labels matching HAR format
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]


class Capture24AccQADataset(QADataset):
    """
    QADataset subclass for Capture-24 accelerometer activity classification.

    This class loads Capture-24 classification data and formats it for training
    with the OpenTSLM Flamingo model. It follows the same pattern as HARAccQADataset.

    Note on caching:
        The QADataset base class uses class-level caching. Once data is loaded
        for a specific configuration, it's cached for all subsequent instances.
        If you need different configurations (window_size_s, effective_hz, label_scheme),
        you must restart the Python session or use a fresh interpreter.

    Args:
        split: Dataset split to load ("train", "test", or "validation")
        EOS_TOKEN: End-of-sequence token for the model
        window_size_s: Window size in seconds (default: 10)
        effective_hz: Effective sampling frequency in Hz (default: 100)
        label_scheme: Label scheme to use (default: "Walmsley2020")
        format_sample_str: If True, format samples as strings (default: False)
        time_series_format_function: Optional function to format time series as strings

    Example:
        >>> dataset = Capture24AccQADataset(
        ...     split="train",
        ...     EOS_TOKEN="",
        ...     window_size_s=10,
        ...     effective_hz=100,
        ...     label_scheme="Walmsley2020"
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Answer: {sample['answer']}")
    """

    # Class-level storage for configuration caching
    _cached_config: tuple = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        window_size_s: int = 10,
        effective_hz: int = 100,
        label_scheme: str = "Walmsley2020",
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        # Store config BEFORE calling super().__init__() because
        # _load_splits() is called in the parent constructor
        self.window_size_s = window_size_s
        self.effective_hz = effective_hz
        self.label_scheme = label_scheme

        # Check for configuration mismatch with cached data
        self._check_cache_config()

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _check_cache_config(self) -> None:
        """
        Check if cached data exists with different configuration.

        Warns the user if they're creating an instance with different parameters
        than the originally cached data.
        """
        current_config = (self.window_size_s, self.effective_hz, self.label_scheme)

        if self.__class__._cached_config is None:
            # First time loading - store the configuration
            self.__class__._cached_config = current_config
        elif self.__class__._cached_config != current_config:
            # Configuration mismatch - warn user
            warnings.warn(
                f"Capture24AccQADataset was previously loaded with config "
                f"(window_size_s={self.__class__._cached_config[0]}, "
                f"effective_hz={self.__class__._cached_config[1]}, "
                f"label_scheme='{self.__class__._cached_config[2]}'). "
                f"Now instantiated with (window_size_s={current_config[0]}, "
                f"effective_hz={current_config[1]}, "
                f"label_scheme='{current_config[2]}'). "
                f"Using cached data from original config. "
                f"Restart Python session for different configuration.",
                UserWarning,
                stacklevel=3,
            )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the Capture-24 classification dataset splits.

        Returns:
            Tuple of (train, validation, test) HuggingFace Dataset objects
        """
        return load_capture24_classification_splits(
            window_size_s=self.window_size_s,
            effective_hz=self.effective_hz,
            label_scheme=self.label_scheme,
        )

    def _get_answer(self, row) -> str:
        """Return the label as the answer."""
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        """Return the instruction text before the time series."""
        return (
            "You are given accelerometer data in all three dimensions from a "
            "wrist-worn sensor. Your task is to predict the person's activity."
        )

    def _get_post_prompt(self, _row) -> str:
        """Return the instruction text after the time series."""
        activities = ", ".join(self.get_labels())
        return f"""
Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any class label until the final sentence.
The following activities (class labels) are possible: {activities}
- You MUST end your response with "Answer: <class label>"
"""

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into TextTimeSeriesPrompt objects.

        Args:
            row: Dataset row containing x_axis, y_axis, z_axis

        Returns:
            List of TextTimeSeriesPrompt objects, one per axis
        """
        series = torch.tensor(
            [
                row["x_axis"],
                row["y_axis"],
                row["z_axis"],
            ],
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

    def extract_answer(self, prediction: str, sample: dict) -> str:
        # Look for "Answer: <label>" pattern, fall back to last line
        match = re.search(r"answer:\s*(.+)", prediction, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".,;:!?")
        return prediction.strip().split("\n")[-1].strip().rstrip(".,;:!?")

    def evaluate_answer(self, prediction: str, sample: dict) -> dict:
        gt = sample.get("label", sample.get("answer", ""))
        gt = gt.replace(self.EOS_TOKEN, "").strip().lower()
        pred = prediction.strip().lower()
        return {"correct": gt == pred}

    def get_labels(self) -> List[str]:
        """
        Return the labels for the configured label scheme.

        Returns:
            Alphabetically sorted list of class labels
        """
        return get_label_list(self.label_scheme)

    def _format_sample(self, row):
        """
        Format a sample for training, including raw data.

        Overrides the base class method to include raw accelerometer data
        and label in the formatted sample.

        Args:
            row: Dataset row

        Returns:
            Dictionary with formatted prompt and raw data
        """
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["x_axis"] = row["x_axis"]
        sample["y_axis"] = row["y_axis"]
        sample["z_axis"] = row["z_axis"]
        return sample


if __name__ == "__main__":
    print("=" * 60)
    print("Capture-24 QADataset Demo")
    print("=" * 60)

    # Configuration
    window_size_s = 10
    effective_hz = 100
    label_scheme = "Walmsley2020"

    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Label scheme: {label_scheme}")

    # Create datasets
    print("\nCreating datasets...")
    dataset = Capture24AccQADataset(
        split="train",
        EOS_TOKEN="",
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )
    dataset_val = Capture24AccQADataset(
        split="validation",
        EOS_TOKEN="",
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )
    dataset_test = Capture24AccQADataset(
        split="test",
        EOS_TOKEN="",
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )

    print(
        f"\nDataset sizes: Train: {len(dataset)}, "
        f"Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
    )

    # Show labels
    print(f"\nLabels ({label_scheme}): {dataset.get_labels()}")

    # Test sample access
    if len(dataset) > 0:
        print("\n" + "=" * 50)
        print("Sample from training set:")
        sample = dataset[0]
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Label: {sample['label']}")
        if "time_series" in sample:
            print(f"  Time series count: {len(sample['time_series'])}")
            for i, ts in enumerate(sample["time_series"]):
                print(f"    Axis {i}: length={len(ts)}, first 3 values={ts[:3]}")

    # Test DataLoader integration
    if len(dataset_test) > 0:
        print("\n" + "=" * 50)
        print("Testing DataLoader integration...")

        dataloader = DataLoader(
            dataset_test,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=4
            ),
        )

        for batch in tqdm(dataloader, total=1, desc="Testing batch"):
            print(f"  Batch keys: {list(batch[0].keys())}")
            print(f"  Batch answer: {batch[0]['answer']}")
            break

    print("\n" + "=" * 60)
    print("QADataset demo complete!")
    print("=" * 60)
