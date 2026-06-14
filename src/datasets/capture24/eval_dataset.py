# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Capture-24 Evaluation QADataset for context length experiments.

This module provides a QADataset subclass that loads pre-sampled evaluation data
for testing how OpenTSLM performance scales with context length. The evaluation
data uses HAR-CoT labels (biking, sitting, standing, walking) for zero-shot
evaluation with the HAR-CoT trained Flamingo checkpoint.

The prompt includes ALL 8 HAR-CoT classes to match the training distribution.
"""

import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import polars as pl
import torch
from datasets import Dataset

from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.datasets.qa_base import QADataset

# Axis labels matching HAR format
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]

# Full HAR-CoT label set (used in prompts to match training distribution)
HAR_COT_LABELS = [
    "biking",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
    "walking_down",
    "walking_up",
]

# Default path for evaluation data
DEFAULT_EVAL_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "capture24" / "eval_context_length"


class Capture24EvalQADataset(QADataset):
    """
    QADataset subclass for Capture-24 context length evaluation.

    Loads pre-sampled evaluation data with HAR-CoT compatible labels for
    zero-shot evaluation. The prompt uses all 8 HAR-CoT classes to match
    the training distribution.

    Args:
        split: Dataset split to load (only "test" is supported for evaluation)
        EOS_TOKEN: End-of-sequence token for the model
        window_size_s: Window size in seconds (one of: 2.56, 10, 30, 60, 300, 900, 1800, 3600)
        effective_hz: Effective sampling frequency in Hz (default: 50)
        eval_data_dir: Directory containing evaluation datasets (default: data/capture24/eval_context_length)
        format_sample_str: If True, format samples as strings (default: False)
        time_series_format_function: Optional function to format time series as strings

    Example:
        >>> # Load 2.56s evaluation dataset (matches HAR-CoT training)
        >>> dataset = Capture24EvalQADataset(
        ...     split="test",
        ...     EOS_TOKEN="",
        ...     window_size_s=2.56,
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Answer: {sample['answer']}")  # One of: biking, sitting, standing, walking
    """

    # Class-level storage for configuration caching
    _cached_config: tuple = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        window_size_s: float = 2.56,
        effective_hz: int = 50,
        eval_data_dir: Optional[Path] = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        # Store config BEFORE calling super().__init__() because
        # _load_splits() is called in the parent constructor
        self.window_size_s = window_size_s
        self.effective_hz = effective_hz
        self.eval_data_dir = Path(eval_data_dir) if eval_data_dir else DEFAULT_EVAL_DIR

        # Validate split - only test is supported for evaluation
        if split != "test":
            raise ValueError(
                f"Capture24EvalQADataset only supports split='test' for evaluation. "
                f"Got: split='{split}'"
            )

        # Store the split for _load_splits to use
        self._requested_split = split

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the evaluation dataset.

        Only loads the test split since this is for evaluation only.

        Returns:
            Tuple of (empty, empty, test) Dataset objects
        """
        # Get the evaluation data path
        # Format window size without trailing .0 for whole numbers (e.g., 60 not 60.0)
        window_str = str(int(self.window_size_s)) if self.window_size_s == int(self.window_size_s) else str(self.window_size_s)
        eval_path = self.eval_data_dir / f"{window_str}s_{self.effective_hz}hz"
        parquet_path = eval_path / "test.parquet"
        metadata_path = eval_path / "metadata.json"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at {parquet_path}. "
                f"Run scripts/phase1_dataset_preparation.py first to create the dataset."
            )

        # Load the data
        df = pl.read_parquet(parquet_path)

        # Load metadata for verification
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            print(f"Loaded evaluation dataset: {self.window_size_s}s @ {self.effective_hz}Hz")
            print(f"  Samples: {len(df)}")
            print(f"  Classes: {metadata.get('class_distribution', {})}")

        # Rename columns to match QADataset expected format
        df_renamed = df.rename({
            "x": "x_axis",
            "y": "y_axis",
            "z": "z_axis",
        }).select(["x_axis", "y_axis", "z_axis", "label"])

        # Convert to HuggingFace Dataset
        pandas_df = df_renamed.to_pandas()
        test_dataset = Dataset.from_pandas(pandas_df)

        # Return empty datasets for train/val since this is eval only
        import pandas as pd
        empty_df = pd.DataFrame({
            "x_axis": pd.Series(dtype=object),
            "y_axis": pd.Series(dtype=object),
            "z_axis": pd.Series(dtype=object),
            "label": pd.Series(dtype=str),
        })
        empty_dataset = Dataset.from_pandas(empty_df)

        return (empty_dataset, empty_dataset, test_dataset)

    def _get_answer(self, row) -> str:
        """Return the label as the answer."""
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        """
        Return the instruction text before the time series.

        Matches HAR-CoT format exactly for zero-shot transfer.
        """
        activities = ", ".join(HAR_COT_LABELS)
        return f"""
        You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.

        Instructions:
        - Begin by analyzing the time series without assuming a specific label.
        - Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
        - Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
        - Do **not** mention any class label until the final sentence.

        Possible activity labels are:
        {activities}.

        - Make sure that your last word is the answer. You MUST end your response with "Answer: "
        """

    def _get_post_prompt(self, _row) -> str:
        """
        Return the instruction text after the time series.

        Matches HAR-CoT format exactly: just "Rationale:" to trigger reasoning.
        """
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into TextTimeSeriesPrompt objects.

        Matches HAR-CoT format: includes mean/std in labels and normalizes data.

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

        # Normalize the data (matching HAR-CoT format)
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True)

        # Handle zero or very small standard deviations
        min_std = 1e-6
        stds = torch.clamp(stds, min=min_std)

        series_norm = (series - means) / stds

        prompts = []
        for time_series_label, time_series, mean, std in zip(
            TIME_SERIES_LABELS,
            series_norm.tolist(),
            means.squeeze().tolist(),
            stds.squeeze().tolist()
        ):
            # Match HAR-CoT format: include mean and std in the label
            text_prompt = f"{time_series_label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, time_series))
        return prompts

    @staticmethod
    def get_labels() -> List[str]:
        """
        Return the full HAR-CoT label set for prompt generation.

        Note: Actual predictions will only be one of the 4 overlapping classes
        (biking, sitting, standing, walking), but we use all 8 in the prompt
        to match the training distribution.
        """
        return HAR_COT_LABELS

    def get_eval_labels(self) -> List[str]:
        """
        Return the 4-class subset used for evaluation.

        These are the only valid prediction classes for this evaluation.
        """
        return ["biking", "sitting", "standing", "walking"]

    def _format_sample(self, row):
        """
        Format a sample for evaluation, including raw data.

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


def load_all_eval_datasets(
    eval_data_dir: Optional[Path] = None,
    EOS_TOKEN: str = "",
) -> dict:
    """
    Load all available evaluation datasets for context length experiment.

    Args:
        eval_data_dir: Directory containing evaluation datasets
        EOS_TOKEN: End-of-sequence token for the model

    Returns:
        Dictionary mapping window_size_s to Capture24EvalQADataset
    """
    eval_dir = Path(eval_data_dir) if eval_data_dir else DEFAULT_EVAL_DIR

    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation data directory not found: {eval_dir}. "
            f"Run scripts/phase1_dataset_preparation.py first."
        )

    datasets = {}

    # Look for all window size directories
    for subdir in sorted(eval_dir.iterdir()):
        if subdir.is_dir() and (subdir / "test.parquet").exists():
            # Parse window size from directory name (e.g., "2.56s_50hz" or "10s_50hz")
            dir_name = subdir.name
            parts = dir_name.split("_")
            if len(parts) == 2:
                window_str = parts[0].replace("s", "").replace("_", ".")
                try:
                    window_size_s = float(window_str)
                    datasets[window_size_s] = Capture24EvalQADataset(
                        split="test",
                        EOS_TOKEN=EOS_TOKEN,
                        window_size_s=window_size_s,
                        eval_data_dir=eval_dir,
                    )
                except (ValueError, FileNotFoundError) as e:
                    print(f"Skipping {subdir}: {e}")

    return datasets


if __name__ == "__main__":
    print("=" * 60)
    print("Capture-24 Evaluation QADataset Demo")
    print("=" * 60)

    # Try to load the 2.56s dataset (should exist if Phase 1 has run)
    window_size_s = 2.56
    effective_hz = 50

    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Labels (prompt): {HAR_COT_LABELS}")

    try:
        dataset = Capture24EvalQADataset(
            split="test",
            EOS_TOKEN="",
            window_size_s=window_size_s,
            effective_hz=effective_hz,
        )

        print(f"\nDataset loaded successfully!")
        print(f"  Size: {len(dataset)}")
        print(f"  Eval labels: {dataset.get_eval_labels()}")

        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"Answer: {sample['answer']}")
            print(f"Label: {sample['label']}")

            if "time_series" in sample:
                for i, ts in enumerate(sample["time_series"]):
                    print(f"  Axis {i}: length={len(ts)}")

    except FileNotFoundError as e:
        print(f"\nDataset not found: {e}")
        print("Run scripts/phase1_dataset_preparation.py first to create the dataset.")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
