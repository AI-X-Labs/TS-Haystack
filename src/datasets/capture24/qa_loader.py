# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Loader for Capture-24 classification data in HuggingFace Dataset format.

This module bridges Phase 2B classification parquet files to the QADataset interface
by converting them to HuggingFace Dataset objects with the expected schema.
"""

from typing import Dict, List, Tuple

import pandas as pd
from datasets import Dataset

from src.datasets.capture24.classification import (
    get_class_names,
    load_classification_dataset,
)


def load_capture24_classification_splits(
    window_size_s: int = 10,
    effective_hz: int = 100,
    label_scheme: str = "Walmsley2020",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Capture-24 classification data as HuggingFace Dataset objects.

    Converts the Phase 2B parquet files into the format expected by QADataset:
    - Renames columns: x -> x_axis, y -> y_axis, z -> z_axis
    - Keeps only required columns: x_axis, y_axis, z_axis, label

    Args:
        window_size_s: Window size in seconds (default: 10)
        effective_hz: Effective sampling frequency in Hz (default: 100)
        label_scheme: Label scheme name (default: "Walmsley2020")

    Returns:
        Tuple of (train, val, test) Dataset objects with schema:
        - x_axis: list[float]
        - y_axis: list[float]
        - z_axis: list[float]
        - label: str

    Raises:
        FileNotFoundError: If classification dataset doesn't exist
    """
    datasets = []

    for split in ["train", "val", "test"]:
        try:
            # Load from parquet using existing function
            df = load_classification_dataset(
                window_size_s=window_size_s,
                effective_hz=effective_hz,
                label_scheme=label_scheme,
                split=split,
            )

            # Rename columns to match HAR format and select required columns
            # Polars DataFrame -> rename -> select -> pandas -> HuggingFace Dataset
            df_renamed = df.rename({
                "x": "x_axis",
                "y": "y_axis",
                "z": "z_axis",
            }).select(["x_axis", "y_axis", "z_axis", "label"])

            # Convert to pandas, then to HuggingFace Dataset
            pandas_df = df_renamed.to_pandas()
            hf_dataset = Dataset.from_pandas(pandas_df)
            datasets.append(hf_dataset)

            print(f"Loaded {split}: {len(hf_dataset)} samples")

        except FileNotFoundError:
            print(f"Warning: {split} split not found, creating empty dataset")
            # Create empty dataset with correct schema
            empty_df = pd.DataFrame({
                "x_axis": pd.Series(dtype=object),
                "y_axis": pd.Series(dtype=object),
                "z_axis": pd.Series(dtype=object),
                "label": pd.Series(dtype=str),
            })
            datasets.append(Dataset.from_pandas(empty_df))

    return tuple(datasets)


def get_label_list(label_scheme: str) -> List[str]:
    """
    Return alphabetically sorted list of labels for a scheme.

    Args:
        label_scheme: Label scheme name (e.g., "Walmsley2020")

    Returns:
        Alphabetically sorted list of unique class names
    """
    return get_class_names(label_scheme)


def get_label_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get label distribution for a dataset.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Dictionary mapping labels to counts
    """
    if len(dataset) == 0:
        return {}
    labels = dataset["label"]
    return dict(pd.Series(labels).value_counts())


def print_dataset_info(dataset: Dataset, name: str) -> None:
    """
    Print information about a dataset split.

    Args:
        dataset: The dataset split
        name: Name of the split (e.g., "Train")
    """
    label_dist = get_label_distribution(dataset)
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")
    if label_dist:
        print(f"  Label distribution:")
        for label, count in sorted(label_dist.items()):
            print(f"    {label}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("Capture-24 QA Loader Demo")
    print("=" * 60)

    # Default configuration
    window_size_s = 10
    effective_hz = 100
    label_scheme = "Walmsley2020"

    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Label scheme: {label_scheme}")

    # Load the dataset splits
    print("\nLoading dataset splits...")
    train_ds, val_ds, test_ds = load_capture24_classification_splits(
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )

    # Print dataset information
    print_dataset_info(train_ds, "Train")
    print_dataset_info(val_ds, "Validation")
    print_dataset_info(test_ds, "Test")

    # Show sample data
    if len(train_ds) > 0:
        print("\n" + "=" * 50)
        print("Sample data from training set:")
        sample = train_ds[0]
        for key, value in sample.items():
            if key in ["x_axis", "y_axis", "z_axis"]:
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {value[:5]}... (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

    # Show labels
    print("\n" + "=" * 50)
    print(f"Labels for {label_scheme}:")
    labels = get_label_list(label_scheme)
    print(f"  {labels}")

    print("\n" + "=" * 60)
    print("Loader demo complete!")
    print("=" * 60)
