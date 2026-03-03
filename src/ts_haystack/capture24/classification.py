# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ts_haystack.capture24.loader import (
    CAPTURE24_DATA_DIR,
    load_label_mappings,
)
from ts_haystack.capture24.windows import (
    format_window_size,
    get_windows_path,
    load_windows,
)


# ---------------------------
# Constants
# ---------------------------

CLASSIFICATION_DIR = os.path.join(CAPTURE24_DATA_DIR, "classification")

# Available label schemes and their corresponding column names in the label mappings parquet
LABEL_SCHEMES = {
    "WillettsSpecific2018": "label_willetts_specific_2018",
    "WillettsMET2018": "label_willetts_met_2018",
    "DohertySpecific2018": "label_doherty_specific_2018",
    "Willetts2018": "label_willetts_2018",
    "Doherty2018": "label_doherty_2018",
    "Walmsley2020": "label_walmsley_2020",
}


# ---------------------------
# Helper Functions
# ---------------------------

def get_classification_path(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str
) -> Path:
    """
    Get path for classification dataset directory based on configuration.

    Args:
        window_size_s: Window size in seconds (can be float like 2.56)
        effective_hz: Effective sampling frequency in Hz
        label_scheme: Label scheme name (e.g., "Walmsley2020")

    Returns:
        Path to classification dataset directory
    """
    window_str = format_window_size(window_size_s)
    dir_name = f"{window_str}_{effective_hz}hz"
    return Path(CLASSIFICATION_DIR) / dir_name / label_scheme


def load_label_mapping(label_scheme: str) -> dict[str, str]:
    """
    Load annotation-to-label mapping for a specific label scheme.

    Args:
        label_scheme: Label scheme name (e.g., "Walmsley2020")

    Returns:
        Dictionary mapping raw annotations to labels
    """
    if label_scheme not in LABEL_SCHEMES:
        raise ValueError(
            f"Unknown label scheme: {label_scheme}. "
            f"Available schemes: {list(LABEL_SCHEMES.keys())}"
        )

    column_name = LABEL_SCHEMES[label_scheme]
    label_mappings = load_label_mappings()

    # Create mapping dict from annotation to label
    mapping = {}
    for row in label_mappings.iter_rows(named=True):
        annotation = row["annotation"]
        label = row[column_name]
        if annotation is not None and label is not None:
            mapping[annotation] = label

    return mapping


def get_class_names(label_scheme: str) -> list[str]:
    """
    Get ordered list of class names for a label scheme.

    Args:
        label_scheme: Label scheme name (e.g., "Walmsley2020")

    Returns:
        Alphabetically sorted list of unique class names
    """
    mapping = load_label_mapping(label_scheme)
    unique_labels = set(mapping.values())
    return sorted(unique_labels)


def get_window_label(
    annotations: list[str],
    mapping: dict[str, str]
) -> tuple[Optional[str], float]:
    """
    Compute the label for a window based on annotation mode (most frequent label).

    Args:
        annotations: List of raw annotation strings for each sample in the window
        mapping: Dictionary mapping raw annotations to labels

    Returns:
        Tuple of (mode_label, confidence) where:
        - mode_label: The most frequent label (or None if no valid annotations)
        - confidence: Fraction of samples with the mode label (0.0-1.0)
    """
    # Map annotations to labels, filtering out unmapped/null annotations
    mapped_labels = []
    for ann in annotations:
        if ann is not None and ann in mapping:
            mapped_labels.append(mapping[ann])

    if not mapped_labels:
        return None, 0.0

    # Count labels and find mode
    label_counts = Counter(mapped_labels)
    mode_label, mode_count = label_counts.most_common(1)[0]

    # Confidence is fraction of valid samples with the mode label
    confidence = mode_count / len(mapped_labels)

    return mode_label, confidence


def create_classification_dataset(
    window_size_s: float = 2.56,
    effective_hz: int = 50,
    label_scheme: str = "Walmsley2020",
    min_confidence: float = 0.0,
    overwrite: bool = False
) -> None:
    """
    Create classification dataset from pre-extracted windows.

    Converts raw annotation windows (Phase 2A) into labeled classification examples
    with a single label per window determined by mode (most frequent label).

    Args:
        window_size_s: Window size in seconds
        effective_hz: Effective sampling frequency in Hz
        label_scheme: Label scheme to use (default: "Walmsley2020")
        min_confidence: Minimum confidence threshold for including windows (default: 0.0)
        overwrite: Force re-creation even if dataset exists (default: False)

    Output structure:
        data/capture24/classification/{window_size_s}s_{effective_hz}hz/{label_scheme}/
        ├── train/data.parquet
        ├── val/data.parquet
        ├── test/data.parquet
        └── metadata.json
    """
    # Validate label scheme
    if label_scheme not in LABEL_SCHEMES:
        raise ValueError(
            f"Unknown label scheme: {label_scheme}. "
            f"Available schemes: {list(LABEL_SCHEMES.keys())}"
        )

    # Check if source windows exist
    windows_path = get_windows_path(window_size_s, effective_hz)
    if not windows_path.exists():
        raise FileNotFoundError(
            f"Windows not found at {windows_path}. "
            f"Run extract_windows(window_size_s={window_size_s}, ..., downsample_hz={effective_hz}) first."
        )

    # Get output path
    output_path = get_classification_path(window_size_s, effective_hz, label_scheme)

    # Check if already exists
    train_path = output_path / "train" / "data.parquet"
    val_path = output_path / "val" / "data.parquet"
    test_path = output_path / "test" / "data.parquet"
    metadata_path = output_path / "metadata.json"

    if not overwrite and all(p.exists() for p in [train_path, val_path, test_path, metadata_path]):
        print(f"Classification dataset already exists at {output_path}")
        return

    if overwrite:
        print(f"Overwrite flag set, re-creating classification dataset...")
    else:
        print(f"Creating classification dataset at {output_path}...")

    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Load label mapping
    mapping = load_label_mapping(label_scheme)
    class_names = get_class_names(label_scheme)
    label_to_id = {label: idx for idx, label in enumerate(class_names)}

    print(f"Label scheme: {label_scheme}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Minimum confidence: {min_confidence:.0%}")
    print()

    # Track statistics
    total_stats = {
        "total_windows": 0,
        "filtered_windows": 0,
        "class_counts": {split: Counter() for split in ["train", "val", "test"]}
    }

    # Process each split
    for split in ["train", "val", "test"]:
        print(f"Processing {split} split...")

        # Load source windows (skip if not found)
        try:
            windows = load_windows(window_size_s, effective_hz, split)
        except FileNotFoundError:
            print(f"  ! No source windows found for {split}, skipping")
            continue

        total_stats["total_windows"] += len(windows)

        if len(windows) == 0:
            print(f"  ! No windows found for {split}")
            continue

        # Process each window to compute label and confidence
        results = []
        for row in tqdm(windows.iter_rows(named=True), total=len(windows), desc=f"  {split}"):
            annotations = row["annotations"]
            label, confidence = get_window_label(annotations, mapping)

            # Skip if no valid label or below confidence threshold
            if label is None or confidence < min_confidence:
                total_stats["filtered_windows"] += 1
                continue

            label_id = label_to_id[label]

            results.append({
                "window_id": row["window_id"],
                "pid": row["pid"],
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "x": row["x"],
                "y": row["y"],
                "z": row["z"],
                "label": label,
                "label_id": label_id,
                "confidence": confidence,
            })

            total_stats["class_counts"][split][label] += 1

        if not results:
            print(f"  ! No windows passed filtering for {split}")
            continue

        # Create DataFrame
        df = pl.DataFrame(results)

        # Ensure correct types
        df = df.with_columns([
            pl.col("label_id").cast(pl.Int32),
            pl.col("confidence").cast(pl.Float32),
        ])

        # Save to parquet
        split_output_path = output_path / split / "data.parquet"
        table = df.to_arrow()
        pq.write_table(
            table,
            split_output_path,
            compression='snappy',
            use_dictionary=True
        )

        print(f"  {split}: {len(df):,} windows saved")

    # Save metadata
    metadata = {
        "label_scheme": label_scheme,
        "window_size_s": window_size_s,
        "effective_hz": effective_hz,
        "min_confidence": min_confidence,
        "class_names": class_names,
        "num_classes": len(class_names),
        "label_to_id": label_to_id,
        "total_windows_processed": total_stats["total_windows"],
        "windows_filtered": total_stats["filtered_windows"],
        "class_distribution": {
            split: dict(counts) for split, counts in total_stats["class_counts"].items()
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nClassification dataset created!")
    print(f"  Total windows processed: {total_stats['total_windows']:,}")
    print(f"  Windows filtered (low confidence or unmapped): {total_stats['filtered_windows']:,}")
    print(f"  Output: {output_path}")

    # Print class distribution summary
    print(f"\nClass distribution:")
    for split in ["train", "val", "test"]:
        counts = total_stats["class_counts"][split]
        if counts:
            total = sum(counts.values())
            print(f"  {split}:")
            for class_name in class_names:
                count = counts.get(class_name, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"    {class_name}: {count:,} ({pct:.1f}%)")


def load_classification_dataset(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
    split: str
) -> pl.DataFrame:
    """
    Load pre-created classification dataset.

    Args:
        window_size_s: Window size in seconds
        effective_hz: Effective sampling frequency in Hz
        label_scheme: Label scheme name (e.g., "Walmsley2020")
        split: One of "train", "val", or "test"

    Returns:
        DataFrame with classification schema:
            - window_id: string
            - pid: string
            - start_ms: int64
            - end_ms: int64
            - x: list[float32]
            - y: list[float32]
            - z: list[float32]
            - label: string
            - label_id: int32
            - confidence: float32
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'")

    dataset_path = get_classification_path(window_size_s, effective_hz, label_scheme)
    data_path = dataset_path / split / "data.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Classification dataset not found at {data_path}. "
            f"Run create_classification_dataset(window_size_s={window_size_s}, "
            f"effective_hz={effective_hz}, label_scheme='{label_scheme}') first."
        )

    return pl.read_parquet(data_path)


def load_classification_metadata(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str
) -> dict:
    """
    Load metadata for a classification dataset.

    Args:
        window_size_s: Window size in seconds
        effective_hz: Effective sampling frequency in Hz
        label_scheme: Label scheme name

    Returns:
        Dictionary with metadata including class_names, label_to_id, etc.
    """
    dataset_path = get_classification_path(window_size_s, effective_hz, label_scheme)
    metadata_path = dataset_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            f"Run create_classification_dataset() first."
        )

    with open(metadata_path, 'r') as f:
        return json.load(f)


def get_class_distribution(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
    split: str
) -> dict[str, int]:
    """
    Get class distribution for a classification dataset split.

    Args:
        window_size_s: Window size in seconds
        effective_hz: Effective sampling frequency in Hz
        label_scheme: Label scheme name
        split: One of "train", "val", or "test"

    Returns:
        Dictionary mapping class names to counts
    """
    metadata = load_classification_metadata(window_size_s, effective_hz, label_scheme)
    return metadata["class_distribution"].get(split, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create classification dataset from Capture-24 windows"
    )
    parser.add_argument(
        '--window-size-s', '-w',
        type=float,
        default=2.56,
        help='Window size in seconds (default: 2.56 to match HAR CoT)'
    )
    parser.add_argument(
        '--effective-hz', '-e',
        type=int,
        default=50,
        help='Effective sampling frequency in Hz (default: 50 to match HAR CoT)'
    )
    parser.add_argument(
        '--label-scheme', '-l',
        type=str,
        default="Walmsley2020",
        choices=list(LABEL_SCHEMES.keys()),
        help='Label scheme to use (default: Walmsley2020)'
    )
    parser.add_argument(
        '--min-confidence', '-c',
        type=float,
        default=0.6,
        help='Minimum confidence threshold (0.0-1.0, default: 0.0)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Force re-creation even if dataset exists'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Capture-24 Classification Dataset Creation")
    print("=" * 60)

    print(f"Window size: {args.window_size_s}s")
    print(f"Effective sampling rate: {args.effective_hz}Hz")
    print(f"Label scheme: {args.label_scheme}")
    print(f"Minimum confidence: {args.min_confidence:.0%}")
    print()

    # Create classification dataset
    create_classification_dataset(
        window_size_s=args.window_size_s,
        effective_hz=args.effective_hz,
        label_scheme=args.label_scheme,
        min_confidence=args.min_confidence,
        overwrite=args.overwrite
    )

    # Quick verification
    print("\nVerifying creation...")
    for split in ["train", "val", "test"]:
        try:
            df = load_classification_dataset(
                args.window_size_s,
                args.effective_hz,
                args.label_scheme,
                split
            )
            print(f"{split}: {len(df):,} windows")
            if len(df) > 0:
                sample_x_len = len(df["x"][0])
                labels = df["label"].unique().sort()
                print(f"  Sensor samples per window: {sample_x_len}")
                print(f"  Labels: {labels.to_list()}")
        except FileNotFoundError:
            print(f"! {split}: Not found")

    print("\n" + "=" * 60)
    print("Classification dataset creation complete!")
    print("=" * 60)
