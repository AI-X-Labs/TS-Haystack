# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ts_haystack.capture24.loader import (
    CAPTURE24_DATA_DIR,
    get_sensor_data_dir,
    load_participant_sensor_data,
    load_participants,
)


# ---------------------------
# Constants
# ---------------------------

WINDOWS_DIR = os.path.join(CAPTURE24_DATA_DIR, "windows")


# ---------------------------
# Helper Functions
# ---------------------------

def format_window_size(window_size_s: float) -> str:
    """
    Format window size for directory naming.

    Args:
        window_size_s: Window size in seconds (can be float like 2.56)

    Returns:
        Formatted string suitable for directory names (e.g., "10s" or "2_56s")
    """
    if window_size_s == int(window_size_s):
        return f"{int(window_size_s)}s"
    else:
        # Replace decimal point with underscore for filesystem compatibility
        return f"{window_size_s}s".replace(".", "_")


def get_windows_path(window_size_s: float, effective_hz: int) -> Path:
    """
    Get path for windows directory based on configuration.

    Args:
        window_size_s: Window size in seconds (can be float like 2.56)
        effective_hz: Effective sampling frequency in Hz

    Returns:
        Path to windows directory
    """
    window_str = format_window_size(window_size_s)
    dir_name = f"{window_str}_{effective_hz}hz"
    return Path(WINDOWS_DIR) / dir_name


def split_participants(
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    source_hz: int = 100
) -> tuple[list[str], list[str], list[str]]:
    """
    Split participants into train/val/test sets with pure random assignment.
    Only includes participants with extracted sensor data.

    Args:
        seed: Random seed for reproducibility
        train_ratio: Fraction of participants for training (default: 0.7)
        val_ratio: Fraction of participants for validation (default: 0.15)
        source_hz: Source data sampling frequency in Hz (default: 100)

    Returns:
        Tuple of (train_pids, val_pids, test_pids)
    """
    participants = load_participants()
    pids = participants["pid"].to_list()

    # Filter to only participants with extracted sensor data
    sensor_dir = Path(get_sensor_data_dir(source_hz))
    existing_pids = []
    for pid in pids:
        participant_path = sensor_dir / f"pid={pid}" / "data.parquet"
        if participant_path.exists():
            existing_pids.append(pid)
    pids = existing_pids

    # Shuffle with fixed seed
    rng = np.random.default_rng(seed)
    pids_shuffled = pids.copy()
    rng.shuffle(pids_shuffled)

    # Calculate split sizes based on percentages
    n_total = len(pids_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = pids_shuffled[:n_train]
    val_pids = pids_shuffled[n_train:n_train + n_val]
    test_pids = pids_shuffled[n_train + n_val:]

    return train_pids, val_pids, test_pids


def downsample_sensor_data(
    data: pl.DataFrame,
    source_hz: int,
    target_hz: int
) -> pl.DataFrame:
    """
    Downsample sensor data from source frequency to target frequency.

    Args:
        data: Sensor data DataFrame
        source_hz: Source sampling frequency
        target_hz: Target sampling frequency

    Returns:
        Downsampled DataFrame
    """
    if target_hz >= source_hz:
        return data

    downsample_factor = source_hz // target_hz
    return data.with_row_index("idx").filter(
        pl.col("idx") % downsample_factor == 0
    ).drop("idx")


def extract_participant_windows(
    pid: str,
    window_size_s: float,
    source_hz: int = 100,
    downsample_hz: Optional[int] = None,
    annotation_threshold: float = 0.6
) -> pl.DataFrame:
    """
    Extract non-overlapping windows from a single participant's sensor data.

    Args:
        pid: Participant ID
        window_size_s: Window size in seconds
        source_hz: Source data sampling frequency in Hz (default: 100)
        downsample_hz: Optional target sampling frequency (None = no downsampling)
        annotation_threshold: Minimum fraction of samples that must have annotations (0.0-1.0)

    Returns:
        DataFrame with window schema:
            - window_id: string
            - pid: string
            - start_ms: int64
            - end_ms: int64
            - x: list[float32]
            - y: list[float32]
            - z: list[float32]
            - annotations: list[string]
    """
    # Load participant data
    data = load_participant_sensor_data(pid, downsample_hz=source_hz)

    if len(data) == 0:
        return pl.DataFrame(schema={
            "window_id": pl.String,
            "pid": pl.String,
            "start_ms": pl.Int64,
            "end_ms": pl.Int64,
            "x": pl.List(pl.Float32),
            "y": pl.List(pl.Float32),
            "z": pl.List(pl.Float32),
            "annotations": pl.List(pl.String),
        })

    # Downsample if requested
    if downsample_hz is not None and downsample_hz < source_hz:
        data = downsample_sensor_data(data, source_hz, downsample_hz)
        effective_hz = downsample_hz
    else:
        effective_hz = source_hz

    # Calculate window size in samples
    window_samples_float = window_size_s * effective_hz # use window 2.56 and effective_hz = 50Hz to match the existing HAR_CoT dataset at 128 samples per time series.
    assert window_samples_float == int(window_samples_float), (
        f"window_size_s ({window_size_s}) * effective_hz ({effective_hz}) = {window_samples_float} "
        f"must be a whole number of samples"
    )
    window_samples = int(window_samples_float)

    # Calculate number of complete windows
    n_samples = len(data)
    n_windows = n_samples // window_samples

    if n_windows == 0:
        return pl.DataFrame(schema={
            "window_id": pl.String,
            "pid": pl.String,
            "start_ms": pl.Int64,
            "end_ms": pl.Int64,
            "x": pl.List(pl.Float32),
            "y": pl.List(pl.Float32),
            "z": pl.List(pl.Float32),
            "annotations": pl.List(pl.String),
        })

    # Truncate to complete windows only
    data_windowed = data.head(n_windows * window_samples)

    # Create window index column
    data_windowed = data_windowed.with_columns(
        (pl.arange(0, len(data_windowed)) // window_samples).alias("window_idx")
    )

    # Group by window and aggregate
    windows = data_windowed.group_by("window_idx").agg([
        pl.col("timestamp_ms").first().alias("start_ms"),
        pl.col("timestamp_ms").last().alias("end_ms"),
        pl.col("x").cast(pl.Float32).alias("x"),
        pl.col("y").cast(pl.Float32).alias("y"),
        pl.col("z").cast(pl.Float32).alias("z"),
        pl.col("annotation").alias("annotations"),
    ]).sort("window_idx")

    # Filter windows by annotation threshold
    windows = windows.with_columns(
        (pl.col("annotations").list.eval(pl.element().is_not_null()).list.sum() /
         pl.col("annotations").list.len()).alias("annotation_ratio")
    ).filter(
        pl.col("annotation_ratio") >= annotation_threshold
    ).drop("annotation_ratio")

    # Create window IDs
    windows = windows.with_columns(
        (pl.lit(f"{pid}_") + pl.col("start_ms").cast(pl.String)).alias("window_id"),
        pl.lit(pid).alias("pid")
    ).select([
        "window_id",
        "pid",
        "start_ms",
        "end_ms",
        "x",
        "y",
        "z",
        "annotations"
    ])

    return windows


def extract_windows(
    window_size_s: float = 2.56,
    source_hz: int = 100,
    downsample_hz: Optional[int] = 50,
    annotation_threshold: float = 0.6,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_participants: Optional[int] = None,
    overwrite: bool = False
) -> None:
    """
    Extract non-overlapping windows from all participants and save to train/val/test splits.

    Args:
        window_size_s: Window size in seconds (default: 2.56, match HAR CoT)
        source_hz: Source data sampling frequency in Hz (default: 2.56, match HAR CoT)
        downsample_hz: Optional target sampling frequency (None = no downsampling)
        annotation_threshold: Minimum fraction of samples that must have annotations (default: 0.6)
        seed: Random seed for participant split (default: 42)
        train_ratio: Fraction of participants for training (default: 0.7)
        val_ratio: Fraction of participants for validation (default: 0.15)
        max_participants: Optional limit on number of participants (for testing)
        overwrite: Force re-extraction even if windows exist (default: False)

    Output structure:
        data/capture24/windows/{window_size_s}s_{downsample_hz}hz/
        ├── train/data.parquet
        ├── val/data.parquet
        └── test/data.parquet
    """
    # Validate source data exists
    source_data_dir = Path(get_sensor_data_dir(source_hz))
    if not source_data_dir.exists():
        raise FileNotFoundError(
            f"Source data directory not found: {source_data_dir}\n"
            f"Expected sensor data at {source_hz}Hz. "
            f"Run ensure_capture24_data(downsample_hz={source_hz}) first."
        )

    # Determine effective Hz for window path
    effective_hz = downsample_hz if downsample_hz is not None else source_hz
    windows_path = get_windows_path(window_size_s, effective_hz)

    # Check if windows already exist
    train_path = windows_path / "train" / "data.parquet"
    val_path = windows_path / "val" / "data.parquet"
    test_path = windows_path / "test" / "data.parquet"

    if not overwrite and all(p.exists() for p in [train_path, val_path, test_path]):
        print(f"Windows already exist at {windows_path}")
        return

    if overwrite:
        print(f"Overwrite flag set, re-extracting windows...")
    else:
        print(f"Extracting windows to {windows_path}...")

    # Create output directories
    for split in ["train", "val", "test"]:
        (windows_path / split).mkdir(parents=True, exist_ok=True)

    # Split participants
    train_pids, val_pids, test_pids = split_participants(seed, train_ratio, val_ratio, source_hz)

    total_pids = len(train_pids) + len(val_pids) + len(test_pids)
    if total_pids == 0:
        raise ValueError(
            f"No participants found in {source_data_dir}\n"
            f"The directory exists but contains no participant data. "
            f"Run ensure_capture24_data(downsample_hz={source_hz}) to extract data."
        )

    # Limit participants if requested (for testing)
    if max_participants is not None:
        all_pids = train_pids + val_pids + test_pids
        all_pids = all_pids[:max_participants]

        # Re-split the limited set
        n_total = len(all_pids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_pids = all_pids[:n_train]
        val_pids = all_pids[n_train:n_train + n_val]
        test_pids = all_pids[n_train + n_val:]

    print(f"Participant split: {len(train_pids)} train, {len(val_pids)} val, {len(test_pids)} test")

    # Process each split
    for split_name, pids in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        print(f"\nProcessing {split_name} split ({len(pids)} participants)...")

        # Extract windows for all participants
        all_windows = [
            extract_participant_windows(
                pid,
                window_size_s,
                source_hz,
                downsample_hz,
                annotation_threshold
            )
            for pid in tqdm(pids, desc=f"  {split_name}")
        ]

        # Combine all windows
        if all_windows:
            combined_windows = pl.concat(all_windows, how="vertical")

            # Save to parquet
            output_path = windows_path / split_name / "data.parquet"
            table = combined_windows.to_arrow()
            pq.write_table(
                table,
                output_path,
                compression='snappy',
                use_dictionary=True
            )

            print(f"  {split_name}: {len(combined_windows):,} windows saved")
        else:
            print(f"  ! {split_name}: No windows extracted")

    print(f"\nWindow extraction complete!")
    print(f"  Window size: {window_size_s}s")
    print(f"  Source sampling rate: {source_hz}Hz")
    if downsample_hz is not None:
        print(f"  Downsampled to: {downsample_hz}Hz")
    else:
        print(f"  No downsampling applied")
    print(f"  Annotation threshold: {annotation_threshold:.0%}")
    print(f"  Output: {windows_path}")


def load_windows(
    window_size_s: float,
    effective_hz: int = 100,
    split: str = "train"
) -> pl.DataFrame:
    """
    Load pre-extracted windows from parquet.

    Args:
        window_size_s: Window size in seconds
        effective_hz: Effective sampling frequency in Hz (default: 100)
        split: One of "train", "val", or "test"

    Returns:
        DataFrame with window schema
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', 'test'")

    windows_path = get_windows_path(window_size_s, effective_hz)
    data_path = windows_path / split / "data.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Windows not found at {data_path}. "
            f"Run extract_windows(window_size_s={window_size_s}, source_hz=..., downsample_hz=...) first."
        )

    return pl.read_parquet(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract windows from Capture-24 sensor data"
    )
    parser.add_argument(
        '--window-size-s', '-w',
        type=float,
        default=2.56,
        help='Window size in seconds (default: 2.56 to match HAR CoT)'
    )
    parser.add_argument(
        '--source-hz',
        type=int,
        default=100,
        help='Source data sampling frequency in Hz (default: 100)'
    )
    parser.add_argument(
        '--downsample-hz', '-d',
        type=int,
        default=50,
        help='Target sampling frequency in Hz (default: 50 to match HAR CoT)'
    )
    parser.add_argument(
        '--annotation-threshold', '-a',
        type=float,
        default=0.6,
        help='Minimum fraction of samples with annotations (default: 0.6)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for participant split (default: 42)'
    )
    parser.add_argument(
        '--train-ratio', '-t',
        type=float,
        default=0.7,
        help='Fraction of participants for training (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio', '-v',
        type=float,
        default=0.15,
        help='Fraction of participants for validation (default: 0.15)'
    )
    parser.add_argument(
        '--max-participants', '-n',
        type=int,
        default=None,
        help='Maximum number of participants to process (for testing)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Force re-extraction even if windows exist'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Capture-24 Window Extraction")
    print("=" * 60)

    print(f"Window size: {args.window_size_s}s")
    print(f"Source sampling rate: {args.source_hz}Hz")
    if args.downsample_hz is not None:
        print(f"Downsampling to: {args.downsample_hz}Hz")
    else:
        print(f"No downsampling")
    print(f"Annotation threshold: {args.annotation_threshold:.0%}")
    print(f"Random seed: {args.seed}")

    if args.max_participants is not None:
        print(f"Limiting to {args.max_participants} participants (testing mode)")

    print()

    # Extract windows
    extract_windows(
        window_size_s=args.window_size_s,
        source_hz=args.source_hz,
        downsample_hz=args.downsample_hz,
        annotation_threshold=args.annotation_threshold,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_participants=args.max_participants,
        overwrite=args.overwrite
    )

    # Quick verification
    print("\nVerifying extraction...")
    effective_hz = args.downsample_hz if args.downsample_hz is not None else args.source_hz
    for split in ["train", "val", "test"]:
        try:
            windows = load_windows(args.window_size_s, effective_hz, split)
            print(f"{split}: {len(windows):,} windows")
            if len(windows) > 0:
                sample_x_len = len(windows["x"][0])
                sample_ann_len = len(windows["annotations"][0])
                print(f"  Sample window: {sample_x_len} sensor samples, {sample_ann_len} annotations")
        except FileNotFoundError as e:
            print(f"! {split}: Not found")

    print("\n" + "=" * 60)
    print("Window extraction complete!")
    print("=" * 60)
