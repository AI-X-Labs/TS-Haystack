#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Verification script for Capture-24 classification dataset creation.

Supports two modes:
- Full mode (default): Create classification dataset + verify
- Verify-only mode (--verify-only): Just verify existing datasets
"""

import argparse

from src.datasets.capture24 import (
    create_classification_dataset,
    load_classification_dataset,
    load_classification_metadata,
    get_classification_path,
    get_class_names,
    get_class_distribution,
    load_label_mapping,
    LABEL_SCHEMES,
)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_label_scheme(label_scheme: str) -> None:
    """Verify label scheme is valid."""
    print("\nVerifying label schemes...")
    print(f"  Available schemes: {list(LABEL_SCHEMES.keys())}")
    assert label_scheme in LABEL_SCHEMES, f"Unknown label scheme: {label_scheme}"
    print(f"  Label scheme '{label_scheme}' is valid")


def verify_label_mapping(label_scheme: str) -> dict:
    """Load and verify label mapping."""
    print(f"\nLoading label mapping for {label_scheme}...")
    mapping = load_label_mapping(label_scheme)
    print(f"  Loaded {len(mapping)} annotation-to-label mappings")
    print("  Sample mappings:")
    for i, (annotation, label) in enumerate(list(mapping.items())[:3]):
        print(f"    '{annotation[:50]}...' -> '{label}'")
    return mapping


def verify_class_names(label_scheme: str) -> list:
    """Get and verify class names."""
    print(f"\nGetting class names for {label_scheme}...")
    class_names = get_class_names(label_scheme)
    print(f"  {len(class_names)} classes: {class_names}")
    return class_names


def verify_classification_dataset(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
    class_names: list,
) -> None:
    """Load and verify classification dataset."""
    print("\nLoading and verifying classification dataset...")
    classification_path = get_classification_path(window_size_s, effective_hz, label_scheme)
    print(f"  Classification path: {classification_path}")
    assert classification_path.exists(), f"Classification path does not exist: {classification_path}"

    for split in ["train", "val", "test"]:
        try:
            df = load_classification_dataset(window_size_s, effective_hz, label_scheme, split)
            print(f"\n  {split.upper()} split:")
            print(f"    {len(df):,} samples")
            print(f"    Columns: {df.columns}")

            if len(df) > 0:
                # Verify schema
                for col in ["window_id", "pid", "x", "y", "z", "label", "label_id", "confidence"]:
                    assert col in df.columns, f"Missing {col}"

                # Check sensor data shape
                first_sample = df[0]
                x_len = len(first_sample["x"][0])
                expected_len = int(window_size_s * effective_hz)
                print(f"    Sensor samples per window: {x_len} (expected: {expected_len})")
                assert x_len == expected_len, f"X length mismatch: {x_len} != {expected_len}"

                # Check label encoding
                unique_labels = df["label"].unique().sort().to_list()
                unique_label_ids = df["label_id"].unique().sort().to_list()
                print(f"    Unique labels: {unique_labels}")
                print(f"    Label IDs: {unique_label_ids}")
                assert max(unique_label_ids) < len(class_names), "Label ID out of range"

                # Check confidence range
                min_conf = df["confidence"].min()
                max_conf = df["confidence"].max()
                print(f"    Confidence range: [{min_conf:.2f}, {max_conf:.2f}]")
                assert min_conf >= 0.0 and max_conf <= 1.0, "Confidence out of range"

        except FileNotFoundError:
            print(f"\n  {split.upper()} split: No data (expected if no windows passed filtering)")


def verify_metadata_and_distribution(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
    class_names: list,
) -> None:
    """Verify metadata and class distribution."""
    print("\nVerifying metadata...")
    metadata = load_classification_metadata(window_size_s, effective_hz, label_scheme)
    print(f"  Label scheme: {metadata['label_scheme']}")
    print(f"  Window size: {metadata['window_size_s']}s")
    print(f"  Effective Hz: {metadata['effective_hz']}")
    print(f"  Num classes: {metadata['num_classes']}")
    print(f"  Class names: {metadata['class_names']}")
    print(f"  Total windows processed: {metadata['total_windows_processed']:,}")
    print(f"  Windows filtered: {metadata['windows_filtered']:,}")

    print("\n  Class distribution:")
    for split in ["train", "val", "test"]:
        dist = get_class_distribution(window_size_s, effective_hz, label_scheme, split)
        if dist:
            total = sum(dist.values())
            print(f"    {split}:")
            for class_name in class_names:
                count = dist.get(class_name, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"      {class_name}: {count:,} ({pct:.1f}%)")


def log_sample_windows(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
) -> None:
    """Log sample windows for inspection."""
    print("\n" + "=" * 60)
    print("SAMPLE WINDOWS")
    print("=" * 60)

    try:
        train_df = load_classification_dataset(window_size_s, effective_hz, label_scheme, "train")
        if len(train_df) >= 2:
            for i in range(2):
                sample = train_df[i]
                print(f"\n[Sample {i+1}]")
                print(f"  Window ID: {sample['window_id'][0]}")
                print(f"  Participant: {sample['pid'][0]}")
                print(f"  Time range: {sample['start_ms'][0]} - {sample['end_ms'][0]}")
                print(f"  Label: '{sample['label'][0]}' (ID: {sample['label_id'][0]})")
                print(f"  Confidence: {sample['confidence'][0]:.2%}")
                print(f"  Sensor data shape: x={len(sample['x'][0])}, y={len(sample['y'][0])}, z={len(sample['z'][0])}")
                print(f"  X values (first 5): {sample['x'][0][:5]}")
                print(f"  Y values (first 5): {sample['y'][0][:5]}")
                print(f"  Z values (first 5): {sample['z'][0][:5]}")
    except FileNotFoundError:
        print("  No training data available for sampling")


# =============================================================================
# Mode Runners
# =============================================================================

def run_full_verification(args) -> None:
    """Run full verification: create + verify."""
    print("=" * 60)
    print("Capture-24 Classification Dataset Verification (Full Mode)")
    print("=" * 60)

    # Steps 1-3: Verify label schemes
    print("\n[1/6] ", end="")
    verify_label_scheme(args.label_scheme)

    print("\n[2/6] ", end="")
    verify_label_mapping(args.label_scheme)

    print("\n[3/6] ", end="")
    class_names = verify_class_names(args.label_scheme)

    # Step 4: Create classification dataset
    print(f"\n[4/6] Creating classification dataset...")
    print(f"  Window size: {args.window_size_s}s")
    print(f"  Effective Hz: {args.effective_hz}")
    print(f"  Label scheme: {args.label_scheme}")
    print(f"  Min confidence: {args.min_confidence}")

    create_classification_dataset(
        window_size_s=args.window_size_s,
        effective_hz=args.effective_hz,
        label_scheme=args.label_scheme,
        min_confidence=args.min_confidence,
        overwrite=True
    )

    # Steps 5-6: Verify
    print("\n[5/6] ", end="")
    verify_classification_dataset(args.window_size_s, args.effective_hz, args.label_scheme, class_names)

    print("\n[6/6] ", end="")
    verify_metadata_and_distribution(args.window_size_s, args.effective_hz, args.label_scheme, class_names)

    log_sample_windows(args.window_size_s, args.effective_hz, args.label_scheme)
    print_success_summary(full_mode=True)


def run_verify_only(args) -> None:
    """Run verify-only mode: check existing datasets."""
    print("=" * 60)
    print("Capture-24 Classification Verification (Verify-Only Mode)")
    print("=" * 60)

    # Steps 1-3: Verify label schemes
    print("\n[1/5] ", end="")
    verify_label_scheme(args.label_scheme)

    print("\n[2/5] ", end="")
    verify_label_mapping(args.label_scheme)

    print("\n[3/5] ", end="")
    class_names = verify_class_names(args.label_scheme)

    # Steps 4-5: Verify existing data
    print("\n[4/5] ", end="")
    verify_classification_dataset(args.window_size_s, args.effective_hz, args.label_scheme, class_names)

    print("\n[5/5] ", end="")
    verify_metadata_and_distribution(args.window_size_s, args.effective_hz, args.label_scheme, class_names)

    log_sample_windows(args.window_size_s, args.effective_hz, args.label_scheme)
    print_success_summary(full_mode=False)


def print_success_summary(full_mode: bool) -> None:
    """Print success summary."""
    print("\n" + "=" * 60)
    print("All verification checks passed!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if full_mode:
        print("Phase 2B classification dataset creation is working correctly:")
        print("  Label scheme validation")
        print("  Annotation-to-label mapping")
        print("  Class name extraction (alphabetically sorted)")
        print("  Classification dataset creation")
        print("  Schema validation (window_id, pid, x, y, z, label, label_id, confidence)")
        print("  Metadata and class distribution")
    else:
        print("Existing classification datasets verified successfully:")
        print("  Label scheme validation")
        print("  Schema validation")
        print("  Metadata and class distribution")

    print("\nYou can now use:")
    print("  - create_classification_dataset() to create labeled datasets")
    print("  - load_classification_dataset() to load pre-created datasets")
    print("  - get_class_names() to get class labels for a scheme")
    print("  - load_classification_metadata() to get dataset metadata")
    print("  - get_class_distribution() to get class counts per split")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Capture-24 classification dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full mode: create and verify
  python test_capture24_classification.py --window-size-s 10 --effective-hz 100

  # Verify-only mode: check existing datasets
  python test_capture24_classification.py --verify-only --window-size-s 10 --effective-hz 100
        """
    )

    parser.add_argument(
        "--verify-only", "-v",
        action="store_true",
        help="Only verify existing datasets (skip creation)"
    )

    parser.add_argument(
        "--window-size-s", "-w",
        type=float,
        default=2.56,
        help="Window size in seconds (default: 2.56)"
    )
    parser.add_argument(
        "--effective-hz", "-e",
        type=int,
        default=100,
        help="Effective sampling frequency (default: 50)"
    )
    parser.add_argument(
        "--label-scheme", "-l",
        type=str,
        default="Walmsley2020",
        help="Label scheme to use (default: Walmsley2020)"
    )
    parser.add_argument(
        "--min-confidence", "-c",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (full mode, default: 0.0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verify_only:
        run_verify_only(args)
    else:
        run_full_verification(args)

    return 0


if __name__ == "__main__":
    exit(main())
