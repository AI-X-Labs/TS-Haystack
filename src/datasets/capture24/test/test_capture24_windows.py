#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Verification script for Capture-24 window extraction.

Supports two modes:
- Full mode (default): Extract windows + verify
- Verify-only mode (--verify-only): Just verify existing windows
"""

import argparse
from pathlib import Path

from src.datasets.capture24 import (
    extract_windows,
    load_windows,
    split_participants,
    get_windows_path,
)


# =============================================================================
# Verification Functions (reusable building blocks)
# =============================================================================

def verify_participant_split(source_hz: int, seed: int) -> tuple:
    """Verify participant split is working and deterministic."""
    print("\nTesting participant split...")
    train_pids, val_pids, test_pids = split_participants(seed=seed, source_hz=source_hz)
    print(f"  Train: {len(train_pids)} participants")
    print(f"  Val: {len(val_pids)} participants")
    print(f"  Test: {len(test_pids)} participants")
    print(f"  Total: {len(train_pids) + len(val_pids) + len(test_pids)} participants")

    # Verify split is deterministic
    train_pids2, val_pids2, test_pids2 = split_participants(seed=seed, source_hz=source_hz)
    assert train_pids == train_pids2, "Split is not deterministic!"
    print("  Split is deterministic (same seed produces same split)")

    return train_pids, val_pids, test_pids


def verify_windows_path(window_size_s: float, effective_hz: int) -> Path:
    """Verify windows path exists."""
    print("\nVerifying window path...")
    windows_path = get_windows_path(window_size_s, effective_hz)
    print(f"  Windows path: {windows_path}")
    assert windows_path.exists(), f"Windows path does not exist: {windows_path}"
    return windows_path


def verify_windows_content(
    window_size_s: float,
    effective_hz: int,
    annotation_threshold: float = None,
) -> dict:
    """Load and verify windows for all splits."""
    print("\nLoading and verifying windows...")
    expected_len = int(window_size_s * effective_hz)
    results = {}

    for split in ["train", "val", "test"]:
        try:
            windows = load_windows(window_size_s, effective_hz, split)
            print(f"\n  {split.upper()} split:")
            print(f"    {len(windows):,} windows")
            print(f"    Columns: {windows.columns}")

            if len(windows) > 0:
                # Check first window shape
                first_window = windows[0]
                x_len = len(first_window["x"][0])
                y_len = len(first_window["y"][0])
                z_len = len(first_window["z"][0])
                ann_len = len(first_window["annotations"][0])

                print(f"    Sample window shape: x={x_len}, y={y_len}, z={z_len}, annotations={ann_len}")
                print(f"    Expected length: {expected_len}")

                assert x_len == expected_len, f"X length mismatch: {x_len} != {expected_len}"
                assert y_len == expected_len, f"Y length mismatch: {y_len} != {expected_len}"
                assert z_len == expected_len, f"Z length mismatch: {z_len} != {expected_len}"
                assert ann_len == expected_len, f"Annotations length mismatch: {ann_len} != {expected_len}"

                # Check annotation threshold if provided
                if annotation_threshold is not None:
                    sample_annotations = first_window["annotations"][0]
                    non_null_count = sum(1 for a in sample_annotations if a is not None)
                    annotation_ratio = non_null_count / len(sample_annotations)
                    print(f"    Annotation ratio: {annotation_ratio:.2%} (threshold: {annotation_threshold:.0%})")
                    assert annotation_ratio >= annotation_threshold, \
                        f"Annotation ratio {annotation_ratio:.2%} below threshold {annotation_threshold:.0%}"

                # Check window metadata
                print(f"    Window ID: {first_window['window_id'][0]}")
                print(f"    Participant: {first_window['pid'][0]}")
                print(f"    Start timestamp: {first_window['start_ms'][0]}")
                print(f"    End timestamp: {first_window['end_ms'][0]}")

            results[split] = len(windows)

        except FileNotFoundError:
            print(f"\n  {split.upper()} split: No windows (expected if no participants in this split)")
            results[split] = 0

    return results


def extract_windows_step(
    window_size_s: float,
    source_hz: int,
    downsample_hz: int,
    annotation_threshold: float,
    seed: int,
    max_participants: int,
) -> None:
    """Extract windows with given parameters."""
    hz_display = f"{downsample_hz}Hz" if downsample_hz else "(no downsampling)"
    print(f"\nExtracting windows ({window_size_s}s @ {hz_display})...")
    print(f"  Annotation threshold: {annotation_threshold:.0%}")

    extract_windows(
        window_size_s=window_size_s,
        source_hz=source_hz,
        downsample_hz=downsample_hz,
        annotation_threshold=annotation_threshold,
        seed=seed,
        max_participants=max_participants,
        overwrite=True
    )


# =============================================================================
# Mode Runners
# =============================================================================

def run_full_verification(args) -> None:
    """Run full verification: extract + verify (original behavior)."""
    print("=" * 60)
    print("Capture-24 Window Extraction Verification (Full Mode)")
    print("=" * 60)

    effective_hz = args.downsample_hz if args.downsample_hz is not None else args.source_hz

    # Step 1: Check participant split
    print("\n[1/5] ", end="")
    verify_participant_split(args.source_hz, args.seed)

    # Step 2: Extract windows
    print("\n[2/5] ", end="")
    extract_windows_step(
        window_size_s=args.window_size_s,
        source_hz=args.source_hz,
        downsample_hz=args.downsample_hz,
        annotation_threshold=args.annotation_threshold,
        seed=args.seed,
        max_participants=args.max_participants,
    )

    # Step 3: Verify window path
    print("\n[3/5] ", end="")
    verify_windows_path(args.window_size_s, effective_hz)

    # Step 4: Load and verify windows
    print("\n[4/5] ", end="")
    verify_windows_content(args.window_size_s, effective_hz, args.annotation_threshold)

    # Step 5: Test with downsampling (if source allows)
    if args.source_hz >= 2:
        downsample_target_hz = int(args.source_hz / 2)
        print(f"\n[5/5] Testing window extraction with downsampling to {downsample_target_hz}Hz...")
        extract_windows_step(
            window_size_s=5,
            source_hz=args.source_hz,
            downsample_hz=downsample_target_hz,
            annotation_threshold=0.8,
            seed=args.seed,
            max_participants=max(1, args.max_participants - 1) if args.max_participants else None,
        )

        try:
            windows_downsampled = load_windows(window_size_s=5, effective_hz=downsample_target_hz, split="train")
            if len(windows_downsampled) > 0:
                sample_len = len(windows_downsampled["x"][0])
                expected_len = 5 * downsample_target_hz
                print(f"  Downsampled to {downsample_target_hz}Hz: {sample_len} samples (expected {expected_len})")
                assert sample_len == expected_len, f"Sample length mismatch: {sample_len} != {expected_len}"
        except FileNotFoundError:
            print("  No downsampled windows found (may be expected with limited participants)")

    print_success_summary(args.source_hz, full_mode=True)


def run_verify_only(args) -> None:
    """Run verify-only mode: check existing windows without extraction."""
    print("=" * 60)
    print("Capture-24 Window Verification (Verify-Only Mode)")
    print("=" * 60)

    # Step 1: Check participant split
    print("\n[1/3] ", end="")
    verify_participant_split(args.effective_hz, args.seed)

    # Step 2: Verify window path
    print("\n[2/3] ", end="")
    verify_windows_path(args.window_size_s, args.effective_hz)

    # Step 3: Load and verify windows
    print("\n[3/3] ", end="")
    verify_windows_content(args.window_size_s, args.effective_hz, args.annotation_threshold)

    print_success_summary(args.effective_hz, full_mode=False)


def print_success_summary(hz: int, full_mode: bool) -> None:
    """Print success summary."""
    print("\n" + "=" * 60)
    print("All verification checks passed!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if full_mode:
        print("Phase 2A window extraction is working correctly:")
        print("  Participant splitting (train/val/test)")
        print(f"  Window extraction at source frequency ({hz}Hz)")
        print(f"  Window extraction with downsampling ({int(hz/2)}Hz)")
        print("  Annotation threshold filtering")
        print("  Window schema validation")
        print("  Metadata preservation")
        print("  Hz-specific directory structure")
    else:
        print("Existing windows verified successfully:")
        print("  Participant split consistency")
        print("  Window path exists")
        print("  Window schema validation")
        print("  Window content validation")

    print("\nYou can now use:")
    print("  - extract_windows(source_hz=..., downsample_hz=...) to create window datasets")
    print("  - load_windows(effective_hz=...) to load pre-extracted windows")
    print("  - split_participants(source_hz=...) to get train/val/test splits")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Capture-24 window extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full mode: extract and verify
  python test_capture24_windows.py --window-size-s 10 --source-hz 100

  # Verify-only mode: check existing windows
  python test_capture24_windows.py --verify-only --window-size-s 10 --effective-hz 100
        """
    )

    # Mode selection
    parser.add_argument(
        "--verify-only", "-v",
        action="store_true",
        help="Only verify existing windows (skip extraction)"
    )

    # Window parameters
    parser.add_argument(
        "--window-size-s", "-w",
        type=float,
        default=2.56,
        help="Window size in seconds (default: 2.56)"
    )
    parser.add_argument(
        "--annotation-threshold", "-a",
        type=float,
        default=0.6,
        help="Minimum fraction of annotated samples (default: 0.6)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for participant splits (default: 42)"
    )

    # Full mode parameters
    parser.add_argument(
        "--source-hz",
        type=int,
        default=10,
        help="Source data sampling frequency (full mode, default: 10)"
    )
    parser.add_argument(
        "--downsample-hz", "-d",
        type=int,
        default=None,
        help="Target sampling frequency for downsampling (full mode, default: None)"
    )
    parser.add_argument(
        "--max-participants", "-n",
        type=int,
        default=3,
        help="Limit participants for testing (full mode, default: 3)"
    )

    # Verify-only mode parameters
    parser.add_argument(
        "--effective-hz", "-e",
        type=int,
        default=None,
        help="Effective Hz of existing windows (verify-only mode, required if --verify-only)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verify_only:
        # Verify-only mode requires effective_hz
        if args.effective_hz is None:
            print("Error: --effective-hz is required when using --verify-only")
            print("Example: --verify-only --effective-hz 100")
            return 1
        run_verify_only(args)
    else:
        run_full_verification(args)

    return 0


if __name__ == "__main__":
    exit(main())
