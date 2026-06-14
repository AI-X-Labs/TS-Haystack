#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Verification script for Capture-24 dataset extraction."""

from src.datasets.capture24 import (
    ensure_capture24_data,
    load_participants,
    load_label_mappings,
    load_participant_sensor_data,
)


def main():
    print("=" * 60)
    print("Capture-24 Dataset Extraction Verification")
    print("=" * 60)

    # Step 1: Ensure data is extracted
    print("\n[1/4] Ensuring data is extracted...")
    ensure_capture24_data()

    # Step 2: Load and verify participants
    print("\n[2/4] Loading participants metadata...")
    participants = load_participants()
    print(f"  ✓ Loaded {len(participants)} participants")
    print(f"  Columns: {participants.columns}")
    print(f"  First 5 participants:\n{participants.head()}")

    # Step 3: Load and verify label mappings
    print("\n[3/4] Loading label mappings...")
    labels = load_label_mappings()
    print(f"  ✓ Loaded {len(labels)} annotation mappings")
    print(f"  Columns: {labels.columns}")
    print(f"  First 3 label schemes:\n{labels.head(3)}")

    # Step 4: Load and verify sensor data for one participant
    print("\n[4/4] Loading sensor data for P001...")
    df = load_participant_sensor_data("P001")
    print(f"  ✓ P001 has {len(df):,} samples")
    print(f"  Columns: {df.columns}")
    print(f"  Data types: {df.dtypes}")
    print(f"  First 5 rows:\n{df.head()}")
    print(f"  Memory usage: {df.estimated_size('mb'):.2f} MB")

    # Verify schema
    print("\n[Verification] Checking schema...")
    assert "timestamp_ms" in df.columns, "Missing timestamp_ms"
    assert "x" in df.columns, "Missing x"
    assert "y" in df.columns, "Missing y"
    assert "z" in df.columns, "Missing z"
    assert "annotation" in df.columns, "Missing annotation"
    print("  ✓ Schema verified")

    # Check data ranges
    print("\n[Verification] Checking data ranges...")
    print(f"  Time range: {df['timestamp_ms'].min()} to {df['timestamp_ms'].max()}")
    print(f"  X range: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    print(f"  Y range: [{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  Z range: [{df['z'].min():.4f}, {df['z'].max():.4f}]")
    print(f"  Samples with annotation: {df['annotation'].is_not_null().sum():,}")
    print(f"  Samples without annotation: {df['annotation'].is_null().sum():,}")

    print("\n" + "=" * 60)
    print("✓ All verification checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
