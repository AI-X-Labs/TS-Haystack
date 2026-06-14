#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Verification script for Capture-24 QADataset integration.

This script tests the complete pipeline from classification parquet files
to the QADataset format used for OpenTSLM training.
"""

from torch.utils.data import DataLoader

from src.datasets.capture24 import (
    Capture24AccQADataset,
    get_label_list,
    load_capture24_classification_splits,
)
from src.datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


def test_loader(window_size_s: int, effective_hz: int, label_scheme: str):
    """Test the capture24_qa_loader module."""
    print("\n[1/5] Testing capture24_qa_loader...")

    # Test load_capture24_classification_splits
    train_ds, val_ds, test_ds = load_capture24_classification_splits(
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )

    print(f"  Loaded splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Verify schema
    if len(train_ds) > 0:
        sample = train_ds[0]
        required_keys = ["x_axis", "y_axis", "z_axis", "label"]
        for key in required_keys:
            assert key in sample, f"Missing required key: {key}"
        print(f"  Schema verified: {list(sample.keys())}")

        # Verify data shapes
        expected_length = window_size_s * effective_hz
        assert len(sample["x_axis"]) == expected_length, (
            f"x_axis length mismatch: {len(sample['x_axis'])} != {expected_length}"
        )
        print(f"  Samples per window: {len(sample['x_axis'])} (expected: {expected_length})")

    # Test get_label_list
    labels = get_label_list(label_scheme)
    print(f"  Labels ({label_scheme}): {labels}")
    assert len(labels) > 0, "No labels found"
    assert labels == sorted(labels), "Labels should be alphabetically sorted"

    print("  [PASS] Loader tests passed")
    return train_ds, val_ds, test_ds


def test_qa_dataset(window_size_s: int, effective_hz: int, label_scheme: str):
    """Test the Capture24AccQADataset class."""
    print("\n[2/5] Testing Capture24AccQADataset...")

    # Create dataset instances
    dataset = Capture24AccQADataset(
        split="train",
        EOS_TOKEN="",
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
    )

    print(f"  Training dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Training dataset should not be empty"

    # Test get_labels method
    labels = dataset.get_labels()
    print(f"  Labels: {labels}")
    assert len(labels) > 0, "Labels should not be empty"

    # Test sample access
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")

    # Verify expected keys
    expected_keys = ["pre_prompt", "time_series", "post_prompt", "answer"]
    for key in expected_keys:
        assert key in sample, f"Missing expected key: {key}"

    # Verify time series format
    assert "time_series" in sample, "Missing time_series"
    assert len(sample["time_series"]) == 3, "Should have 3 axes (x, y, z)"
    print(f"  Time series axes: {len(sample['time_series'])}")

    # Verify answer is a valid label
    assert sample["answer"] in labels, f"Answer '{sample['answer']}' not in labels"
    print(f"  Answer: {sample['answer']}")

    # Verify raw data is preserved
    assert "label" in sample, "Missing raw label"
    assert "x_axis" in sample, "Missing raw x_axis"
    assert "y_axis" in sample, "Missing raw y_axis"
    assert "z_axis" in sample, "Missing raw z_axis"
    print("  Raw data preserved in sample")

    print("  [PASS] QADataset tests passed")
    return dataset


def test_all_splits(window_size_s: int, effective_hz: int, label_scheme: str):
    """Test all dataset splits can be created."""
    print("\n[3/5] Testing all splits...")

    splits = ["train", "validation", "test"]
    for split in splits:
        dataset = Capture24AccQADataset(
            split=split,
            EOS_TOKEN="",
            window_size_s=window_size_s,
            effective_hz=effective_hz,
            label_scheme=label_scheme,
        )
        print(f"  {split}: {len(dataset)} samples")

    print("  [PASS] All splits loaded successfully")


def test_dataloader_integration(dataset):
    """Test DataLoader integration with collate function."""
    print("\n[4/5] Testing DataLoader integration...")

    if len(dataset) == 0:
        print("  [SKIP] Dataset empty, skipping DataLoader test")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    # Get one batch
    for batch in dataloader:
        print(f"  Batch size: {len(batch)}")
        print(f"  Batch[0] keys: {list(batch[0].keys())}")

        # Verify batch structure
        assert len(batch) > 0, "Batch should not be empty"
        assert "time_series" in batch[0], "Batch items should have time_series"
        assert "answer" in batch[0], "Batch items should have answer"

        # Check time series shape after padding
        ts = batch[0]["time_series"]
        print(f"  Time series shape after padding: {len(ts)} axes")
        for i, axis_ts in enumerate(ts):
            print(f"    Axis {i}: length={len(axis_ts)}")

        break  # Only test one batch

    print("  [PASS] DataLoader integration tests passed")


def test_prompt_structure(dataset):
    """Test the prompt structure is correct."""
    print("\n[5/5] Testing prompt structure...")

    if len(dataset) == 0:
        print("  [SKIP] Dataset empty, skipping prompt test")
        return

    sample = dataset[0]

    # Check pre_prompt
    pre_prompt = sample["pre_prompt"]
    assert isinstance(pre_prompt, str), "pre_prompt should be a string"
    assert len(pre_prompt) > 0, "pre_prompt should not be empty"
    assert "accelerometer" in pre_prompt.lower(), "pre_prompt should mention accelerometer"
    print(f"  Pre-prompt: '{pre_prompt[:50]}...'")

    # Check post_prompt
    post_prompt = sample["post_prompt"]
    assert isinstance(post_prompt, str), "post_prompt should be a string"
    assert len(post_prompt) > 0, "post_prompt should not be empty"
    assert "Answer:" in post_prompt, "post_prompt should contain 'Answer:' instruction"

    # Check that labels are mentioned in post_prompt
    labels = dataset.get_labels()
    for label in labels:
        assert label in post_prompt, f"Label '{label}' should be in post_prompt"
    print(f"  Post-prompt contains all {len(labels)} labels")

    # Check answer format
    answer = sample["answer"]
    assert answer in labels, f"Answer '{answer}' should be a valid label"
    print(f"  Answer: '{answer}'")

    print("  [PASS] Prompt structure tests passed")


def main():
    print("=" * 60)
    print("Capture-24 QADataset Integration Test")
    print("=" * 60)

    # Test configuration
    window_size_s = 10
    effective_hz = 100
    label_scheme = "Walmsley2020"

    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Label scheme: {label_scheme}")

    # Run tests
    try:
        # Test 1: Loader
        train_ds, val_ds, test_ds = test_loader(
            window_size_s, effective_hz, label_scheme
        )

        # Test 2: QADataset
        dataset = test_qa_dataset(window_size_s, effective_hz, label_scheme)

        # Test 3: All splits
        test_all_splits(window_size_s, effective_hz, label_scheme)

        # Test 4: DataLoader integration
        test_dataloader_integration(dataset)

        # Test 5: Prompt structure
        test_prompt_structure(dataset)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

        # Summary
        print("\nSummary:")
        print("  [PASS] capture24_qa_loader.py - HuggingFace Dataset loading")
        print("  [PASS] Capture24AccQADataset.py - QADataset subclass")
        print("  [PASS] All splits (train/validation/test)")
        print("  [PASS] DataLoader integration with collate function")
        print("  [PASS] Prompt structure (pre_prompt, time_series, post_prompt, answer)")

        print("\nThe Capture-24 dataset is ready for OpenTSLM training!")
        print("\nUsage example:")
        print("  from src.datasets.capture24 import Capture24AccQADataset")
        print("")
        print("  train_dataset = Capture24AccQADataset(")
        print("      split='train',")
        print("      EOS_TOKEN=tokenizer.eos_token,")
        print("      window_size_s=10,")
        print("      effective_hz=100,")
        print("      label_scheme='Walmsley2020'")
        print("  )")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
