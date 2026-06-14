# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for LTAF-Haystack runtime parquet loader and dataset wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.datasets.ltaf_haystack import qa_dataset as qa_dataset_module
from src.datasets.ltaf_haystack.cot_qa_dataset import LTAFHaystackCoTQADataset
from src.datasets.ltaf_haystack.qa_dataset import LTAFHaystackQADataset
from src.datasets.ltaf_haystack.qa_loader import (
    get_available_context_lengths,
    get_available_tasks,
    load_ltaf_haystack_splits,
)


@pytest.fixture
def stub_signal_loader(monkeypatch):
    """Stub memmap hydration so tests don't need a real .npy cache."""

    def _fake(record_id, window_start_ms, window_end_ms, source_hz=128):
        n = max(1, int(round((window_end_ms - window_start_ms) / 1000.0 * source_hz)))
        return np.stack(
            [np.linspace(-0.2, 0.2, n, dtype=np.float32),
             np.linspace(-0.1, 0.1, n, dtype=np.float32)],
            axis=1,
        )

    monkeypatch.setattr(qa_dataset_module, "load_window_ms", _fake)
    return _fake


def _sample_row(
    task_type: str = "existence",
    answer: str = "yes",
    answer_type: str = "boolean",
    rationale: str | None = None,
) -> dict:
    row = {
        "record_id": "rec_001",
        "context_length_samples": 6,
        "source_hz": 128,
        "window_start_ms": 0,
        "window_end_ms": 46,
        "question": "Is there any AFIB event in this ECG recording?",
        "answer": answer,
        "answer_type": answer_type,
        "task_type": task_type,
        "metadata": "{}",
        "is_valid": True,
        "invalid_reason": "",
    }
    if rationale is not None:
        row["rationale"] = rationale
    return row


def _write_task_parquet(base_dir: Path, context_dir: str, task: str, split_dir: str, rows: list[dict]) -> Path:
    out_path = base_dir / context_dir / task / split_dir / "data.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_path)
    return out_path


def _clear_dataset_cache(dataset_cls: type) -> None:
    attrs = [
        "_raw_split_cache",
        "_formatted_split_cache",
        "_raw_loaded",
        "_raw_train",
        "_raw_val",
        "_raw_test",
        "loaded",
        "_train_dataset",
        "_validation_dataset",
        "_test_dataset",
    ]
    for attr in attrs:
        if attr in dataset_cls.__dict__:
            delattr(dataset_cls, attr)


def test_ltaf_loader_discovers_tasks_and_contexts(temp_dir):
    _write_task_parquet(temp_dir, "2_56s", "existence", "train", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "10_0s", "counting", "train", [_sample_row(task_type="counting", answer="2", answer_type="integer")])

    tasks = get_available_tasks(base_dir=temp_dir)
    context_lengths = get_available_context_lengths(base_dir=temp_dir)

    assert tasks == ["counting", "existence"]
    assert context_lengths == [2.56, 10.0]


def test_ltaf_loader_reads_filtered_splits_with_validation_alias(temp_dir):
    _write_task_parquet(temp_dir, "2_56s", "existence", "train", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "validation", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "test", [_sample_row(task_type="existence")])

    train, val, test = load_ltaf_haystack_splits(
        tasks=["existence"],
        context_lengths_seconds=["2.56"],
        base_dir=temp_dir,
    )

    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    row = train[0]
    assert row["task_type"] == "existence"
    assert row["record_id"] == "rec_001"
    assert row["answer"] == "yes"


def test_ltaf_qadataset_formats_loaded_sample(temp_dir, stub_signal_loader):
    _write_task_parquet(temp_dir, "2_56s", "existence", "train", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "val", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "test", [_sample_row(task_type="existence")])

    _clear_dataset_cache(LTAFHaystackQADataset)

    ds = LTAFHaystackQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["existence"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
    )

    assert len(ds) == 1
    sample = ds[0]
    assert sample["answer"].endswith("<eos>")
    assert len(sample["time_series"]) == 2
    assert sample["task_type"] == "existence"
    assert sample["answer_type"] == "boolean"


def test_ltaf_cot_qadataset_uses_rationale_answer(temp_dir, stub_signal_loader):
    rationale = "Reasoning... Answer: yes"
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "train",
        [_sample_row(task_type="existence", rationale=rationale)],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "val",
        [_sample_row(task_type="existence", rationale=rationale)],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "test",
        [_sample_row(task_type="existence", rationale=rationale)],
    )

    _clear_dataset_cache(LTAFHaystackCoTQADataset)

    ds = LTAFHaystackCoTQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["existence"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
    )

    sample = ds[0]
    assert sample["answer"].startswith("Reasoning")
    assert sample["answer"].endswith("<eos>")
    assert "Answer:" in sample["answer"]


def test_ltaf_qadataset_cache_isolated_by_task_selection(temp_dir, stub_signal_loader):
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "train",
        [_sample_row(task_type="existence", answer="yes", answer_type="boolean")],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "val",
        [_sample_row(task_type="existence", answer="yes", answer_type="boolean")],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "existence",
        "test",
        [_sample_row(task_type="existence", answer="yes", answer_type="boolean")],
    )

    _write_task_parquet(
        temp_dir,
        "2_56s",
        "counting",
        "train",
        [_sample_row(task_type="counting", answer="2", answer_type="integer")],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "counting",
        "val",
        [_sample_row(task_type="counting", answer="2", answer_type="integer")],
    )
    _write_task_parquet(
        temp_dir,
        "2_56s",
        "counting",
        "test",
        [_sample_row(task_type="counting", answer="2", answer_type="integer")],
    )

    _clear_dataset_cache(LTAFHaystackQADataset)

    existence_ds = LTAFHaystackQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["existence"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
    )

    counting_ds = LTAFHaystackQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["counting"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
    )

    assert existence_ds[0]["task_type"] == "existence"
    assert counting_ds[0]["task_type"] == "counting"


def test_ltaf_qadataset_string_mode_preserves_metadata(temp_dir, stub_signal_loader):
    _write_task_parquet(temp_dir, "2_56s", "existence", "train", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "val", [_sample_row(task_type="existence")])
    _write_task_parquet(temp_dir, "2_56s", "existence", "test", [_sample_row(task_type="existence")])

    _clear_dataset_cache(LTAFHaystackQADataset)

    ds = LTAFHaystackQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["existence"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
        format_sample_str=True,
    )

    sample = ds[0]
    assert sample["task_type"] == "existence"
    assert sample["answer_type"] == "boolean"
    assert sample["context_length_samples"] == 6
    assert sample["source_hz"] == 128
    assert "question" in sample


def test_ltaf_qadataset_evaluate_answer_uses_iou_for_time_ranges(temp_dir):
    _write_task_parquet(temp_dir, "2_56s", "localization", "train", [_sample_row(task_type="localization")])
    _write_task_parquet(temp_dir, "2_56s", "localization", "val", [_sample_row(task_type="localization")])
    _write_task_parquet(temp_dir, "2_56s", "localization", "test", [_sample_row(task_type="localization")])

    _clear_dataset_cache(LTAFHaystackQADataset)

    ds = LTAFHaystackQADataset(
        split="train",
        EOS_TOKEN="<eos>",
        tasks=["localization"],
        context_lengths_seconds=[2.56],
        base_dir=temp_dir,
        lazy_loading=True,
    )

    sample = {
        "answer_type": "time_range",
        "answer": "From 3:25:04.240 AM to 3:25:08.570 AM",
    }

    good_pred = "From 3:25:04.500 AM to 3:25:08.000 AM"
    bad_pred = "From 3:25:20.000 AM to 3:25:22.000 AM"

    good = ds.evaluate_answer(good_pred, sample)
    bad = ds.evaluate_answer(bad_pred, sample)

    assert good["correct"] is True
    assert isinstance(good["iou"], float)
    assert bad["correct"] is False
