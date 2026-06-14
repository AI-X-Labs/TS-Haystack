# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Unit tests for the natural-only LTAF-Haystack task generators.

Tasks consume a real-slice :class:`LTAFRecordingSample`. Here we fabricate
minimal recordings with scripted rhythm + beat timelines (no real waveform
access) and verify each task can emit a valid :class:`LTAFGeneratedSample`
and that context-length gating matches the coverage table.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBeatEvent,
    LTAFBoutRecord,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.core.ltaf_prompt_templates import (
    LTAFPromptTemplateBank,
)
from src.datasets.ltaf_haystack.core.seed_manager import LTAFSeedManager
from src.datasets.ltaf_haystack.tasks import TASK_REGISTRY, list_available_tasks


SOURCE_HZ = 128
CTX_SAMPLES = 900 * SOURCE_HZ  # 15 min


class _StubRecordingSampler:
    """Minimal stand-in for :class:`RecordingSampler` — always returns `recording`."""

    class _Idx:
        context_length_s = 900

    def __init__(self, recording: LTAFRecordingSample):
        self._rec = recording
        self.window_index = self._Idx()
        self._source_hz = SOURCE_HZ

    def sample_recording(self, rng):
        return self._rec

    def sample_recording_for_activity(self, activity, want_present, rng):
        present = activity in self._rec.activity_index
        if present == want_present:
            return self._rec
        return None

    def sample_recording_with_beat_count(self, beat_type, min_count, rng):
        count = sum(1 for b in self._rec.beats_timeline if b.symbol == beat_type)
        return self._rec if count >= min_count else None

    def load_signals(self, recording):
        return recording.signals


def _make_recording(
    rhythm_bouts: List[LTAFBoutRecord],
    beats: List[LTAFBeatEvent],
    source_hz: int = SOURCE_HZ,
    ctx_samples: int = CTX_SAMPLES,
) -> LTAFRecordingSample:
    activity_index: Dict[str, List[LTAFBoutRecord]] = {}
    for b in rhythm_bouts:
        activity_index.setdefault(b.activity, []).append(b)
    return LTAFRecordingSample(
        record_id="rec_test",
        window_start_ms=0,
        window_end_ms=int(round(ctx_samples * 1000 / source_hz)),
        source_hz=source_hz,
        rhythm_timeline=list(rhythm_bouts),
        beats_timeline=list(beats),
        activity_index=activity_index,
        signals=np.zeros((ctx_samples, 2), dtype=np.float32),
    )


def _make_generator(task_name: str, recording: LTAFRecordingSample):
    template_bank = LTAFPromptTemplateBank()
    seed_manager = LTAFSeedManager(master_seed=123)
    sampler = _StubRecordingSampler(recording)
    return TASK_REGISTRY[task_name](
        recording_sampler=sampler,
        template_bank=template_bank,
        seed_manager=seed_manager,
        label_class="rhythms",
    )


# --------------------------------------------------------------------------- #
# Registry invariants
# --------------------------------------------------------------------------- #


def test_registry_has_10_tasks():
    tasks = list_available_tasks()
    assert len(tasks) == 10
    assert set(tasks) == {
        "existence", "localization", "counting", "ordering", "state_query",
        "antecedent", "comparison", "multi_hop", "anomaly_detection",
        "anomaly_localization",
    }


def test_task_classes_are_natural_only():
    for name, cls in TASK_REGISTRY.items():
        assert hasattr(cls, "_generate"), name
        assert not hasattr(cls, "_generate_natural"), name
        assert not hasattr(cls, "_generate_inserted"), name
        assert not hasattr(cls, "generate_insertion_sample"), name


# --------------------------------------------------------------------------- #
# Single-rhythm recordings (tasks that don't need ≥2 bouts)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "task_name",
    ["existence", "localization", "counting", "state_query",
     "anomaly_detection", "anomaly_localization"],
)
def test_single_rhythm_tasks_run(task_name):
    # Three short NSR bouts (~60s each) so localization's duration cap
    # (<=30% of ctx) leaves at least one legal candidate.
    bout_samples = 60 * SOURCE_HZ
    rhythm = [
        LTAFBoutRecord(0, bout_samples, "NSR", bout_samples),
        LTAFBoutRecord(300 * SOURCE_HZ, 300 * SOURCE_HZ + bout_samples,
                       "NSR", bout_samples),
        LTAFBoutRecord(600 * SOURCE_HZ, 600 * SOURCE_HZ + bout_samples,
                       "NSR", bout_samples),
    ]
    beats = [
        LTAFBeatEvent(sample=10 * SOURCE_HZ, time_ms=10_000, symbol="N"),
        LTAFBeatEvent(sample=15 * SOURCE_HZ, time_ms=15_000, symbol="V"),
        LTAFBeatEvent(sample=20 * SOURCE_HZ, time_ms=20_000, symbol="A"),
    ]
    rec = _make_recording(rhythm, beats)
    gen = _make_generator(task_name, rec)
    rng = np.random.default_rng(0)

    sample = None
    for _ in range(20):
        sample = gen.generate_sample(rec, rng)
        if sample.is_valid:
            break
    assert sample is not None and sample.is_valid, sample.invalid_reason
    assert sample.task_type == gen.task_name
    assert sample.answer_type == gen.answer_type
    assert sample.question != ""
    assert sample.answer != ""


# --------------------------------------------------------------------------- #
# Multi-rhythm recordings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "task_name",
    ["ordering", "antecedent", "comparison", "multi_hop"],
)
def test_multi_rhythm_tasks_run(task_name):
    rhythm = [
        LTAFBoutRecord(0, 2000, "NSR", 2000),
        LTAFBoutRecord(2000, 5000, "AFIB", 3000),
        LTAFBoutRecord(5000, 6000, "VT", 1000),
        LTAFBoutRecord(6000, 9000, "NSR", 3000),
        LTAFBoutRecord(9000, 11000, "AFIB", 2000),
        LTAFBoutRecord(11000, CTX_SAMPLES, "NSR", CTX_SAMPLES - 11000),
    ]
    beats = [
        LTAFBeatEvent(sample=1000, time_ms=7812, symbol="N"),
        LTAFBeatEvent(sample=5500, time_ms=42968, symbol="V"),
        LTAFBeatEvent(sample=10000, time_ms=78125, symbol="V"),
    ]
    rec = _make_recording(rhythm, beats)
    gen = _make_generator(task_name, rec)
    rng = np.random.default_rng(1)

    sample = None
    for _ in range(60):
        sample = gen.generate_sample(rec, rng)
        if sample.is_valid:
            break
    assert sample is not None and sample.is_valid, sample.invalid_reason
    assert sample.task_type == gen.task_name


# --------------------------------------------------------------------------- #
# Gating coverage table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "task_name,supported,unsupported",
    [
        ("existence",            [100, 900, 3600], [7200]),
        ("localization",         [100, 900, 3600, 7200], []),
        ("counting",             [100, 900, 3600, 7200], []),
        ("ordering",             [100, 900, 3600, 7200], []),
        ("antecedent",           [100, 900, 3600, 7200], []),
        ("comparison",           [100, 900, 3600, 7200], []),
        ("multi_hop",            [100, 900, 3600, 7200], []),
        ("state_query",          [100, 900, 3600, 7200], []),
        ("anomaly_detection",    [100, 900], [3600, 7200]),
        ("anomaly_localization", [100, 900, 3600, 7200], []),
    ],
)
def test_coverage_table(task_name, supported, unsupported):
    cls = TASK_REGISTRY[task_name]
    for ctx in supported:
        assert cls.supports_context_length("rhythms", ctx), f"{task_name} should support {ctx}s"
    for ctx in unsupported:
        assert not cls.supports_context_length("rhythms", ctx), (
            f"{task_name} should NOT support {ctx}s"
        )
    assert cls.supports_context_length("rhythms", None) is False
