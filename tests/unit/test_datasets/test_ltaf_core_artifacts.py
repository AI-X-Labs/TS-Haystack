# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for LTAF-Haystack core artifact builders."""

from pathlib import Path

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBoutRecord,
    LTAFParticipantTimeline,
)
from src.datasets.ltaf_haystack.core.ltaf_bout_indexer import (
    LTAFBoutIndexer,
    get_ltaf_bout_index_path,
)
from src.datasets.ltaf_haystack.core.ltaf_timeline_builder import (
    _build_bouts_from_changes,
    _extract_rhythm_changes,
    _parse_rhythm_aux_note,
    LTAFTimelineBuilder,
)


class _FakeAnnotation:
    def __init__(self, sample, symbol, aux_note):
        self.sample = sample
        self.symbol = symbol
        self.aux_note = aux_note


class TestRhythmParsing:
    def test_parse_rhythm_aux_note_maps_common_labels(self):
        assert _parse_rhythm_aux_note("(N") == "NSR"
        assert _parse_rhythm_aux_note("(AF") == "AFIB"
        assert _parse_rhythm_aux_note(b"(AFL") == "AFL"

    def test_extract_rhythm_changes_uses_symbol_and_dedupes(self):
        ann = _FakeAnnotation(
            sample=[10, 10, 20, 30],
            symbol=["+", "+", "N", "+"],
            aux_note=["(N", "(AF", "(AF", "(AFL"],
        )

        # Only "+" symbols define rhythm changes; duplicate sample keeps last value.
        changes = _extract_rhythm_changes(ann)
        assert changes == [(10, "AFIB"), (30, "AFL")]


class TestBoutConstruction:
    def test_build_bouts_respects_boundaries_and_default_rhythm(self):
        bouts = _build_bouts_from_changes(
            signal_length=100,
            changes=[(20, "AFIB"), (60, "NSR")],
            min_bout_duration_samples=10,
        )

        assert [(b.start_sample, b.end_sample, b.activity) for b in bouts] == [
            (0, 20, "NSR"),
            (20, 60, "AFIB"),
            (60, 100, "NSR"),
        ]

    def test_build_bouts_filters_short_segments(self):
        bouts = _build_bouts_from_changes(
            signal_length=100,
            changes=[(5, "AFIB"), (95, "NSR")],
            min_bout_duration_samples=10,
        )

        # First and last segments are too short and should be dropped.
        assert len(bouts) == 1
        assert bouts[0].activity == "AFIB"
        assert bouts[0].start_sample == 5
        assert bouts[0].end_sample == 95


class TestSplitManifest:
    def test_split_manifest_is_deterministic(self):
        participants = [f"p{i:02d}" for i in range(10)]

        m1 = LTAFTimelineBuilder.create_split_manifest(participants, seed=123)
        m2 = LTAFTimelineBuilder.create_split_manifest(participants, seed=123)

        assert m1 == m2
        assert m1["n_participants"] == 10
        assert len(m1["train"]) + len(m1["validation"]) + len(m1["test"]) == 10


class TestBoutIndex:
    def test_build_index_computes_counts_and_stats(self):
        timelines = {
            "p1": LTAFParticipantTimeline(
                participant_id="p1",
                record_id="r1",
                source_hz=128,
                timeline=[
                    LTAFBoutRecord(0, 50, "NSR", 50),
                    LTAFBoutRecord(50, 120, "AFIB", 70),
                ],
            ),
            "p2": LTAFParticipantTimeline(
                participant_id="p2",
                record_id="r2",
                source_hz=128,
                timeline=[
                    LTAFBoutRecord(0, 30, "NSR", 30),
                ],
            ),
        }

        indexer = LTAFBoutIndexer(min_bout_duration_samples=16)
        index = indexer.build_index(timelines)

        assert index.total_bouts == 3
        assert set(index.activities) == {"NSR", "AFIB"}
        assert index.activity_stats["NSR"].count == 2
        assert index.activity_stats["AFIB"].count == 1


class TestPathOverrides:
    def test_builder_uses_custom_output_root(self):
        builder = LTAFTimelineBuilder(
            raw_data_dir=Path("data/ltafdb/raw"),
            output_root=Path("data/custom_ltaf_output"),
        )
        assert builder.get_timelines_dir() == Path("data/custom_ltaf_output/timelines")
        assert builder.get_split_manifest_path() == Path("data/custom_ltaf_output/split_manifest.json")

    def test_bout_index_path_override(self):
        output_root = Path("data/custom_ltaf_output")
        assert get_ltaf_bout_index_path("parquet", output_root=output_root) == output_root / "bout_index.parquet"
        assert get_ltaf_bout_index_path("json", output_root=output_root) == output_root / "bout_index.json"
