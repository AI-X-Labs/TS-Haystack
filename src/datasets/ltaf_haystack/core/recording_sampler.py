# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Natural-path recording sampler for LTAF-Haystack.

Given a split of record IDs and a pre-built ``LTAFWindowIndex``, samples
``LTAFRecordingSample`` objects whose rhythm and beat timelines are clipped
and shifted to be window-relative. Tasks consume this shape without
caring whether the sample came from the natural or insertion path.

The sampler exposes three entry points:

  * ``sample_recording(rng)`` — uniform window.
  * ``sample_recording_for_activity(activity, want_present, rng)`` —
    existence / ordering tasks' balanced positive/negative draw.
  * ``sample_recording_with_beat_count(beat_type, min_count, rng)`` —
    anomaly_detection / localization tasks.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import (
    BEAT_EVENT_TYPES,
    get_all_activities,
)
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBeatEvent,
    LTAFBoutRecord,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.core.ltaf_timeline_builder import (
    LTAFTimelineBuilder,
    get_ltaf_beat_timeline_path,
    get_ltaf_timeline_path,
)
from src.datasets.ltaf_haystack.core.window_index import LTAFWindowIndex
from src.datasets.ltaf_haystack.loader import (
    SOURCE_HZ,
    load_bout_signal,
    parse_header,
)


def _ms_to_samples(ms: int, source_hz: int = SOURCE_HZ) -> int:
    return int(round(ms * source_hz / 1000.0))


def _samples_to_ms(samples: int, source_hz: int = SOURCE_HZ) -> int:
    return int(round(samples * 1000.0 / source_hz))


class RecordingSampler:
    """Sample windowed LTAF recordings scoped to a split.

    The sampler preloads rhythm timelines, beat timelines, and headers for
    every record in the split. It also builds per-(rhythm, present) pools
    and per-(beat_type, count≥k) views over the window index so the
    task-specific sampling entry points run in O(1) after init.
    """

    def __init__(
        self,
        record_ids: List[str],
        label_class: str,
        window_index: LTAFWindowIndex,
    ):
        if window_index is None:
            raise ValueError("LTAF RecordingSampler requires a window_index")

        self.record_ids = list(record_ids)
        self.label_class = label_class
        self.window_index = window_index
        self._regime_set = set(get_all_activities(label_class))
        self._source_hz = int(window_index.source_hz)
        self._ctx_ms = window_index.context_length_ms
        self._ctx_samples = _ms_to_samples(self._ctx_ms, self._source_hz)

        self._rhythm_timelines: Dict[str, List[LTAFBoutRecord]] = {}
        self._beat_samples_by_sym: Dict[str, Dict[str, List[int]]] = {}
        self._beats_with_time: Dict[str, List[LTAFBeatEvent]] = {}
        self._headers: Dict[str, Dict] = {}

        for rid in self.record_ids:
            self._rhythm_timelines[rid] = self._load_rhythm_timeline(rid)
            self._beats_with_time[rid] = self._load_beat_timeline(rid)
            self._beat_samples_by_sym[rid] = self._group_beat_samples(
                self._beats_with_time[rid]
            )
            try:
                self._headers[rid] = parse_header(rid)
            except FileNotFoundError:
                self._headers[rid] = {"record_id": rid, "fs": self._source_hz, "sig_len": 0}

        # Pre-build per-(activity, present) + indexed-pair pools.
        split_set = set(self.record_ids)
        regime = window_index.regime
        bit_for = {a: 1 << i for i, a in enumerate(regime)}

        self._indexed_pairs: List[Tuple[str, int]] = []
        self._pairs_by_activity_presence: Dict[Tuple[str, bool], List[Tuple[str, int]]] = {
            (a, True): [] for a in regime
        }
        for a in regime:
            self._pairs_by_activity_presence[(a, False)] = []

        self._pairs_by_beat_min_count: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}

        for rid, entry in window_index.windows_by_subject.items():
            if rid not in split_set:
                continue
            starts = entry["starts"]
            masks = entry["masks"]
            beat_counts = entry["beat_counts"]
            for start_ms, mask, counts in zip(starts, masks, beat_counts):
                pair = (rid, int(start_ms))
                self._indexed_pairs.append(pair)
                for a in regime:
                    present = bool(mask & bit_for[a])
                    self._pairs_by_activity_presence[(a, present)].append(pair)
                for sym_idx, sym in enumerate(window_index.beat_symbols):
                    c = int(counts[sym_idx])
                    # Materialize cumulative tier pools on-the-fly for tiers 1..c.
                    # Callers usually ask for a small `k`, so we store by-symbol
                    # lists indexed at exact counts and filter at query time.
                    # Faster approach: bucket per threshold requested.
                    # Lazy: expose via a direct filter helper (below).
                    pass

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    @staticmethod
    def _load_rhythm_timeline(record_id: str) -> List[LTAFBoutRecord]:
        path = get_ltaf_timeline_path(record_id)
        if not path.exists():
            return []
        return LTAFTimelineBuilder._load_timeline(path).timeline

    @staticmethod
    def _load_beat_timeline(record_id: str) -> List[LTAFBeatEvent]:
        path = get_ltaf_beat_timeline_path(record_id)
        if not path.exists():
            return []
        return LTAFTimelineBuilder.load_beat_timeline(path)

    @staticmethod
    def _group_beat_samples(beats: List[LTAFBeatEvent]) -> Dict[str, List[int]]:
        by: Dict[str, List[int]] = {s: [] for s in BEAT_EVENT_TYPES}
        for b in beats:
            if b.symbol in by:
                by[b.symbol].append(b.sample)
        for sym in by:
            by[sym].sort()
        return by

    # ------------------------------------------------------------------
    # Public sampling API
    # ------------------------------------------------------------------
    def sample_recording(self, rng: np.random.Generator) -> LTAFRecordingSample:
        if not self._indexed_pairs:
            raise ValueError(
                f"Window index has no usable windows for this split "
                f"(label_class={self.label_class}, "
                f"context_length_s={self.window_index.context_length_s})"
            )
        idx = int(rng.integers(0, len(self._indexed_pairs)))
        rid, start_ms = self._indexed_pairs[idx]
        return self._load_windowed_recording(rid, start_ms)

    def sample_recording_for_activity(
        self,
        activity: str,
        want_present: bool,
        rng: np.random.Generator,
    ) -> Optional[LTAFRecordingSample]:
        pool = self._pairs_by_activity_presence.get((activity, want_present), [])
        if not pool:
            return None
        idx = int(rng.integers(0, len(pool)))
        rid, start_ms = pool[idx]
        return self._load_windowed_recording(rid, start_ms)

    def sample_recording_with_beat_count(
        self,
        beat_type: str,
        min_count: int,
        rng: np.random.Generator,
    ) -> Optional[LTAFRecordingSample]:
        pairs = self.window_index.windows_with_beat_count_gte(
            beat_type=beat_type,
            min_count=min_count,
            record_ids=self.record_ids,
        )
        if not pairs:
            return None
        idx = int(rng.integers(0, len(pairs)))
        rid, start_ms = pairs[idx]
        return self._load_windowed_recording(rid, start_ms)

    def count_activity_pool(self, activity: str, want_present: bool = True) -> int:
        """How many indexed (record, start) pairs match the predicate."""
        return len(self._pairs_by_activity_presence.get((activity, want_present), []))

    def count_beat_pool(self, beat_type: str, min_count: int) -> int:
        return len(
            self.window_index.windows_with_beat_count_gte(
                beat_type=beat_type,
                min_count=min_count,
                record_ids=self.record_ids,
            )
        )

    @property
    def available_record_ids(self) -> List[str]:
        return list(self.record_ids)

    # ------------------------------------------------------------------
    # Window loader: clip + shift
    # ------------------------------------------------------------------
    def _load_windowed_recording(
        self,
        record_id: str,
        window_start_ms: int,
        load_signals: bool = False,
    ) -> LTAFRecordingSample:
        ctx_ms = self._ctx_ms
        ctx_samples = self._ctx_samples
        source_hz = self._source_hz

        start_samp = _ms_to_samples(window_start_ms, source_hz)
        end_samp = start_samp + ctx_samples

        rhythm_clipped = self._clip_rhythm_timeline(
            self._rhythm_timelines[record_id], start_samp, end_samp
        )
        beats_clipped = self._clip_beat_timeline(
            self._beats_with_time[record_id], start_samp, end_samp, source_hz
        )

        activity_index: Dict[str, List[LTAFBoutRecord]] = defaultdict(list)
        for bout in rhythm_clipped:
            if bout.activity in self._regime_set:
                activity_index[bout.activity].append(bout)

        signals: Optional[np.ndarray] = None
        if load_signals:
            signals = load_bout_signal(record_id, start_samp, end_samp)

        return LTAFRecordingSample(
            record_id=record_id,
            window_start_ms=int(window_start_ms),
            window_end_ms=int(window_start_ms + ctx_ms),
            source_hz=source_hz,
            rhythm_timeline=rhythm_clipped,
            beats_timeline=beats_clipped,
            activity_index=dict(activity_index),
            signals=signals,
        )

    def load_signals(self, sample: LTAFRecordingSample) -> np.ndarray:
        """Materialize the signal for a recording sample (lazy path)."""
        if sample.signals is not None:
            return sample.signals
        start_samp = _ms_to_samples(sample.window_start_ms, sample.source_hz)
        end_samp = start_samp + _ms_to_samples(sample.duration_ms, sample.source_hz)
        signals = load_bout_signal(sample.record_id, start_samp, end_samp)
        sample.signals = signals
        return signals

    # ------------------------------------------------------------------
    # Clipping helpers (sample-space, per LTAFBoutRecord schema)
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_rhythm_timeline(
        timeline: List[LTAFBoutRecord],
        win_start_samp: int,
        win_end_samp: int,
    ) -> List[LTAFBoutRecord]:
        if not timeline:
            return []
        out: List[LTAFBoutRecord] = []
        for bout in timeline:
            if bout.end_sample <= win_start_samp or bout.start_sample >= win_end_samp:
                continue
            new_start = max(bout.start_sample, win_start_samp) - win_start_samp
            new_end = min(bout.end_sample, win_end_samp) - win_start_samp
            out.append(
                LTAFBoutRecord(
                    start_sample=int(new_start),
                    end_sample=int(new_end),
                    activity=bout.activity,
                    duration_samples=int(new_end - new_start),
                    clipped_left=bool(bout.start_sample < win_start_samp),
                    clipped_right=bool(bout.end_sample > win_end_samp),
                )
            )
        out.sort(key=lambda b: b.start_sample)
        return out

    @staticmethod
    def _clip_beat_timeline(
        beats: List[LTAFBeatEvent],
        win_start_samp: int,
        win_end_samp: int,
        source_hz: int,
    ) -> List[LTAFBeatEvent]:
        if not beats:
            return []
        # Use bisect for O(log n + k) instead of O(n).
        samples = [b.sample for b in beats]
        lo = bisect.bisect_left(samples, win_start_samp)
        hi = bisect.bisect_left(samples, win_end_samp)
        out: List[LTAFBeatEvent] = []
        for i in range(lo, hi):
            b = beats[i]
            new_samp = int(b.sample - win_start_samp)
            out.append(
                LTAFBeatEvent(
                    sample=new_samp,
                    time_ms=_samples_to_ms(new_samp, source_hz),
                    symbol=b.symbol,
                )
            )
        return out
