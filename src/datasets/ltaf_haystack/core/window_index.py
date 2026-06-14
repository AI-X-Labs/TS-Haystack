# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Per-(record, context-length) window index for LTAF-Haystack.

For each record, enumerates candidate window starts (stepped by stride) such
that the window contains at least one bout from any rhythm regime. Each
window entry carries:

  * A rhythm presence bitmask (9 bits, one per rhythm in the canonical order
    from ``activity_regimes.py``). O(1) filtering for existence/ordering.
  * Per-beat-type counts ``(N, A, V, Q)`` for the window. O(1) filtering for
    ``anomaly_detection`` ("≥1 V beat") and ``anomaly_localization``.

Persisted as JSON per (label_class, context_length_s) so generation is
reproducible and fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.datasets.ltaf_haystack.core.activity_regimes import (
    BEAT_EVENT_TYPES,
    get_activities_list,
)
from src.datasets.ltaf_haystack.core.ltaf_timeline_builder import (
    LTAF_HAYSTACK_DIR,
    LTAFTimelineBuilder,
    get_ltaf_beat_timeline_path,
    get_ltaf_timeline_path,
)
from src.datasets.ltaf_haystack.loader import SOURCE_HZ


def default_stride_ms(context_length_ms: int) -> int:
    """`max(1s, min(ctx/4, 30min))` per plan Phase-config."""
    cap = 30 * 60 * 1000
    return max(1000, min(context_length_ms // 4, cap))


def get_window_index_dir(label_class: str) -> Path:
    return LTAF_HAYSTACK_DIR / label_class / "window_index"


def get_window_index_path(label_class: str, context_length_s: float) -> Path:
    return get_window_index_dir(label_class) / f"ctx_{int(context_length_s)}s.json"


def _ms_to_samples(ms: int, source_hz: int = SOURCE_HZ) -> int:
    return int(round(ms * source_hz / 1000.0))


def _load_rhythm_timeline(record_id: str) -> List:
    path = get_ltaf_timeline_path(record_id)
    if not path.exists():
        return []
    return LTAFTimelineBuilder._load_timeline(path).timeline


def _load_beat_samples_by_symbol(record_id: str) -> Dict[str, List[int]]:
    """Return ``{symbol: [sample, ...]}`` with samples sorted ascending."""
    path = get_ltaf_beat_timeline_path(record_id)
    if not path.exists():
        return {s: [] for s in BEAT_EVENT_TYPES}

    import polars as pl  # local import keeps top-level cheap for consumers

    df = pl.read_parquet(path)
    by: Dict[str, List[int]] = {s: [] for s in BEAT_EVENT_TYPES}
    if len(df) == 0:
        return by
    for row in df.iter_rows(named=True):
        sym = str(row["symbol"])
        if sym in by:
            by[sym].append(int(row["sample"]))
    for sym in by:
        by[sym].sort()
    return by


def _count_beats_in_window(
    samples_sorted: List[int],
    win_start_samp: int,
    win_end_samp: int,
) -> int:
    """Binary-search count of samples in ``[start, end)``."""
    import bisect

    lo = bisect.bisect_left(samples_sorted, win_start_samp)
    hi = bisect.bisect_left(samples_sorted, win_end_samp)
    return hi - lo


class LTAFWindowIndex:
    """Build / load / query candidate windows.

    Index layout on disk::

        {
          "label_class": "rhythms",
          "context_length_s": 900,
          "stride_ms": 225000,
          "regime": ["NSR","AFIB","SBR","AB","B","T","SVTA","VT","IVR"],
          "beat_symbols": ["N","A","V","Q"],
          "records": {
            "00": {"starts": [0, 225000, ...],
                   "masks":  [1, 3, ...],
                   "beat_counts": [[1024, 0, 3, 0], ...]}
          }
        }
    """

    def __init__(
        self,
        label_class: str,
        context_length_s: float,
        windows_by_subject: Dict[str, Dict[str, List]],
        regime: Optional[List[str]] = None,
        stride_ms: Optional[int] = None,
        beat_symbols: Optional[List[str]] = None,
        source_hz: int = SOURCE_HZ,
    ):
        self.label_class = label_class
        self.context_length_s = float(context_length_s)
        self.windows_by_subject = windows_by_subject
        self.regime = regime or get_activities_list(label_class)
        self.beat_symbols = beat_symbols or list(BEAT_EVENT_TYPES)
        self._stride_ms = stride_ms
        self.source_hz = int(source_hz)

    @property
    def context_length_ms(self) -> int:
        return int(self.context_length_s * 1000)

    def subjects(self) -> List[str]:
        return [s for s, w in self.windows_by_subject.items() if w.get("starts")]

    def activity_bit(self, activity: str) -> int:
        return 1 << self.regime.index(activity)

    def beat_count_idx(self, symbol: str) -> int:
        return self.beat_symbols.index(symbol)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        label_class: str,
        context_length_s: float,
        record_ids: List[str],
        stride_ms: Optional[int] = None,
        source_hz: int = SOURCE_HZ,
        verbose: bool = True,
    ) -> "LTAFWindowIndex":
        regime = get_activities_list(label_class)
        regime_set = set(regime)
        bit_for = {a: 1 << i for i, a in enumerate(regime)}
        full_mask = (1 << len(regime)) - 1
        beat_symbols = list(BEAT_EVENT_TYPES)

        ctx_ms = int(context_length_s * 1000)
        stride = stride_ms if stride_ms is not None else default_stride_ms(ctx_ms)

        def compute_rhythm_mask(sorted_bouts, win_start_samp: int, win_end_samp: int) -> int:
            mask = 0
            for bout in sorted_bouts:
                if bout.end_sample <= win_start_samp:
                    continue
                if bout.start_sample >= win_end_samp:
                    break
                if bout.activity in regime_set:
                    mask |= bit_for[bout.activity]
                    if mask == full_mask:
                        return mask
            return mask

        windows_by_subject: Dict[str, Dict[str, List]] = {}

        iterator = tqdm(
            record_ids,
            desc=f"Indexing {int(context_length_s)}s windows",
            disable=not verbose,
        )
        for rid in iterator:
            bouts = _load_rhythm_timeline(rid)
            if not bouts:
                continue
            sorted_bouts = sorted(bouts, key=lambda b: b.start_sample)
            duration_samp = sorted_bouts[-1].end_sample
            if duration_samp < _ms_to_samples(ctx_ms, source_hz):
                continue

            beats_by_sym = _load_beat_samples_by_symbol(rid)

            starts: List[int] = []
            masks: List[int] = []
            beat_counts: List[Tuple[int, int, int, int]] = []

            start_ms = 0
            last_start_ms = int(duration_samp * 1000.0 / source_hz) - ctx_ms
            ctx_samp = _ms_to_samples(ctx_ms, source_hz)
            while start_ms <= last_start_ms:
                start_samp = _ms_to_samples(start_ms, source_hz)
                end_samp = start_samp + ctx_samp
                mask = compute_rhythm_mask(sorted_bouts, start_samp, end_samp)
                if mask != 0:
                    counts = tuple(
                        _count_beats_in_window(beats_by_sym[s], start_samp, end_samp)
                        for s in beat_symbols
                    )
                    starts.append(start_ms)
                    masks.append(mask)
                    beat_counts.append(counts)
                start_ms += stride

            if starts:
                windows_by_subject[rid] = {
                    "starts": starts,
                    "masks": masks,
                    "beat_counts": [list(c) for c in beat_counts],
                }

        return cls(
            label_class=label_class,
            context_length_s=context_length_s,
            windows_by_subject=windows_by_subject,
            regime=regime,
            stride_ms=stride,
            beat_symbols=beat_symbols,
            source_hz=source_hz,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> Path:
        path = get_window_index_path(self.label_class, self.context_length_s)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label_class": self.label_class,
            "context_length_s": self.context_length_s,
            "stride_ms": self._stride_ms,
            "regime": self.regime,
            "beat_symbols": self.beat_symbols,
            "source_hz": self.source_hz,
            "records": self.windows_by_subject,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        return path

    @classmethod
    def load(cls, label_class: str, context_length_s: float) -> "LTAFWindowIndex":
        path = get_window_index_path(label_class, context_length_s)
        if not path.exists():
            raise FileNotFoundError(f"LTAF window index not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload.get("records") or payload.get("subjects") or {}
        if records:
            first = next(iter(records.values()))
            if "beat_counts" not in first:
                raise FileNotFoundError(
                    f"Window index at {path} lacks beat_counts. Delete it and rebuild."
                )

        return cls(
            label_class=payload["label_class"],
            context_length_s=payload["context_length_s"],
            windows_by_subject=records,
            regime=payload.get("regime"),
            stride_ms=payload.get("stride_ms"),
            beat_symbols=payload.get("beat_symbols", list(BEAT_EVENT_TYPES)),
            source_hz=payload.get("source_hz", SOURCE_HZ),
        )

    @classmethod
    def get_or_build(
        cls,
        label_class: str,
        context_length_s: float,
        record_ids: List[str],
        stride_ms: Optional[int] = None,
        source_hz: int = SOURCE_HZ,
        verbose: bool = True,
    ) -> "LTAFWindowIndex":
        try:
            return cls.load(label_class, context_length_s)
        except FileNotFoundError:
            idx = cls.build(
                label_class=label_class,
                context_length_s=context_length_s,
                record_ids=record_ids,
                stride_ms=stride_ms,
                source_hz=source_hz,
                verbose=verbose,
            )
            idx.save()
            return idx

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def windows_with_beat_count_gte(
        self,
        beat_type: str,
        min_count: int,
        record_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, int]]:
        """Return ``(record_id, start_ms)`` pairs where the window has ≥ ``min_count``
        beats of ``beat_type``."""
        idx = self.beat_count_idx(beat_type)
        subset = None if record_ids is None else set(record_ids)
        out: List[Tuple[str, int]] = []
        for rid, entry in self.windows_by_subject.items():
            if subset is not None and rid not in subset:
                continue
            for start_ms, counts in zip(entry["starts"], entry["beat_counts"]):
                if counts[idx] >= min_count:
                    out.append((rid, int(start_ms)))
        return out

    def total_windows(self) -> int:
        return sum(len(v["starts"]) for v in self.windows_by_subject.values())
