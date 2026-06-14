"""
Sample real mains windows where a target appliance is OFF for the entire window.

No precomputed background index: the Phase 2 bout index already enumerates the
ON intervals of every appliance, so OFF intervals are its complement and can be
computed cheaply at runtime (low-thousands of bouts per (house, appliance)).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl

from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.core.data_structures import (
    BackgroundSample,
    BoutRef,
)
from src.datasets.uk_dale_haystack.loader import (
    UKD_HAYSTACK_DIR,
    load_meter_window_grid,
    mains_meter_id,
    meter_gaps,
    NOMINAL_DT_S,
)


@dataclass(frozen=True)
class _HouseTimeframe:
    house_id: int
    start_ns: int
    end_ns: int


def _load_house_timeframes() -> dict[int, _HouseTimeframe]:
    manifest = json.loads((UKD_HAYSTACK_DIR / "manifest.json").read_text())
    out: dict[int, _HouseTimeframe] = {}
    for h_str, meta in manifest["houses"].items():
        tf = meta["timeframe"]
        start = pd.Timestamp(tf["start"]).tz_convert("UTC").value
        end = pd.Timestamp(tf["end"]).tz_convert("UTC").value
        out[int(h_str)] = _HouseTimeframe(int(h_str), int(start), int(end))
    return out


def _interval_complement(
    sorted_bouts: np.ndarray,    # (n, 2) int64 [start_ns, end_ns)
    full_start_ns: int,
    full_end_ns: int,
) -> np.ndarray:
    """Return OFF intervals = [full_start, full_end) \\ union(bouts).

    Output shape: (m, 2) int64.
    """
    if sorted_bouts.size == 0:
        return np.array([[full_start_ns, full_end_ns]], dtype="int64")

    out: list[tuple[int, int]] = []
    cursor = full_start_ns
    for s, e in sorted_bouts:
        s_i = int(max(s, full_start_ns))
        e_i = int(min(e, full_end_ns))
        if s_i > cursor:
            out.append((cursor, s_i))
        cursor = max(cursor, e_i)
        if cursor >= full_end_ns:
            break
    if cursor < full_end_ns:
        out.append((cursor, full_end_ns))
    return np.array(out, dtype="int64") if out else np.empty((0, 2), dtype="int64")


def _interval_subtract(
    intervals: np.ndarray,        # (n, 2)
    blockers: list[tuple[int, int, float]],
    pad_ns: int,
) -> np.ndarray:
    """Subtract blocker intervals (with optional pad) from intervals.

    blockers come from the conversion-manifest gap list -- (start_ns, end_ns, dt_s).
    """
    if not blockers:
        return intervals
    blocker_arr = np.array(
        [(s - pad_ns, e + pad_ns) for s, e, _ in blockers],
        dtype="int64",
    )
    blocker_arr = blocker_arr[np.argsort(blocker_arr[:, 0])]

    result: list[tuple[int, int]] = []
    for iv_start, iv_end in intervals:
        cursor = int(iv_start)
        # Quick locate of relevant blockers
        idx = np.searchsorted(blocker_arr[:, 1], cursor, side="right")
        for bs, be in blocker_arr[idx:]:
            if bs >= iv_end:
                break
            bs = int(bs); be = int(be)
            if bs > cursor:
                result.append((cursor, min(bs, int(iv_end))))
            cursor = max(cursor, be)
            if cursor >= iv_end:
                break
        if cursor < iv_end:
            result.append((cursor, int(iv_end)))
    return np.array(result, dtype="int64") if result else np.empty((0, 2), dtype="int64")


class BackgroundSampler:
    """Sample target-OFF mains windows.

    The bout index is filtered to the requested split once at construction;
    all subsequent .sample() calls reuse the cached frame.
    """

    def __init__(
        self,
        bout_index: pl.DataFrame,
        split: str,
        dt_s: float = NOMINAL_DT_S,
    ):
        self.split = split
        self.dt_s = float(dt_s)
        self._bouts = bout_index.filter(pl.col("split") == split).select([
            "house_id", "appliance", "start_ns", "end_ns",
            "duration_s", "peak_w", "mean_w",
        ])
        # Houses present in the split
        self._houses_present = sorted({int(h) for h in self._bouts["house_id"].unique().to_list()})
        # Per-(house, appliance) cached numpy arrays of [start_ns, end_ns]
        self._cache: dict[tuple[int, str], np.ndarray] = {}
        self._timeframes = _load_house_timeframes()
        # All v1-vocab bouts per house, for "other_bouts" lookup
        self._all_bouts_per_house: dict[int, np.ndarray] = {}
        # Houses that have a bout for the target appliance (per appliance)
        self._appliance_houses: dict[str, list[int]] = {}
        for app in V1_VOCAB:
            houses = sorted({
                int(h) for h in self._bouts.filter(pl.col("appliance") == app)["house_id"].unique().to_list()
            })
            if houses:
                self._appliance_houses[app] = houses

    def _bouts_for(self, house: int, appliance: str) -> np.ndarray:
        key = (house, appliance)
        if key not in self._cache:
            df = (self._bouts
                  .filter((pl.col("house_id") == house) & (pl.col("appliance") == appliance))
                  .select(["start_ns", "end_ns"])
                  .sort("start_ns"))
            self._cache[key] = df.to_numpy().astype("int64")
        return self._cache[key]

    def _all_bouts_for(self, house: int) -> tuple[np.ndarray, list[str]]:
        """Return ((n,2) int64 [start, end], list of appliance strings) for the house."""
        if house not in self._all_bouts_per_house:
            df = (self._bouts
                  .filter(pl.col("house_id") == house)
                  .select(["start_ns", "end_ns", "appliance",
                           "duration_s", "peak_w", "mean_w"])
                  .sort("start_ns"))
            self._all_bouts_per_house[house] = df
        df = self._all_bouts_per_house[house]
        return df

    @property
    def appliances_with_bouts(self) -> list[str]:
        return sorted(self._appliance_houses.keys())

    def houses_for_target(self, target: str) -> list[int]:
        return list(self._appliance_houses.get(target, []))

    def sample(
        self,
        target: str,
        context_length_s: int | float,
        rng: np.random.Generator,
        *,
        min_other_bouts: int = 0,
        same_house_as: int | None = None,
        max_attempts: int = 50,
        margin_s: float | None = None,
        allow_target_on: bool = False,
        extra_off_targets: list[str] | None = None,
    ) -> BackgroundSample | None:
        """Sample a window where ``target`` is OFF (default) or anywhere
        (``allow_target_on=True``).

        ``allow_target_on=True`` drops the target-OFF complement step and
        samples uniformly from the house's full recording timeframe. The
        target's own natural bouts are then included in the returned
        ``other_bouts`` list so callers can account for them.

        ``extra_off_targets`` lists additional appliances that must also
        be OFF for the entire window. The OFF complement is then computed
        against the *union* of bouts of ``target`` plus every
        ``extra_off_targets`` entry. Used by ordering / antecedent /
        multi_hop, where natural bouts of the second appliance would
        otherwise create an ambiguous ground-truth answer.
        """
        ctx_s = float(context_length_s)
        if margin_s is None:
            margin_s = min(0.05 * ctx_s, 60.0)
        margin_ns = int(margin_s * 1e9)
        ctx_ns = int(ctx_s * 1e9)
        min_off_ns = ctx_ns + 2 * margin_ns

        if same_house_as is not None:
            candidate_houses = [int(same_house_as)] if same_house_as in self._houses_present else []
        else:
            candidate_houses = [
                h for h in self._houses_present
                if (target not in self._appliance_houses) or (h in self._appliance_houses[target])
            ] or self._houses_present

        if not candidate_houses:
            return None

        extras = list(extra_off_targets or [])

        for attempt in range(max_attempts):
            house = candidate_houses[rng.integers(0, len(candidate_houses))]
            tf = self._timeframes[house]
            if allow_target_on:
                # No bout-complement -- the entire recording timeframe is fair game.
                off_intervals = np.array([[tf.start_ns, tf.end_ns]], dtype="int64")
            else:
                bout_arrays = [self._bouts_for(house, target)]
                for extra in extras:
                    bout_arrays.append(self._bouts_for(house, extra))
                non_empty = [a for a in bout_arrays if a.size]
                if non_empty:
                    merged = np.concatenate(non_empty, axis=0)
                    merged = merged[np.argsort(merged[:, 0])]
                else:
                    merged = np.empty((0, 2), dtype="int64")
                off_intervals = _interval_complement(merged, tf.start_ns, tf.end_ns)

            # Subtract mains-meter gaps (any gap > 24 s in the mains data).
            # Pad each gap by margin so the window can't sit right at a gap edge.
            gaps = meter_gaps(house, mains_meter_id(house))
            off_intervals = _interval_subtract(off_intervals, gaps, pad_ns=margin_ns)

            # Filter for intervals long enough to fit ctx + 2*margin
            long_enough = off_intervals[
                (off_intervals[:, 1] - off_intervals[:, 0]) >= min_off_ns
            ]
            if long_enough.size == 0:
                continue

            # Pick an interval weighted by (slack length)
            slacks = (long_enough[:, 1] - long_enough[:, 0] - min_off_ns).astype("float64")
            total_slack = float(slacks.sum() + len(slacks))  # +1 each so always positive
            probs = (slacks + 1.0) / total_slack
            iv_idx = int(rng.choice(len(long_enough), p=probs))
            iv_start, iv_end = int(long_enough[iv_idx, 0]), int(long_enough[iv_idx, 1])

            # Random window start inside the interval, leaving margin slack
            start_ns_low = iv_start + margin_ns
            start_ns_high = iv_end - ctx_ns - margin_ns
            window_start_ns = int(rng.integers(start_ns_low, start_ns_high + 1))
            window_end_ns = window_start_ns + ctx_ns

            # Pull mains on the regular grid
            mains_w = load_meter_window_grid(
                house, mains_meter_id(house), window_start_ns, window_end_ns,
                dt_s=self.dt_s,
            )

            # other_bouts metadata: any v1 bout intersecting the window. When
            # allow_target_on=True we INCLUDE target bouts so the caller can
            # see how many were already present.
            other_bouts = self._other_bouts_in_window(
                house, target, window_start_ns, window_end_ns,
                include_target=allow_target_on,
            )
            if len(other_bouts) < min_other_bouts:
                continue

            tz_local = "Europe/London"
            t_start = pd.Timestamp(window_start_ns, tz="UTC").tz_convert(tz_local)
            t_end = pd.Timestamp(window_end_ns, tz="UTC").tz_convert(tz_local)
            return BackgroundSample(
                house_id=house,
                start_ns=window_start_ns,
                end_ns=window_end_ns,
                dt_s=self.dt_s,
                mains_w=mains_w.astype("float32", copy=False),
                other_bouts=other_bouts,
                recording_time_context=(t_start.strftime("%H:%M"), t_end.strftime("%H:%M")),
            )
        return None

    def _other_bouts_in_window(
        self, house: int, target: str, window_start_ns: int, window_end_ns: int,
        *, include_target: bool = False,
    ) -> list[BoutRef]:
        df = self._all_bouts_for(house)
        ws, we = int(window_start_ns), int(window_end_ns)
        starts = df["start_ns"].to_numpy()
        ends = df["end_ns"].to_numpy()
        apps = df["appliance"].to_numpy()
        durs = df["duration_s"].to_numpy()
        peaks = df["peak_w"].to_numpy()
        means = df["mean_w"].to_numpy()
        # Intersection: starts < we AND ends > ws (and !=target unless including)
        mask = (starts < we) & (ends > ws)
        if not include_target:
            mask &= (apps != target)
        out: list[BoutRef] = []
        dt_ns = int(self.dt_s * 1e9)
        for i in np.flatnonzero(mask):
            s_clip = max(int(starts[i]), ws)
            e_clip = min(int(ends[i]), we)
            ss = (s_clip - ws) // dt_ns
            ee = (e_clip - ws) // dt_ns
            out.append(BoutRef(
                appliance=str(apps[i]),
                start_sample=int(ss),
                end_sample=int(ee + 1),
                duration_s=float(durs[i]),
                peak_w=float(peaks[i]),
                mean_w=float(means[i]),
            ))
        return out
