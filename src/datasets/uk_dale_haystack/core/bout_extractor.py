"""
Contextual-ON hysteresis bout extractor.

Implements the contextual-ON absorption + hysteresis algorithm:

  1. mask = power_w > on_threshold_w
  2. RLE -> runs of consecutive ON / OFF
  3. ABSORB: any OFF run shorter than min_off_duration_s is re-marked ON
     (this is what merges washing-machine cycle dips into a single bout)
  4. After absorption, drop ON runs shorter than min_on_duration_s
  5. Each surviving ON run is one bout. Compute peak_w, mean_w, total_kwh
  6. Optionally trim trailing edge using off_threshold_w (hysteresis)

The algorithm operates on an irregular timeline (UK-DALE has missing samples
and 6-12 s nominal sampling). All durations are computed in seconds from the
timestamps array, not samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawBout:
    start_idx: int
    end_idx: int       # exclusive
    start_ns: int
    end_ns: int        # last sample's timestamp; bout is [start_ns, end_ns]
    duration_s: float
    peak_w: float
    mean_w: float
    total_kwh: float


def _rle_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length-encode a boolean mask.

    Returns (run_starts, run_lengths, run_values) where len(...) == n_runs.
    """
    n = mask.shape[0]
    if n == 0:
        return (np.empty(0, dtype="int64"),
                np.empty(0, dtype="int64"),
                np.empty(0, dtype="bool"))
    # Edges where the value changes
    diffs = np.diff(mask.astype("int8"))
    change_idx = np.flatnonzero(diffs) + 1
    starts = np.concatenate([[0], change_idx])
    ends = np.concatenate([change_idx, [n]])
    lengths = ends - starts
    values = mask[starts]
    return starts, lengths, values


def extract_bouts(
    power_w: np.ndarray,
    ts_ns: np.ndarray,
    on_threshold_w: float,
    off_threshold_w: float,
    min_on_duration_s: float,
    min_off_duration_s: float,
    max_gap_factor: float = 4.0,
    nominal_dt_s: float = 6.0,
) -> list[RawBout]:
    """Extract bouts via contextual-ON hysteresis. See module docstring.

    Parameters
    ----------
    power_w, ts_ns
        Parallel arrays for one meter, sorted by timestamp. ``ts_ns`` is in
        nanoseconds since the Unix epoch.
    on_threshold_w, off_threshold_w
        ``off_threshold_w < on_threshold_w`` by convention -- the off threshold
        gives a small hysteresis band so brief dips just below ``on_threshold_w``
        don't terminate a bout. (For long_cycle absorption the more important
        knob is ``min_off_duration_s``.)
    min_on_duration_s, min_off_duration_s
        See module docstring. ``min_off_duration_s`` is the absorption window.
    max_gap_factor
        Bouts whose internal sample gap exceeds ``max_gap_factor * nominal_dt_s``
        are rejected (the gap likely represents a missing-data interval where we
        can't tell whether the appliance was actually on).
    """
    if power_w.size == 0:
        return []
    if power_w.shape != ts_ns.shape:
        raise ValueError("power_w and ts_ns must have the same shape")

    # 1. on/off mask
    mask = power_w > on_threshold_w

    # Bookkeeping for absorption: we need durations of OFF runs in seconds.
    starts, lengths, values = _rle_runs(mask)
    if starts.size == 0:
        return []

    # The duration of run i is the time difference between the *first* sample
    # of the next run and the first sample of run i. For the last run we use
    # the recording end (last_ts + nominal_dt).
    next_start_ns = np.concatenate([
        ts_ns[starts[1:]] if starts.size > 1 else np.empty(0, dtype="int64"),
        np.array([ts_ns[-1] + int(nominal_dt_s * 1e9)], dtype="int64"),
    ])
    run_start_ns = ts_ns[starts]
    run_durations_s = (next_start_ns - run_start_ns).astype("float64") / 1e9

    # 3. ABSORB: short OFF runs become ON
    short_off = (~values) & (run_durations_s < min_off_duration_s)
    if short_off.any():
        # Materialise mask edits
        new_mask = mask.copy()
        for i in np.flatnonzero(short_off):
            seg_start = starts[i]
            seg_end = seg_start + lengths[i]
            new_mask[seg_start:seg_end] = True
        # Re-RLE
        starts, lengths, values = _rle_runs(new_mask)
        next_start_ns = np.concatenate([
            ts_ns[starts[1:]] if starts.size > 1 else np.empty(0, dtype="int64"),
            np.array([ts_ns[-1] + int(nominal_dt_s * 1e9)], dtype="int64"),
        ])
        run_start_ns = ts_ns[starts]
        run_durations_s = (next_start_ns - run_start_ns).astype("float64") / 1e9

    # 4 + 5: keep ON runs >= min_on, compute stats
    bouts: list[RawBout] = []
    gap_threshold_ns = int(max_gap_factor * nominal_dt_s * 1e9)
    for i in np.flatnonzero(values):
        dur_s = float(run_durations_s[i])
        if dur_s < min_on_duration_s:
            continue
        seg_start = int(starts[i])
        seg_end = int(seg_start + lengths[i])
        if seg_end - seg_start < 1:
            continue

        seg_p = power_w[seg_start:seg_end]
        seg_t = ts_ns[seg_start:seg_end]

        # Reject bouts that internally span a missing-data gap
        if seg_t.size > 1:
            internal_gaps = np.diff(seg_t.astype("int64"))
            if np.any(internal_gaps > gap_threshold_ns):
                continue

        # 6: hysteresis trim -- drop leading AND trailing samples below
        # off_threshold_w (these were dragged in by the absorption step).
        # Trimming both edges prevents bouts from carrying silent zero-padding
        # at the start, which otherwise makes inserted needles appear shifted
        # away from their declared insert_position_samples.
        keep_idx = np.flatnonzero(seg_p > off_threshold_w)
        if keep_idx.size == 0:
            continue
        first_kept = int(keep_idx[0])
        last_kept = int(keep_idx[-1])
        seg_p = seg_p[first_kept : last_kept + 1]
        seg_t = seg_t[first_kept : last_kept + 1]

        # Recompute true duration as (last_ts + nominal_dt) - first_ts so a
        # one-sample bout has duration nominal_dt rather than 0.
        bout_dur_s = (
            (int(seg_t[-1]) + int(nominal_dt_s * 1e9) - int(seg_t[0]))
            / 1e9
        )
        peak_w = float(seg_p.max())
        mean_w = float(seg_p.mean())
        # Trapezoid energy estimate (cap dt at 4 * nominal to avoid blow-up)
        if seg_t.size > 1:
            dt_s = np.diff(seg_t.astype("int64")).astype("float64") / 1e9
            dt_s = np.where(dt_s > max_gap_factor * nominal_dt_s, nominal_dt_s, dt_s)
            energy_ws = float(np.sum(seg_p[:-1].astype("float64") * dt_s))
        else:
            energy_ws = float(seg_p[0]) * nominal_dt_s
        total_kwh = energy_ws / 1000.0 / 3600.0

        bouts.append(RawBout(
            start_idx=seg_start,
            end_idx=seg_start + len(seg_p),
            start_ns=int(seg_t[0]),
            end_ns=int(seg_t[-1]),
            duration_s=bout_dur_s,
            peak_w=peak_w,
            mean_w=mean_w,
            total_kwh=total_kwh,
        ))
    return bouts


def naive_runs(
    power_w: np.ndarray,
    ts_ns: np.ndarray,
    on_threshold_w: float,
    nominal_dt_s: float = 6.0,
) -> list[tuple[int, int, float]]:
    """Reference: simple `power_w > threshold` runs, for plot comparison.

    Returns list of (start_ns, end_ns, duration_s).
    """
    mask = power_w > on_threshold_w
    starts, lengths, values = _rle_runs(mask)
    out = []
    for i in np.flatnonzero(values):
        s = int(starts[i])
        e = int(s + lengths[i] - 1)
        dur = (int(ts_ns[e]) - int(ts_ns[s]) + int(nominal_dt_s * 1e9)) / 1e9
        out.append((int(ts_ns[s]), int(ts_ns[e]), float(dur)))
    return out
