"""Tests for the contextual-ON bout extractor."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.uk_dale_haystack.core.bout_extractor import (
    extract_bouts,
    naive_runs,
)


def _make_signal_with_dips(
    pattern: list[tuple[float, float]], dt_s: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """pattern: list of (power_w, duration_s). Returns (power, ts_ns)."""
    powers: list[float] = []
    for power, dur in pattern:
        n = max(1, int(round(dur / dt_s)))
        powers.extend([power] * n)
    p = np.asarray(powers, dtype="float32")
    ts = (np.arange(p.size, dtype="int64") * int(dt_s * 1e9))
    return p, ts


def test_naive_vs_contextual_washer_cycle():
    """A simulated washer cycle (motor + heater bursts with cycle dips) should
    fragment under naive thresholding and consolidate under contextual-ON."""
    pattern = [
        (5.0, 60.0),     # idle
        (2200.0, 30.0),  # motor burst
        (5.0, 60.0),     # cycle dip
        (1800.0, 60.0),  # heater
        (5.0, 60.0),     # cycle dip
        (1500.0, 60.0),  # rinse
        (5.0, 60.0),     # idle (this one >= min_off so should split bouts)
    ]
    p, ts = _make_signal_with_dips(pattern)

    # naive: threshold > 20W -> 3 separate runs
    naive = naive_runs(p, ts, on_threshold_w=20.0)
    assert len(naive) == 3

    # contextual: min_off=120s absorbs the 60 s cycle dips -> 1 bout
    bouts = extract_bouts(
        p, ts,
        on_threshold_w=20.0, off_threshold_w=10.0,
        min_on_duration_s=60.0, min_off_duration_s=120.0,
    )
    assert len(bouts) == 1
    bout = bouts[0]
    assert bout.peak_w == pytest.approx(2200.0, rel=1e-2)
    # Bout spans the 3 active blocks plus the 2 absorbed dips
    expected_dur = 30.0 + 60.0 + 60.0 + 60.0 + 60.0  # active + dips
    assert bout.duration_s == pytest.approx(expected_dur, rel=0.1)


def test_kettle_no_absorption_needed():
    """Kettle has impulse signature: naive == contextual for short bouts."""
    pattern = [
        (5.0, 60.0),
        (3000.0, 120.0),   # kettle bout 1
        (5.0, 600.0),
        (3000.0, 90.0),    # kettle bout 2
        (5.0, 60.0),
    ]
    p, ts = _make_signal_with_dips(pattern)
    bouts = extract_bouts(
        p, ts,
        on_threshold_w=2000.0, off_threshold_w=100.0,
        min_on_duration_s=12.0, min_off_duration_s=12.0,
    )
    assert len(bouts) == 2
    assert all(b.peak_w == pytest.approx(3000.0, rel=1e-2) for b in bouts)


def test_min_on_filters_short_runs():
    pattern = [
        (5.0, 60.0),
        (3000.0, 6.0),     # 1-sample blip < min_on
        (5.0, 60.0),
        (3000.0, 30.0),    # real bout
        (5.0, 60.0),
    ]
    p, ts = _make_signal_with_dips(pattern)
    bouts = extract_bouts(
        p, ts,
        on_threshold_w=2000.0, off_threshold_w=100.0,
        min_on_duration_s=12.0, min_off_duration_s=12.0,
    )
    assert len(bouts) == 1
    assert bouts[0].peak_w == pytest.approx(3000.0, rel=1e-2)


def test_empty_signal_returns_no_bouts():
    p = np.zeros(0, dtype="float32")
    ts = np.zeros(0, dtype="int64")
    assert extract_bouts(p, ts, 2000.0, 100.0, 12.0, 12.0) == []
