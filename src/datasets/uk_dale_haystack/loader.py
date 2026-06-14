"""
Memmap loader for UK-DALE-Haystack signals.

Two sidecars per (building, meter) live under
  data/uk_dale/uk_dale_haystack/signals/h{B}/m{M}.npy   (float32 watts, (N,))
  data/uk_dale/uk_dale_haystack/signals/h{B}/m{M}.t.npy (int64   ns,    (N,))

Window slicing is O(log N) via np.searchsorted on the timestamp memmap.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

UK_DALE_DIR = Path("data/uk_dale")
UKD_HAYSTACK_DIR = UK_DALE_DIR / "uk_dale_haystack"
SIGNALS_DIR = UKD_HAYSTACK_DIR / "signals"
MANIFEST_PATH = UKD_HAYSTACK_DIR / "manifest.json"
CONVERSION_MANIFEST_PATH = SIGNALS_DIR / "conversion_manifest.json"

NOMINAL_DT_S = 6.0  # UK-DALE 6 s sampling
NOMINAL_HZ = 1.0 / NOMINAL_DT_S


# ---------------------------------------------------------------------------
# Manifest accessors (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} missing; run scripts/data/uk_dale/build_uk_dale_manifest.py first."
        )
    return json.loads(MANIFEST_PATH.read_text())


@lru_cache(maxsize=1)
def load_conversion_manifest() -> dict[str, Any]:
    if not CONVERSION_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{CONVERSION_MANIFEST_PATH} missing; run scripts/data/uk_dale/convert_uk_dale_to_npy.py first."
        )
    return json.loads(CONVERSION_MANIFEST_PATH.read_text())


def list_meters(building: int) -> dict[int, str]:
    """{meter_id: kind} for a house, where kind is 'mains' or 'submeter:<canon>'."""
    cm = load_conversion_manifest()
    out: dict[int, str] = {}
    for key, info in cm["per_meter"].items():
        # key = "h{B}_m{M}"
        b_str, m_str = key.split("_")
        b = int(b_str[1:])
        m = int(m_str[1:])
        if b == building:
            out[m] = info["kind"]
    return out


def mains_meter_id(building: int) -> int:
    return int(load_manifest()["houses"][str(building)]["mains_meter"])


def meter_gaps(building: int, meter: int) -> list[tuple[int, int, float]]:
    """List of (gap_start_ns, gap_end_ns, dt_s) gaps > 24 s in this meter."""
    cm = load_conversion_manifest()
    info = cm["per_meter"][f"h{building}_m{meter}"]
    return [(int(s), int(e), float(dt)) for s, e, dt in info.get("gaps", [])]


# ---------------------------------------------------------------------------
# Memmap loaders
# ---------------------------------------------------------------------------

def _signal_path(building: int, meter: int) -> Path:
    return SIGNALS_DIR / f"h{building}" / f"m{meter}.npy"


def _timestamps_path(building: int, meter: int) -> Path:
    return SIGNALS_DIR / f"h{building}" / f"m{meter}.t.npy"


def load_meter_signal_mmap(building: int, meter: int) -> np.memmap:
    """Read-only float32 (N,) watts."""
    return np.load(_signal_path(building, meter), mmap_mode="r")


def load_meter_timestamps_mmap(building: int, meter: int) -> np.memmap:
    """Read-only int64 (N,) Unix ns."""
    return np.load(_timestamps_path(building, meter), mmap_mode="r")


# ---------------------------------------------------------------------------
# Window access
# ---------------------------------------------------------------------------

def load_meter_window(
    building: int,
    meter: int,
    start_ns: int,
    end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_ns, power_w) for samples in [start_ns, end_ns).

    Uses np.searchsorted on the memmapped timestamp array -> O(log N) random
    access without loading the full meter into RAM.
    """
    ts = load_meter_timestamps_mmap(building, meter)
    sig = load_meter_signal_mmap(building, meter)
    lo = int(np.searchsorted(ts, start_ns, side="left"))
    hi = int(np.searchsorted(ts, end_ns, side="left"))
    if lo == hi:
        return (np.empty(0, dtype="int64"), np.empty(0, dtype="float32"))
    # Materialise to ndarray (copy out of memmap so callers can mutate)
    return (np.asarray(ts[lo:hi], dtype="int64"),
            np.asarray(sig[lo:hi], dtype="float32"))


def resample_to_grid(
    ts_ns: np.ndarray,
    values: np.ndarray,
    grid_start_ns: int,
    grid_end_ns: int,
    dt_s: float = NOMINAL_DT_S,
) -> np.ndarray:
    """Nearest-neighbour resample irregular (ts_ns, values) onto a regular grid.

    Returns an array of shape (n_samples,) where
      n_samples = round((grid_end_ns - grid_start_ns) / (dt_s * 1e9)).

    Out-of-range or no-source positions get 0.0 (mains "off"). For the typical
    UK-DALE case (median dt = 6 s, target grid = 6 s) this is essentially a
    no-op slice + minor edge alignment.
    """
    dt_ns = int(round(dt_s * 1e9))
    n_samples = int(round((grid_end_ns - grid_start_ns) / dt_ns))
    if n_samples <= 0:
        return np.zeros(0, dtype="float32")

    grid = grid_start_ns + np.arange(n_samples, dtype="int64") * dt_ns
    out = np.zeros(n_samples, dtype="float32")
    if ts_ns.size == 0:
        return out
    idx = np.searchsorted(ts_ns, grid, side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    # If the nearest preceding sample is older than 4 * dt_s, treat as missing
    too_old_ns = int(4 * dt_ns)
    age_ns = grid - ts_ns[np.clip(idx, 0, ts_ns.size - 1)]
    out[(age_ns > too_old_ns) | ~valid] = 0.0
    return out


def load_meter_window_grid(
    building: int,
    meter: int,
    start_ns: int,
    end_ns: int,
    dt_s: float = NOMINAL_DT_S,
) -> np.ndarray:
    """Return power_w on a regular dt_s grid spanning [start_ns, end_ns)."""
    ts, vals = load_meter_window(
        building, meter, start_ns - int(2 * dt_s * 1e9), end_ns,
    )
    return resample_to_grid(ts, vals, start_ns, end_ns, dt_s=dt_s)
