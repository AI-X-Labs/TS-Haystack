# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Loader utilities for the LTAF dataset.

Exposes memmap access to the .npy signal cache produced by
``scripts/data/convert_ltaf_to_npy.py`` plus thin .hea parsing helpers.
The natural-window sampler reads through this module so the underlying
storage format can change without touching the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

LTAF_DATA_DIR = Path("data/ltafdb")
LTAF_RAW_DIR = LTAF_DATA_DIR / "raw"
LTAF_TRAINING_DIR = LTAF_DATA_DIR / "training"

SOURCE_HZ = 128
CHANNEL_NAMES = ["ECG1", "ECG2"]
N_CHANNELS = 2


def get_record_dir(record_id: str) -> Path:
    return LTAF_TRAINING_DIR / record_id


def get_npy_path(record_id: str) -> Path:
    return get_record_dir(record_id) / f"{record_id}.npy"


def load_conversion_manifest() -> Dict[str, Dict]:
    path = LTAF_TRAINING_DIR / "conversion_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_all_record_ids() -> List[str]:
    """Enumerate record IDs from the conversion manifest (preferred) or disk."""
    manifest = load_conversion_manifest()
    if manifest:
        return sorted(manifest.keys())
    if LTAF_TRAINING_DIR.exists():
        return sorted(
            d.name
            for d in LTAF_TRAINING_DIR.iterdir()
            if d.is_dir() and (d / f"{d.name}.npy").exists()
        )
    return []


def parse_header(record_id: str) -> Dict:
    """Parse the WFDB .hea file for a record.

    Returns a dict with ``fs``, ``sig_len``, ``start_time``, ``record_id``.
    """
    hea_path = LTAF_RAW_DIR / f"{record_id}.hea"
    with open(hea_path) as f:
        lines = f.readlines()

    parts = lines[0].strip().split()
    n_signals = int(parts[1])
    fs = int(parts[2])
    sig_len = int(parts[3])
    start_time = parts[4] if len(parts) > 4 else None

    return {
        "record_id": record_id,
        "fs": fs,
        "n_signals": n_signals,
        "sig_len": sig_len,
        "start_time": start_time,
    }


def load_record_signals_mmap(record_id: str) -> np.ndarray:
    """Return a read-only memmap view of the record's (N, 2) float32 signals."""
    return np.load(str(get_npy_path(record_id)), mmap_mode="r")


def load_bout_signal(record_id: str, start_sample: int, end_sample: int) -> np.ndarray:
    """Return a contiguous float32 (end-start, 2) slice from the record memmap."""
    if end_sample <= start_sample:
        raise ValueError(
            f"end_sample ({end_sample}) must be > start_sample ({start_sample})"
        )
    mm = load_record_signals_mmap(record_id)
    n_total = mm.shape[0]
    start = max(0, min(int(start_sample), n_total))
    end = max(start, min(int(end_sample), n_total))
    return np.ascontiguousarray(mm[start:end]).astype(np.float32, copy=False)


def load_window_ms(
    record_id: str,
    window_start_ms: int,
    window_end_ms: int,
    source_hz: int = SOURCE_HZ,
) -> np.ndarray:
    """Hydrate a QA window's (N, 2) float32 signal slice from the .npy memmap.

    Called by the QA dataset at ``__getitem__`` time: parquets store only
    ``(record_id, window_start_ms, window_end_ms, source_hz)`` and defer the
    signal read to here so each shard is kilobytes instead of gigabytes.
    """
    start_sample = int(round(window_start_ms / 1000.0 * source_hz))
    end_sample = int(round(window_end_ms / 1000.0 * source_hz))
    return load_bout_signal(record_id, start_sample, end_sample)
