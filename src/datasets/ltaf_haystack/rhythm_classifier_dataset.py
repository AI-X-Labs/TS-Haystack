# SPDX-License-Identifier: CC-BY-NC-4.0
"""LTAF rhythm-classifier dataset.

Yields (signal, label) pairs sampled from rhythm bouts in
``data/ltafdb/ltaf_haystack/timelines/<rid>.parquet``. Signals are
loaded on demand via memmap (no on-disk signal cache) and per-channel
z-scored at ``__getitem__`` time.

The natural-window sampler draws one window per bout per epoch with a
random offset such that the window overlaps the bout by at least
``min(window_seconds / 2, bout_length)`` seconds. Re-rolled each epoch
via ``set_epoch``.

Used by both the rhythm trainer and the TTA evaluator.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm.auto import tqdm

from src.datasets.ltaf_haystack.loader import (
    LTAF_DATA_DIR,
    SOURCE_HZ,
    load_bout_signal,
    load_record_signals_mmap,
)


HAYSTACK_DIR = LTAF_DATA_DIR / "ltaf_haystack"
SPLIT_MANIFEST = HAYSTACK_DIR / "split_manifest.json"
TIMELINES_DIR = HAYSTACK_DIR / "timelines"

# Default 9-class LTAF rhythm taxonomy. Pass ``classes=...`` to subset.
RHYTHM_CLASS_NAMES = ["NSR", "AFIB", "SBR", "AB", "SVTA", "B", "VT", "T", "IVR"]


def _zscore(chunk: np.ndarray) -> np.ndarray:
    """Per-channel z-score on a (C, L) float32 array."""
    mean = chunk.mean(axis=-1, keepdims=True)
    std = chunk.std(axis=-1, keepdims=True)
    return ((chunk - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def load_ecg_window(record_id: str, start_sample: int, end_sample: int) -> np.ndarray:
    """Memmap-slice a (2, L) float32 window at 128 Hz and z-score per channel."""
    chunk = load_bout_signal(record_id, start_sample, end_sample)  # (L, 2)
    chunk = np.ascontiguousarray(chunk.T)
    return _zscore(chunk)


def _record_total_samples(record_id: str) -> int:
    return int(load_record_signals_mmap(record_id).shape[0])


class EcgRhythmsClassifierDataset(Dataset):
    """Windows drawn from rhythm bouts in the LTAF timelines.

    Args:
        record_ids: list of LTAF record ids to sample from.
        window_seconds: length of each output window (seconds).
        encoder_window: target sample-length for the model; shorter windows
            are zero-padded so mixed-window batches are uniform.
        base_seed: seed for window-offset sampling (re-rolled per epoch).
        classes: ordered list of rhythm labels to keep. Anything outside
            the list is ignored (default: all 9 LTAF rhythms).
    """

    def __init__(
        self,
        record_ids: List[str],
        window_seconds: float,
        encoder_window: int,
        base_seed: int = 42,
        classes: List[str] | None = None,
    ):
        self.class_names = list(classes) if classes is not None else list(RHYTHM_CLASS_NAMES)
        self.label_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.window_seconds = float(window_seconds)
        self.window_samples = int(round(window_seconds * SOURCE_HZ))
        self.encoder_window = int(encoder_window)
        self.window_native = self.window_samples
        self.min_overlap_native = max(1, self.window_samples // 2)
        self.base_seed = int(base_seed)

        # Per-record (record_id, list[(start_sample, end_sample, rhythm)], total_samples).
        self._record_bouts: List[Tuple[str, List[Tuple[int, int, str]], int]] = []
        self.entries: List[Tuple[str, int, int, int]] = []

        for rid in tqdm(record_ids, desc="indexing rhythms"):
            timeline_path = TIMELINES_DIR / f"{rid}.parquet"
            if not timeline_path.exists():
                continue
            try:
                df = pd.read_parquet(timeline_path)
                total_samples = _record_total_samples(rid)
            except Exception as e:
                print(f"  skip {rid}: {e}")
                continue
            bouts: List[Tuple[int, int, str]] = []
            for row in df.itertuples(index=False):
                lbl = row.activity
                if lbl not in self.label_to_idx:
                    continue
                bouts.append((int(row.start_sample), int(row.end_sample), lbl))
            if not bouts:
                continue
            self._record_bouts.append((rid, bouts, total_samples))

        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.base_seed * 1_000_003 + epoch)
        entries: List[Tuple[str, int, int, int]] = []

        for rid, bouts, total_samples in self._record_bouts:
            for start, end, lbl in bouts:
                bout_len = max(1, end - start)
                target = min(self.min_overlap_native, bout_len)
                # Valid window start ws satisfies:
                #   ws + W >= start + target  ->  ws >= start - W + target
                #   ws     <= end - target
                low = start - self.window_native + target
                high = end - target
                low = max(0, low)
                high = min(total_samples - self.window_native, high)
                if high < low:
                    continue  # bout shorter than window near a recording edge
                ws_native = rng.randint(low, high)
                we_native = ws_native + self.window_native
                entries.append((rid, ws_native, we_native, self.label_to_idx[lbl]))

        self.entries = entries

    @property
    def labels(self) -> np.ndarray:
        return np.array([e[3] for e in self.entries], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        rid, ws, we, label = self.entries[idx]
        signal = load_ecg_window(rid, ws, we)  # (2, ~window_samples)
        if signal.shape[1] > self.window_samples:
            signal = signal[:, : self.window_samples]
        if signal.shape[1] < self.encoder_window:
            pad = self.encoder_window - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)))
        return torch.from_numpy(signal).float(), int(label)


def set_epoch_recursive(ds, epoch: int) -> None:
    """Call ``set_epoch`` on a Dataset or each member of a ConcatDataset."""
    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            if hasattr(sub, "set_epoch"):
                sub.set_epoch(epoch)
    elif hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)


def collect_labels(ds) -> np.ndarray:
    """Collect labels across a Dataset or ConcatDataset for class-weighting."""
    if isinstance(ds, ConcatDataset):
        return np.concatenate([sub.labels for sub in ds.datasets])
    return ds.labels


def load_split_manifest() -> dict:
    with open(SPLIT_MANIFEST) as f:
        return json.load(f)
