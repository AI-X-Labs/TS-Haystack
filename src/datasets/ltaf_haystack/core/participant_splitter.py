# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Deterministic record-level train/val/test split for LTAF-Haystack.

Also computes per-record paced-beat ratios so paced recordings can be
flagged and excluded from the insertion donor pool. Paced morphology is
categorically different from intrinsic conduction, and splicing a paced
needle into a non-paced background is clinically nonsensical.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import polars as pl

from src.datasets.ltaf_haystack.core.ltaf_timeline_builder import (
    LTAF_HAYSTACK_DIR,
    get_ltaf_beat_timelines_dir,
    get_ltaf_split_manifest_path,
)


DEFAULT_PACED_THRESHOLD = 0.05  # >5% Q beats → paced


def _beat_counts_for_record(beat_timeline_path: Path) -> Counter:
    if not beat_timeline_path.exists():
        return Counter()
    df = pl.read_parquet(beat_timeline_path)
    if len(df) == 0:
        return Counter()
    symbols = df["symbol"].to_list()
    return Counter(symbols)


def compute_paced_ratios(
    participant_ids: Iterable[str],
    beat_timelines_dir: Path | None = None,
) -> Dict[str, float]:
    """Return ``{participant_id: paced_ratio}`` where paced_ratio = n_Q / n_total.

    Missing beat timelines are treated as ratio 0 (they will be flagged as
    such in downstream reports but not as paced).
    """
    beat_dir = Path(beat_timelines_dir) if beat_timelines_dir else get_ltaf_beat_timelines_dir()
    ratios: Dict[str, float] = {}
    for pid in participant_ids:
        counts = _beat_counts_for_record(beat_dir / f"{pid}.parquet")
        total = sum(counts.values())
        ratios[pid] = float(counts.get("Q", 0) / total) if total else 0.0
    return ratios


def flag_paced_records(
    paced_ratios: Dict[str, float],
    threshold: float = DEFAULT_PACED_THRESHOLD,
) -> List[str]:
    return sorted(pid for pid, ratio in paced_ratios.items() if ratio > threshold)


def split_participants(
    participant_ids: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, object]:
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    unique_ids = sorted(set(participant_ids))
    n = len(unique_ids)

    rng = np.random.default_rng(int(seed))
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    if n == 0:
        return {
            "seed": int(seed),
            "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
            "n_participants": 0,
            "train": [],
            "validation": [],
            "test": [],
        }

    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    if n >= 3:
        if n_val == 0:
            if n_train > 1:
                n_train -= 1
                n_val = 1
            elif n_test > 1:
                n_test -= 1
                n_val = 1
        if n_test == 0:
            if n_train > 1:
                n_train -= 1
                n_test = 1
            elif n_val > 1:
                n_val -= 1
                n_test = 1

    if n_train == 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    n_test = n - n_train - n_val

    train = sorted(shuffled[:n_train])
    validation = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])

    return {
        "seed": int(seed),
        "ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        "n_participants": n,
        "train": train,
        "validation": validation,
        "test": test,
    }


def build_split_manifest(
    participant_ids: Iterable[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    paced_threshold: float = DEFAULT_PACED_THRESHOLD,
    beat_timelines_dir: Path | None = None,
) -> Dict[str, object]:
    """Compute split + paced-ratio fields in one shot."""
    manifest = split_participants(
        participant_ids=participant_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    paced_ratios = compute_paced_ratios(
        participant_ids=list(manifest["train"])
        + list(manifest["validation"])
        + list(manifest["test"]),
        beat_timelines_dir=beat_timelines_dir,
    )
    manifest["paced_threshold"] = float(paced_threshold)
    manifest["paced_ratio_per_record"] = paced_ratios
    manifest["paced_records"] = flag_paced_records(paced_ratios, threshold=paced_threshold)
    return manifest


def save_split_manifest(manifest: Dict[str, object], path: Path | None = None) -> Path:
    target = Path(path) if path else get_ltaf_split_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return target


def load_split_manifest(path: Path | None = None) -> Dict[str, object]:
    target = Path(path) if path else get_ltaf_split_manifest_path()
    if not target.exists():
        raise FileNotFoundError(f"LTAF split manifest not found at {target}")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_paced_records_from_manifest(manifest: Dict[str, object]) -> List[str]:
    return sorted(str(r) for r in manifest.get("paced_records", []))


__all__ = [
    "DEFAULT_PACED_THRESHOLD",
    "LTAF_HAYSTACK_DIR",
    "compute_paced_ratios",
    "flag_paced_records",
    "split_participants",
    "build_split_manifest",
    "save_split_manifest",
    "load_split_manifest",
    "get_paced_records_from_manifest",
]
