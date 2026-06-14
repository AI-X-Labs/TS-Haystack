"""
Additive needle insertion. Composes BackgroundSample + N NeedleSamples into a
GeneratedSample.

`mains_with_target = background_mains + needle_submeter` -- the submeter trace
is the appliance's actual physical contribution to mains, so the sum is a
clinically-plausible mains signal. No splice boundary, no style transfer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from src.datasets.uk_dale_haystack.core.data_structures import (
    BackgroundSample,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
)


def sample_positions(
    ctx_samples: int,
    needle_lens_samples: Sequence[int],
    margin_samples: int,
    min_gap_samples: int,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> list[int] | None:
    """Reject-resample non-overlapping start positions for k needles.

    Each needle stays >= margin_samples from window edges; needles are
    separated by >= min_gap_samples. Returns None after max_attempts failures.
    """
    k = len(needle_lens_samples)
    if k == 0:
        return []

    for _ in range(max_attempts):
        positions: list[int] = []
        # Sort needles longest-first for tighter packing (helps when total
        # required space is close to ctx_samples).
        order = sorted(range(k), key=lambda i: -needle_lens_samples[i])
        ok = True
        # Track placed [start, end] intervals
        placed: list[tuple[int, int]] = []
        for idx_in_sorted in order:
            n_len = int(needle_lens_samples[idx_in_sorted])
            placed_pos = None
            for _ in range(max_attempts):
                lo = margin_samples
                hi = ctx_samples - margin_samples - n_len
                if hi <= lo:
                    placed_pos = None
                    break
                pos = int(rng.integers(lo, hi + 1))
                # Check non-overlap with placed
                bad = False
                for ps, pe in placed:
                    if not (pos + n_len + min_gap_samples <= ps or pe + min_gap_samples <= pos):
                        bad = True
                        break
                if not bad:
                    placed_pos = pos
                    break
            if placed_pos is None:
                ok = False
                break
            placed.append((placed_pos, placed_pos + n_len))
            positions.append((idx_in_sorted, placed_pos))
        if ok:
            positions.sort(key=lambda t: t[0])
            return [p for _, p in positions]
    return None


def _fmt_local_time(start_ns: int, offset_samples: int, dt_s: float) -> str:
    t_ns = int(start_ns + offset_samples * dt_s * 1e9)
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).strftime("%H:%M:%S")


def insert_needles(
    bg: BackgroundSample,
    needles: Sequence[NeedleSample],
    positions_samples: Sequence[int],
    *,
    task_type: str,
    question: str,
    answer: str,
    answer_type: str,
    difficulty_config: dict | None = None,
    metadata: dict | None = None,
) -> GeneratedSample:
    """Sum needles into the background mains. No edge blending."""
    mains = bg.mains_w.copy()
    inserted: list[InsertedNeedle] = []
    ctx_samples = mains.shape[0]
    for n, pos in zip(needles, positions_samples):
        n_samples = int(n.submeter_w.shape[0])
        end = int(pos) + n_samples
        if pos < 0 or end > ctx_samples:
            raise ValueError(
                f"needle out of window: pos={pos} end={end} ctx={ctx_samples} "
                f"appliance={n.appliance}"
            )
        mains[pos:end] += n.submeter_w
        inserted.append(InsertedNeedle(
            appliance=n.appliance,
            insert_position_samples=int(pos),
            insert_duration_samples=n_samples,
            duration_s=float(n.duration_s),
            peak_w=float(n.peak_w),
            source_house_id=int(n.source_house_id),
            source_meter_id=int(n.source_meter_id),
            source_start_ns=int(n.start_ns),
            source_end_ns=int(n.end_ns),
            timestamp_start=_fmt_local_time(bg.start_ns, int(pos), bg.dt_s),
            timestamp_end=_fmt_local_time(bg.start_ns, end, bg.dt_s),
            is_anomalous=bool(n.is_anomalous),
            anomaly_class=n.anomaly_class,
            anomaly_params=dict(n.anomaly_params),
        ))

    return GeneratedSample(
        task_type=task_type,
        question=question,
        answer=answer,
        answer_type=answer_type,
        mains_w=mains.astype("float32", copy=False),
        context_length_samples=ctx_samples,
        context_length_s=float(ctx_samples * bg.dt_s),
        dt_s=float(bg.dt_s),
        background_house_id=int(bg.house_id),
        background_start_ns=int(bg.start_ns),
        background_end_ns=int(bg.end_ns),
        needles=inserted,
        other_bouts_metadata=list(bg.other_bouts),
        difficulty_config=dict(difficulty_config or {}),
        is_valid=True,
        validation_notes=None,
        metadata=dict(metadata or {}),
    )


def trim_needle_to_fit(
    needle: NeedleSample, max_samples: int,
) -> NeedleSample:
    """Defensive trimmer: clip the needle's submeter trace to ``max_samples``
    samples (used when the bout barely overflows the context window).
    """
    if needle.submeter_w.shape[0] <= max_samples:
        return needle
    new_w = needle.submeter_w[:max_samples].copy()
    return NeedleSample(
        source_house_id=needle.source_house_id,
        source_meter_id=needle.source_meter_id,
        appliance=needle.appliance,
        start_ns=needle.start_ns,
        end_ns=int(needle.start_ns + max_samples * needle.dt_s * 1e9),
        duration_s=float(max_samples * needle.dt_s),
        submeter_w=new_w,
        dt_s=needle.dt_s,
        peak_w=float(new_w.max()),
        mean_w=float(new_w.mean()),
        is_anomalous=needle.is_anomalous,
        anomaly_class=needle.anomaly_class,
        anomaly_params=dict(needle.anomaly_params),
    )
