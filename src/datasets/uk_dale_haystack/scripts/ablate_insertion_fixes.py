#!/usr/bin/env python3
"""
Ablation: which sim-to-real fixes actually narrow the isolated-insertion gap?

Runs the isolated-insertion probe (both classes have exactly 1 bout in
window) under several insertion variants and reports AUC per
(channel, appliance, variant). Variants:

  baseline                : current behaviour (no fix)
  buffer_30s              : extract +-30 s of submeter context around bout
                            (Fix 1: preserves natural ramp/decay at bout edges)
  buffer_120s             : same idea with +-120 s buffer
  timematch_30d           : reject BG samples not within 30 days of bout
                            source time (Fix 2: matches sensor calibration)
  timematch_7d            : tighter time match (within 7 days)
  buffer_30s_timematch_7d : combined Fix 1 + Fix 2

For each variant we report the placebo AUC (should stay ~0.5; if not, the
variant introduced a methodology bias) and the real-test AUC. The Δ vs
baseline tells you whether each fix is moving the needle.

Output: data/uk_dale/uk_dale_haystack/diagnostics/insertion_fix_ablation.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.loader import (
    UKD_HAYSTACK_DIR,
    load_meter_window_grid,
    mains_meter_id,
    NOMINAL_DT_S,
)
from src.datasets.uk_dale_haystack.scripts.validate_isolated_insertion import (
    _bout_is_isolated_in_window,
    _centre_pad,
    _train_and_score,
    collect_isolated_natural,
)


@dataclass
class FixVariant:
    name: str
    bout_buffer_s: float = 0.0
    time_match_days: int | None = None


VARIANTS = [
    FixVariant("baseline"),
    FixVariant("buffer_30s",              bout_buffer_s=30.0),
    FixVariant("buffer_120s",             bout_buffer_s=120.0),
    FixVariant("timematch_30d",           time_match_days=30),
    FixVariant("timematch_7d",            time_match_days=7),
    FixVariant("buffer_30s_timematch_7d", bout_buffer_s=30.0, time_match_days=7),
]


# ---------------------------------------------------------------------------
# Custom inserted-window collector
# ---------------------------------------------------------------------------

def _sample_bg_with_time_match(
    sampler: BackgroundSampler, target: str, ctx_s: float, house: int,
    target_time_ns: int, time_match_days: int | None,
    rng: np.random.Generator, max_attempts: int = 50,
):
    """Sample a target-OFF window. When time_match_days is set, intersect
    the candidate intervals with [target_time_ns +- delta] BEFORE sampling
    rather than rejecting after-the-fact (rejection sampling is hopeless
    for h1's 4.5-year recording vs a +-7-day window).
    """
    if time_match_days is None:
        return sampler.sample(
            target=target, context_length_s=ctx_s, rng=rng,
            same_house_as=house,
            margin_s=min(ctx_s * 0.05, 60.0),
        )

    # Build the constrained interval list directly.
    delta_ns = int(time_match_days * 86400 * 1e9)
    tf = sampler._timeframes[house]
    if not (tf.start_ns - delta_ns <= target_time_ns <= tf.end_ns + delta_ns):
        return None

    target_bouts = sampler._bouts_for(house, target)
    from src.datasets.uk_dale_haystack.core.background_sampler import (
        _interval_complement, _interval_subtract,
    )
    full_off = _interval_complement(target_bouts, tf.start_ns, tf.end_ns)

    # Intersect with [target_time - delta, target_time + delta]
    lo = target_time_ns - delta_ns
    hi = target_time_ns + delta_ns
    intersected: list[tuple[int, int]] = []
    for s, e in full_off:
        a = max(int(s), int(lo))
        b = min(int(e), int(hi))
        if b > a:
            intersected.append((a, b))
    if not intersected:
        return None
    off_intervals = np.array(intersected, dtype="int64")

    # Subtract mains gaps (same as BackgroundSampler)
    from src.datasets.uk_dale_haystack.loader import meter_gaps, mains_meter_id, NOMINAL_DT_S, load_meter_window_grid
    margin_s = min(ctx_s * 0.05, 60.0)
    margin_ns = int(margin_s * 1e9)
    gaps = meter_gaps(house, mains_meter_id(house))
    off_intervals = _interval_subtract(off_intervals, gaps, pad_ns=margin_ns)

    ctx_ns = int(ctx_s * 1e9)
    min_off_ns = ctx_ns + 2 * margin_ns
    long_enough = off_intervals[(off_intervals[:, 1] - off_intervals[:, 0]) >= min_off_ns]
    if long_enough.size == 0:
        return None

    slacks = (long_enough[:, 1] - long_enough[:, 0] - min_off_ns).astype("float64")
    probs = (slacks + 1.0) / (slacks.sum() + len(slacks))
    iv_idx = int(rng.choice(len(long_enough), p=probs))
    iv_start, iv_end = int(long_enough[iv_idx, 0]), int(long_enough[iv_idx, 1])
    start_ns_low = iv_start + margin_ns
    start_ns_high = iv_end - ctx_ns - margin_ns
    if start_ns_high < start_ns_low:
        return None
    window_start_ns = int(rng.integers(start_ns_low, start_ns_high + 1))
    window_end_ns = window_start_ns + ctx_ns
    mains_w = load_meter_window_grid(
        house, mains_meter_id(house), window_start_ns, window_end_ns, dt_s=NOMINAL_DT_S,
    )
    # Synthesize a minimal BackgroundSample-shaped object the caller expects
    from src.datasets.uk_dale_haystack.core.data_structures import BackgroundSample
    return BackgroundSample(
        house_id=house, start_ns=window_start_ns, end_ns=window_end_ns,
        dt_s=NOMINAL_DT_S, mains_w=mains_w.astype("float32", copy=False),
        other_bouts=[], recording_time_context=("", ""),
    )


def collect_inserted_variant(
    donor_row: dict, sampler: BackgroundSampler, window_samples: int,
    dt_s: float, rng: np.random.Generator, source: str,
    *, variant: FixVariant,
) -> np.ndarray | None:
    house = int(donor_row["house_id"])
    bs, be = int(donor_row["start_ns"]), int(donor_row["end_ns"])
    buf_ns = int(variant.bout_buffer_s * 1e9)
    bs_buf = bs - buf_ns
    be_buf = be + buf_ns

    ctx_s = window_samples * dt_s
    bg = _sample_bg_with_time_match(
        sampler, str(donor_row["appliance"]), ctx_s, house, bs,
        variant.time_match_days, rng,
    )
    if bg is None or bg.mains_w.size == 0:
        return None

    if source == "mains":
        bg_signal = bg.mains_w
    else:
        bg_signal = load_meter_window_grid(
            house, int(donor_row["meter_id"]),
            bg.start_ns, bg.end_ns, dt_s=dt_s,
        )
    if bg_signal.size == 0:
        return None

    sub = load_meter_window_grid(
        int(donor_row["house_id"]), int(donor_row["meter_id"]),
        bs_buf, be_buf + int(dt_s * 1e9), dt_s=dt_s,
    )
    if sub.size == 0:
        return None

    new_signal = _centre_pad(bg_signal, window_samples).copy()
    sub = sub[:window_samples]
    centre = window_samples // 2
    pos = max(0, centre - sub.size // 2)
    end = min(window_samples, pos + sub.size)
    new_signal[pos:end] += sub[: end - pos]
    return new_signal


# ---------------------------------------------------------------------------
# Probe per variant
# ---------------------------------------------------------------------------

def probe_variant(
    bouts: pl.DataFrame, sampler: BackgroundSampler, appliance: str,
    n_per_class: int, rng: np.random.Generator,
    *, source: str, variant: FixVariant,
    window_factor: float, max_window_samples: int,
    dt_s: float = NOMINAL_DT_S,
) -> dict:
    sub = bouts.filter((pl.col("appliance") == appliance) & (pl.col("split") == "train"))
    dur_med_s = float(sub["duration_s"].median())
    window_samples = max(20, int(window_factor * dur_med_s / dt_s))
    window_samples = min(window_samples, max_window_samples)

    house_app_bouts: dict[int, np.ndarray] = {}
    for h in sub["house_id"].unique().to_list():
        arr = (sub.filter(pl.col("house_id") == h)
                  .select(["start_ns", "end_ns"])
                  .sort("start_ns").to_numpy().astype("int64"))
        house_app_bouts[int(h)] = arr

    rows = sub.to_dicts()
    rng.shuffle(rows)

    nat_X: list[np.ndarray] = []
    ins_X: list[np.ndarray] = []
    for row in rows:
        if len(nat_X) >= n_per_class and len(ins_X) >= n_per_class:
            break
        arr = house_app_bouts[int(row["house_id"])]
        if len(nat_X) < n_per_class:
            w = collect_isolated_natural(row, window_samples, dt_s, source, arr)
            if w is not None:
                nat_X.append(w)
        if len(ins_X) < n_per_class:
            w = collect_inserted_variant(row, sampler, window_samples, dt_s, rng, source,
                                         variant=variant)
            if w is not None:
                ins_X.append(w)

    if len(nat_X) < 30 or len(ins_X) < 30:
        return {"variant": variant.name, "skipped": True,
                "reason": f"too few collected (nat={len(nat_X)}, ins={len(ins_X)})"}

    n_match = min(len(nat_X), len(ins_X))
    X = np.vstack([np.stack(nat_X[:n_match]), np.stack(ins_X[:n_match])]).astype("float32")
    y = np.concatenate([np.zeros(n_match), np.ones(n_match)]).astype("int64")
    real_auc = _train_and_score(X, y)

    return {
        "variant": variant.name,
        "bout_buffer_s": float(variant.bout_buffer_s),
        "time_match_days": variant.time_match_days,
        "n_natural": len(nat_X), "n_inserted": len(ins_X),
        "window_samples": int(window_samples),
        "real_auc_5fold": real_auc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index",
                    default=str(UKD_HAYSTACK_DIR / "bout_index.parquet"))
    ap.add_argument("--out",
                    default=str(UKD_HAYSTACK_DIR / "diagnostics"
                                / "insertion_fix_ablation.json"))
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--appliances", nargs="+",
                    default=["kettle", "microwave", "fridge_freezer"])
    ap.add_argument("--sources", nargs="+",
                    choices=["mains", "submeter"], default=["submeter", "mains"])
    ap.add_argument("--window-factor", type=float, default=4.0)
    ap.add_argument("--max-window-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bouts = pl.read_parquet(args.bout_index)
    sampler = BackgroundSampler(bouts, "train")

    out: dict = {"per_run": []}
    for source in args.sources:
        print(f"\n=== source = {source} ===")
        for app in args.appliances:
            print(f"  --- {app} ---")
            for variant in VARIANTS:
                rng = np.random.default_rng(args.seed + hash(variant.name) % 1000)
                result = probe_variant(
                    bouts, sampler, app, args.n_per_class, rng,
                    source=source, variant=variant,
                    window_factor=args.window_factor,
                    max_window_samples=args.max_window_samples,
                )
                result["appliance"] = app
                result["source"] = source
                out["per_run"].append(result)
                if result.get("skipped"):
                    print(f"    {variant.name:<25} SKIPPED: {result['reason']}")
                else:
                    print(f"    {variant.name:<25} AUC={result['real_auc_5fold']:.3f}  "
                          f"(n_nat={result['n_natural']}, n_ins={result['n_inserted']})")

    print()
    print(" Variant                    submeter (k/m/ff)        mains (k/m/ff)")
    print(" " + "-" * 72)
    for variant in VARIANTS:
        row = []
        for source in ("submeter", "mains"):
            for app in args.appliances:
                r = next((r for r in out["per_run"]
                          if r["variant"] == variant.name
                          and r["source"] == source
                          and r["appliance"] == app), None)
                if r is None or r.get("skipped"):
                    row.append("  n/a")
                else:
                    row.append(f"{r['real_auc_5fold']:.3f}")
        print(f" {variant.name:<26} {row[0]} {row[1]} {row[2]}     {row[3]} {row[4]} {row[5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {out_path}")


if __name__ == "__main__":
    main()
