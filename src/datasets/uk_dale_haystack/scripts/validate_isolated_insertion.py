#!/usr/bin/env python3
"""
Controlled experiment for additive-insertion fidelity.

Both classes contain EXACTLY one occurrence of the target appliance per
window, so the temporal-clustering shortcut (the dominant signal in the
default submeter probe) is removed. Whatever AUC remains is purely the
"is the bout naturally embedded vs additively inserted into a flat
background" signal -- the cleanest possible test of insertion fidelity.

Per appliance + per channel we run:

  REAL TEST   class 0: real-channel window centred on an ISOLATED real bout
                       (the appliance ran exactly once in the window)
              class 1: target-OFF same-house BG window (which on the submeter
                       channel is essentially flat 0 W) + a DIFFERENT real
                       bout of the same appliance additively inserted
                       at the centre

  PLACEBO     class 0: isolated natural window from one bout
              class 1: isolated natural window from a different bout
              (arbitrary labels; expected AUC = 0.5)

If the REAL TEST AUC drops to placebo level when isolation is enforced on
both sides, the insertion is statistically indistinguishable from a real
isolated bout. If it stays high, the additive sum is leaving a detectable
fingerprint that goes beyond the BG-distribution / clustering gaps.

Output: data/uk_dale/uk_dale_haystack/diagnostics/isolated_insertion_validation.json
        + console table + per-mode gate evaluation
"""
from __future__ import annotations

import argparse
import json
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


PLACEBO_TOLERANCE = 0.10
INSERTION_BLOCKING = 0.80


# ---------------------------------------------------------------------------
# Window collectors (specialised for the isolated-insertion experiment)
# ---------------------------------------------------------------------------

def _centre_pad(arr: np.ndarray, target_n: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.zeros(target_n, dtype="float32")
    if n == 0:
        return out
    if n >= target_n:
        lo = (n - target_n) // 2
        out[:] = arr[lo:lo + target_n]
        return out
    pad = (target_n - n) // 2
    out[pad:pad + n] = arr
    return out


def _bout_is_isolated_in_window(
    house_app_bouts: np.ndarray,
    target_start_ns: int,
    win_start: int,
    win_end: int,
) -> bool:
    if house_app_bouts.size == 0:
        return False
    starts = house_app_bouts[:, 0]
    ends = house_app_bouts[:, 1]
    overlap = (starts < win_end) & (ends > win_start)
    not_self = starts != target_start_ns
    return not (overlap & not_self).any()


def collect_isolated_natural(
    bout_row: dict, window_samples: int, dt_s: float, source: str,
    house_app_bouts: np.ndarray,
) -> np.ndarray | None:
    """Real-channel window centred on a bout that is the ONLY occurrence
    of its appliance in the window."""
    house = int(bout_row["house_id"])
    bs, be = int(bout_row["start_ns"]), int(bout_row["end_ns"])
    centre = (bs + be) // 2
    half_ns = (window_samples * int(dt_s * 1e9)) // 2
    win_start = centre - half_ns
    win_end = win_start + window_samples * int(dt_s * 1e9)
    if not _bout_is_isolated_in_window(house_app_bouts, bs, win_start, win_end):
        return None
    if source == "mains":
        meter = mains_meter_id(house)
    else:
        meter = int(bout_row["meter_id"])
    sig = load_meter_window_grid(house, meter, win_start, win_end, dt_s=dt_s)
    if sig.size == 0:
        return None
    return _centre_pad(sig, window_samples)


def collect_isolated_inserted(
    donor_row: dict, sampler: BackgroundSampler, window_samples: int,
    dt_s: float, rng: np.random.Generator, source: str,
) -> np.ndarray | None:
    """Target-OFF same-house BG window + the donor bout additively inserted
    at the centre. The BG is by construction a window where the appliance
    is OFF for the entire span, so the submeter channel is essentially flat
    0 W and the inserted result has exactly 1 occurrence of the appliance."""
    house = int(donor_row["house_id"])
    bs, be = int(donor_row["start_ns"]), int(donor_row["end_ns"])
    ctx_s = window_samples * dt_s
    bg = sampler.sample(
        target=str(donor_row["appliance"]),
        context_length_s=ctx_s,
        rng=rng,
        same_house_as=house,
        margin_s=min(ctx_s * 0.05, 60.0),
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
        bs, be + int(dt_s * 1e9), dt_s=dt_s,
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
# Probe
# ---------------------------------------------------------------------------

def _train_and_score(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    try:
        from xgboost import XGBClassifier
        clf_factory = lambda: XGBClassifier(
            max_depth=4, n_estimators=200, learning_rate=0.1,
            objective="binary:logistic", eval_metric="auc",
            verbosity=0, n_jobs=1,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        clf_factory = lambda: GradientBoostingClassifier(
            max_depth=4, n_estimators=200, learning_rate=0.1,
        )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))
    return float(np.mean(aucs))


def probe_appliance_isolated(
    bouts: pl.DataFrame, sampler: BackgroundSampler, appliance: str,
    n_per_class: int, rng: np.random.Generator,
    *,
    source: str,
    window_factor: float,
    max_window_samples: int,
    dt_s: float = NOMINAL_DT_S,
) -> dict:
    sub = bouts.filter((pl.col("appliance") == appliance) & (pl.col("split") == "train"))
    if sub.height < 50:
        return {"appliance": appliance, "source": source, "skipped": True,
                "reason": f"only {sub.height} bouts"}

    dur_med_s = float(sub["duration_s"].median())
    window_samples = max(20, int(window_factor * dur_med_s / dt_s))
    window_samples = min(window_samples, max_window_samples)

    # Per-house bout arrays for the isolation check
    house_app_bouts: dict[int, np.ndarray] = {}
    for h in sub["house_id"].unique().to_list():
        arr = (sub.filter(pl.col("house_id") == h)
                  .select(["start_ns", "end_ns"])
                  .sort("start_ns").to_numpy().astype("int64"))
        house_app_bouts[int(h)] = arr

    rows = sub.to_dicts()
    rng.shuffle(rows)

    # ----- REAL TEST: isolated natural vs flat-BG + inserted bout -----
    real_nat: list[np.ndarray] = []
    real_ins: list[np.ndarray] = []
    n_rejected_clustered = 0
    for row in rows:
        if len(real_nat) >= n_per_class and len(real_ins) >= n_per_class:
            break
        arr = house_app_bouts[int(row["house_id"])]
        if len(real_nat) < n_per_class:
            w = collect_isolated_natural(row, window_samples, dt_s, source, arr)
            if w is None:
                n_rejected_clustered += 1
            else:
                real_nat.append(w)
        if len(real_ins) < n_per_class:
            w = collect_isolated_inserted(row, sampler, window_samples, dt_s, rng, source)
            if w is not None:
                real_ins.append(w)

    real_auc = None
    if len(real_nat) >= 30 and len(real_ins) >= 30:
        n_match = min(len(real_nat), len(real_ins))
        X = np.vstack([np.stack(real_nat[:n_match]), np.stack(real_ins[:n_match])]).astype("float32")
        y = np.concatenate([np.zeros(n_match), np.ones(n_match)]).astype("int64")
        real_auc = _train_and_score(X, y)

    # ----- PLACEBO: two disjoint draws from the isolated-natural pool -----
    rng_placebo = np.random.default_rng(rng.integers(0, 2**31))
    rows_p = list(rows); rng_placebo.shuffle(rows_p)
    pl_a: list[np.ndarray] = []
    pl_b: list[np.ndarray] = []
    for row in rows_p:
        if len(pl_a) >= n_per_class and len(pl_b) >= n_per_class:
            break
        arr = house_app_bouts[int(row["house_id"])]
        w = collect_isolated_natural(row, window_samples, dt_s, source, arr)
        if w is None:
            continue
        if len(pl_a) <= len(pl_b):
            pl_a.append(w)
        else:
            pl_b.append(w)

    placebo_auc = None
    if len(pl_a) >= 30 and len(pl_b) >= 30:
        n_match = min(len(pl_a), len(pl_b))
        X = np.vstack([np.stack(pl_a[:n_match]), np.stack(pl_b[:n_match])]).astype("float32")
        y = np.concatenate([np.zeros(n_match), np.ones(n_match)]).astype("int64")
        placebo_auc = _train_and_score(X, y)

    return {
        "appliance": appliance,
        "source": source,
        "window_samples": int(window_samples),
        "window_seconds": float(window_samples * dt_s),
        "median_bout_duration_s": dur_med_s,
        "n_rejected_clustered": int(n_rejected_clustered),
        "n_real_natural": len(real_nat),
        "n_real_inserted": len(real_ins),
        "n_placebo_a": len(pl_a),
        "n_placebo_b": len(pl_b),
        "real_auc_5fold": real_auc,
        "placebo_auc_5fold": placebo_auc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index",
                    default=str(UKD_HAYSTACK_DIR / "bout_index.parquet"))
    ap.add_argument("--out",
                    default=str(UKD_HAYSTACK_DIR / "diagnostics"
                                / "isolated_insertion_validation.json"))
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--appliances", nargs="+", default=None,
                    help="Default: any appliance with >=200 train bouts.")
    ap.add_argument("--sources", nargs="+",
                    choices=["mains", "submeter"], default=["submeter", "mains"],
                    help="submeter is the primary channel for this experiment.")
    ap.add_argument("--window-factor", type=float, default=4.0)
    ap.add_argument("--max-window-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bouts = pl.read_parquet(args.bout_index)
    sampler = BackgroundSampler(bouts, "train")

    if args.appliances:
        targets = list(args.appliances)
    else:
        targets = []
        for app in bouts.filter(pl.col("split") == "train")["appliance"].unique().to_list():
            n = bouts.filter((pl.col("appliance") == app) & (pl.col("split") == "train")).height
            if n >= 200:
                targets.append(app)
    targets.sort()

    out: dict = {"per_run": []}
    print()
    for source in args.sources:
        print(f"=== source = {source} ===")
        for app in targets:
            rng = np.random.default_rng(args.seed)
            result = probe_appliance_isolated(
                bouts, sampler, app, args.n_per_class, rng,
                source=source,
                window_factor=args.window_factor,
                max_window_samples=args.max_window_samples,
            )
            out["per_run"].append(result)
            if result.get("skipped"):
                print(f"  {app:<18} SKIPPED: {result['reason']}")
                continue
            print(f"  {app:<18}  placebo={_fmt(result['placebo_auc_5fold'])}  "
                  f"real={_fmt(result['real_auc_5fold'])}  "
                  f"(window={result['window_samples']}samp/{result['window_seconds']:.0f}s, "
                  f"isolation_rejected={result['n_rejected_clustered']})")
        print()

    print(" Channel    Appliance         Placebo  Real test   Δ")
    print(" " + "-" * 60)
    for source in args.sources:
        for r in out["per_run"]:
            if r["source"] != source or r.get("skipped"):
                continue
            p = r["placebo_auc_5fold"]; t = r["real_auc_5fold"]
            if p is None or t is None:
                continue
            print(f" {source:<10} {r['appliance']:<18} {p:>5.3f}    {t:>5.3f}    {t - p:+.3f}")

    print()
    print(" Gate evaluation:")
    for source in args.sources:
        rows = [r for r in out["per_run"]
                if r["source"] == source and not r.get("skipped")
                and r["placebo_auc_5fold"] is not None
                and r["real_auc_5fold"] is not None]
        if not rows:
            continue
        max_p_dev = max(abs(r["placebo_auc_5fold"] - 0.5) for r in rows)
        max_real = max(r["real_auc_5fold"] for r in rows)
        print(f"  [{source}]  placebo max |AUC-0.5| = {max_p_dev:.3f} "
              f"({'PASS' if max_p_dev <= PLACEBO_TOLERANCE else 'FAIL'} <= {PLACEBO_TOLERANCE})")
        print(f"  [{source}]  real-test max AUC      = {max_real:.3f} "
              f"({'PASS' if max_real <= INSERTION_BLOCKING else 'FAIL'} <= {INSERTION_BLOCKING})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {out_path}")


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  n/a"


if __name__ == "__main__":
    main()
