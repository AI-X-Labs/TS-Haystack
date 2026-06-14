#!/usr/bin/env python3
"""
Phase 9.3: insertion shortcut probe.

Probes two channels independently:

  source='mains'   -- the AGGREGATE mains signal the downstream model sees
  source='submeter'-- the appliance's OWN submeter trace around the bout

Both compare:
  - natural window: real recording slice centred on a real bout
  - inserted window: target-OFF window of the same channel + the bout
    additively inserted at the centre

The classifier is fed the **raw time series** (one float32 watt-value per
sample) so XGBoost has to find sample-position thresholds, the same level
of evidence the downstream model has access to.

Submeter probe rationale: if natural submeter windows show bursts of repeated
same-appliance activity (kettle-tea-kettle is common in real life) while
inserted submeter windows are guaranteed clean (BG sampler enforces
target-OFF), the classifier picks up the temporal-clustering shortcut even
though the mains-only probe might miss it. Useful diagnostic for the
"target-OFF is anti-correlated with surrounding appliance activity" failure
mode.

Plan gate:
  AUC <= 0.65 acceptable  (some separability is inevitable -- natural windows
                           carry causally-correct surrounding activity that
                           inserted windows cannot)
  AUC >  0.80 blocking    (insertion strategy must be tuned and re-probed)
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


# ---------------------------------------------------------------------------
# Raw time-series collectors
# ---------------------------------------------------------------------------

def _centre_pad(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Crop or zero-pad ``arr`` to length ``target_n``, keeping the centre."""
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


def _bout_window_bounds(bs: int, be: int, window_samples: int, dt_s: float) -> tuple[int, int]:
    centre = (bs + be) // 2
    half_ns = (window_samples * int(dt_s * 1e9)) // 2
    return centre - half_ns, centre - half_ns + window_samples * int(dt_s * 1e9)


def _bout_is_isolated_in_window(
    house_appliance_bouts: np.ndarray,    # (n, 2) [start, end] of the appliance for the house
    target_start_ns: int,
    window_start_ns: int,
    window_end_ns: int,
) -> bool:
    """Return True iff the bout starting at ``target_start_ns`` is the ONLY
    bout of its appliance overlapping [window_start_ns, window_end_ns).
    """
    if house_appliance_bouts.size == 0:
        return False
    starts = house_appliance_bouts[:, 0]
    ends = house_appliance_bouts[:, 1]
    overlap_mask = (starts < window_end_ns) & (ends > window_start_ns)
    not_self_mask = starts != target_start_ns
    return not (overlap_mask & not_self_mask).any()


def collect_natural_window(
    bout_row: dict, window_samples: int, dt_s: float, source: str,
    *,
    natural_isolated_only: bool = False,
    house_appliance_bouts: np.ndarray | None = None,
) -> np.ndarray | None:
    """Read a fixed-length window centred on the bout from REAL mains/submeter.

    ``natural_isolated_only``: when True, return None unless the centered bout
    is the only bout of its appliance overlapping the window (closes the
    temporal-clustering shortcut at the submeter level).
    """
    house = int(bout_row["house_id"])
    bs, be = int(bout_row["start_ns"]), int(bout_row["end_ns"])
    win_start, win_end = _bout_window_bounds(bs, be, window_samples, dt_s)
    if natural_isolated_only:
        if house_appliance_bouts is None:
            raise ValueError("natural_isolated_only requires house_appliance_bouts")
        if not _bout_is_isolated_in_window(house_appliance_bouts, bs, win_start, win_end):
            return None
    if source == "mains":
        meter = mains_meter_id(house)
    elif source == "submeter":
        meter = int(bout_row["meter_id"])
    else:
        raise ValueError(f"unknown source: {source}")
    sig = load_meter_window_grid(house, meter, win_start, win_end, dt_s=dt_s)
    if sig.size == 0:
        return None
    return _centre_pad(sig, window_samples)


def collect_inserted_window(
    bout_row: dict, sampler: BackgroundSampler, window_samples: int,
    dt_s: float, rng: np.random.Generator, source: str,
    *, allow_target_on: bool = False, inject_bout: bool = True,
) -> np.ndarray | None:
    """Pull a same-house window of length ``window_samples`` from the requested
    channel, then (optionally) additively insert the bout submeter at the
    centre.

    ``allow_target_on``: when True, the BG window can include natural bouts of
    the target appliance (so the BG distribution matches the natural-window
    distribution). When False (default), the BG sampler enforces target-OFF.

    ``inject_bout``: when False (null/placebo mode), return the BG window
    untouched -- no bout inserted. Used to measure how much of the probe's
    AUC comes from BG-sampling methodology bias alone (vs. the insertion
    artifact itself).
    """
    house = int(bout_row["house_id"])
    bs, be = int(bout_row["start_ns"]), int(bout_row["end_ns"])
    ctx_s = window_samples * dt_s
    bg = sampler.sample(
        target=str(bout_row["appliance"]),
        context_length_s=ctx_s,
        rng=rng,
        same_house_as=house,
        margin_s=min(ctx_s * 0.05, 60.0),
        allow_target_on=allow_target_on,
    )
    if bg is None or bg.mains_w.size == 0:
        return None

    if source == "mains":
        bg_signal = bg.mains_w
    elif source == "submeter":
        # Read the same time window from the appliance's own submeter. By
        # construction it should be ~0 for the entire window (target-OFF).
        bg_signal = load_meter_window_grid(
            house, int(bout_row["meter_id"]),
            bg.start_ns, bg.end_ns, dt_s=dt_s,
        )
    else:
        raise ValueError(f"unknown source: {source}")

    if bg_signal.size == 0:
        return None

    if not inject_bout:
        return _centre_pad(bg_signal, window_samples)

    sub = load_meter_window_grid(
        int(bout_row["house_id"]), int(bout_row["meter_id"]),
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

BASELINE_MODES = ("insertion", "empty_bg", "natural_natural")


def probe_appliance(
    bouts: pl.DataFrame, sampler: BackgroundSampler, appliance: str,
    n_per_class: int, rng: np.random.Generator,
    *,
    source: str = "mains",
    dt_s: float = NOMINAL_DT_S,
    window_factor: float = 4.0,
    max_window_samples: int = 600,
    allow_target_on: bool = False,
    natural_isolated_only: bool = False,
    baseline_mode: str = "insertion",
) -> dict:
    """Train XGBoost on raw windows of the requested channel; return AUC.

    ``baseline_mode``:
      - 'insertion'       : real probe (natural vs target-OFF BG + bout)
      - 'empty_bg'        : null-1 (natural vs target-OFF BG, no bout)
      - 'natural_natural' : placebo (two disjoint natural-window draws,
                            labels assigned arbitrarily). Expected AUC = 0.5
                            if the classifier has no spurious signal.
    """
    if baseline_mode not in BASELINE_MODES:
        raise ValueError(f"unknown baseline_mode: {baseline_mode}")

    sub = bouts.filter((pl.col("appliance") == appliance) & (pl.col("split") == "train"))
    if sub.height < 50:
        return {"appliance": appliance, "source": source, "skipped": True,
                "reason": f"only {sub.height} bouts"}

    dur_med_s = float(sub["duration_s"].median())
    window_samples = max(20, int(window_factor * dur_med_s / dt_s))
    window_samples = min(window_samples, max_window_samples)

    house_app_bouts: dict[int, np.ndarray] = {}
    if natural_isolated_only:
        for h in sub["house_id"].unique().to_list():
            arr = (sub.filter(pl.col("house_id") == h)
                      .select(["start_ns", "end_ns"])
                      .sort("start_ns").to_numpy().astype("int64"))
            house_app_bouts[int(h)] = arr

    rows = sub.to_dicts()
    rng.shuffle(rows)

    nat_X: list[np.ndarray] = []
    ins_X: list[np.ndarray] = []
    n_natural_rejected_clustered = 0

    def _make_natural(row):
        arr = house_app_bouts.get(int(row["house_id"])) if natural_isolated_only else None
        return collect_natural_window(
            row, window_samples, dt_s, source,
            natural_isolated_only=natural_isolated_only,
            house_appliance_bouts=arr,
        )

    if baseline_mode == "natural_natural":
        # Both classes drawn from the same natural-window pool, but using
        # DISJOINT row pools to avoid trivial duplicates.
        for row in rows:
            if len(nat_X) >= n_per_class and len(ins_X) >= n_per_class:
                break
            w = _make_natural(row)
            if w is None:
                if natural_isolated_only:
                    n_natural_rejected_clustered += 1
                continue
            # Alternate which class gets each successful sample
            if len(nat_X) <= len(ins_X):
                nat_X.append(w)
            else:
                ins_X.append(w)
    else:
        inject_bout = (baseline_mode == "insertion")
        for row in rows:
            if len(nat_X) < n_per_class:
                w = _make_natural(row)
                if w is not None:
                    nat_X.append(w)
                elif natural_isolated_only:
                    n_natural_rejected_clustered += 1
            if len(ins_X) < n_per_class:
                w = collect_inserted_window(
                    row, sampler, window_samples, dt_s, rng, source,
                    allow_target_on=allow_target_on,
                    inject_bout=inject_bout,
                )
                if w is not None:
                    ins_X.append(w)
            if len(nat_X) >= n_per_class and len(ins_X) >= n_per_class:
                break

    if len(nat_X) < 30 or len(ins_X) < 30:
        return {
            "appliance": appliance, "source": source, "skipped": True,
            "reason": f"too few collected (nat={len(nat_X)}, ins={len(ins_X)}, "
                      f"isolated_rejected={n_natural_rejected_clustered})",
        }

    X = np.vstack([np.stack(nat_X), np.stack(ins_X)]).astype("float32")
    y = np.concatenate([np.zeros(len(nat_X)), np.ones(len(ins_X))]).astype("int64")

    auc, importances = _train_and_score(X, y)
    top_k = sorted(range(window_samples), key=lambda i: -importances[i])[:8]

    return {
        "appliance": appliance,
        "source": source,
        "allow_target_on": bool(allow_target_on),
        "natural_isolated_only": bool(natural_isolated_only),
        "baseline_mode": str(baseline_mode),
        "n_class0": len(nat_X), "n_class1": len(ins_X),
        "n_natural_rejected_clustered": int(n_natural_rejected_clustered),
        "window_samples": int(window_samples),
        "window_seconds": float(window_samples * dt_s),
        "median_bout_duration_s": dur_med_s,
        "auc_5fold_mean": float(auc),
        "top_sample_positions": [
            {"sample_idx": int(i),
             "seconds_from_window_start": float(i * dt_s),
             "importance": float(importances[i])}
            for i in top_k
        ],
    }


def _train_and_score(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
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
    importances_acc = np.zeros(X.shape[1], dtype="float64")
    for train_idx, test_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[test_idx], proba)))
        if hasattr(clf, "feature_importances_"):
            importances_acc += clf.feature_importances_
    return float(np.mean(aucs)), importances_acc / 5.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index",
                    default=str(UKD_HAYSTACK_DIR / "bout_index.parquet"))
    ap.add_argument("--out",
                    default=str(UKD_HAYSTACK_DIR / "diagnostics" / "shortcut_probe.json"))
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--appliances", nargs="+", default=None,
                    help="Default: any appliance with >=200 train bouts.")
    ap.add_argument("--window-factor", type=float, default=4.0,
                    help="Window length = window_factor * median_bout_duration.")
    ap.add_argument("--max-window-samples", type=int, default=600,
                    help="Cap on window length in samples (600 @ 6s = 1h).")
    ap.add_argument("--sources", nargs="+",
                    choices=["mains", "submeter"], default=["mains", "submeter"],
                    help="Which channels to probe (default: both).")
    ap.add_argument("--allow-target-on", action="store_true",
                    help="Drop the target-OFF constraint when sampling BG "
                         "(matches the BG distribution to natural windows; "
                         "the inserted bout is then summed on top of any "
                         "natural target activity already in the window).")
    ap.add_argument("--natural-isolated-only", action="store_true",
                    help="Restrict natural samples to bouts that are the ONLY "
                         "instance of their appliance in the window (closes the "
                         "temporal-clustering shortcut on the submeter channel).")
    ap.add_argument(
        "--baseline-mode",
        choices=BASELINE_MODES, default="insertion",
        help=("'insertion' (default): real probe, natural vs (target-OFF BG + bout). "
              "'empty_bg': null-1, natural-with-bout vs target-OFF BG with NO bout "
              "(measures BG-sampler-vs-natural-bout-time methodology bias). "
              "'natural_natural': placebo, two disjoint natural-window draws with "
              "arbitrary labels (expected AUC = 0.5; any deviation surfaces "
              "classifier bias from the windowing/feature pipeline)."),
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
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

    out = {"per_appliance": []}
    for source in args.sources:
        print(f"\n=== source = {source} ===")
        for app in sorted(targets):
            print(f"  {app:<18} ...", end=" ", flush=True)
            result = probe_appliance(
                bouts, sampler, app, args.n_per_class, rng,
                source=source,
                window_factor=args.window_factor,
                max_window_samples=args.max_window_samples,
                allow_target_on=args.allow_target_on,
                natural_isolated_only=args.natural_isolated_only,
                baseline_mode=args.baseline_mode,
            )
            out["per_appliance"].append(result)
            if result.get("skipped"):
                print(f"SKIPPED: {result['reason']}")
            else:
                print(f"AUC={result['auc_5fold_mean']:.3f}  "
                      f"window={result['window_samples']}samp/{result['window_seconds']:.0f}s  "
                      f"(n0={result['n_class0']}, n1={result['n_class1']})")

    out["summary_by_source"] = {}
    for source in args.sources:
        aucs = [r["auc_5fold_mean"] for r in out["per_appliance"]
                if r.get("source") == source and not r.get("skipped")]
        out["summary_by_source"][source] = {
            "n_appliances": len(aucs),
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_max": float(np.max(aucs)) if aucs else None,
            "n_above_065": int(sum(1 for a in aucs if a > 0.65)),
            "n_above_080": int(sum(1 for a in aucs if a > 0.80)),
            "blocking_threshold": 0.80,
            "passes_gate": all(a <= 0.80 for a in aucs) if aucs else None,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {out_path}")
    for source, summary in out["summary_by_source"].items():
        if summary["auc_mean"] is None:
            continue
        print(f"  [{source}]  mean AUC={summary['auc_mean']:.3f}  "
              f"max={summary['auc_max']:.3f}  "
              f"passes gate={summary['passes_gate']}")


if __name__ == "__main__":
    main()
