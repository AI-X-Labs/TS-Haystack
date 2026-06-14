"""
Cross-distribution classifier transfer test.

Trains a multi-class XGBoost on INSERTED appliance samples (target-OFF BG +
bout) using only TRAIN-split bouts. Then evaluates accuracy on:

  - test-inserted: held-out inserted samples (TEST-split bouts, same pipeline)
  - test-natural: natural samples (real recording windows centred on TEST-split
    bouts)

If accuracy on test-natural is comparable to test-inserted, the synthetic
distribution preserves enough of the appliance signature for a classifier to
generalize -- even though natural and inserted are statistically distinguishable
(per the insertion-shortcut probe), they share the structural appliance signal
that a classifier can learn.

Reports per-class accuracy + confusion matrices on both test sets, for mains
and submeter channels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.loader import (
    NOMINAL_DT_S,
    load_meter_window_grid,
    mains_meter_id,
)
from src.datasets.uk_dale_haystack.scripts.probe_insertion_shortcut import (
    _bout_window_bounds,
    _centre_pad,
)

DEFAULT_BOUT_INDEX = Path("data/uk_dale/uk_dale_haystack/bout_index.parquet")
DEFAULT_OUT = Path("data/uk_dale/uk_dale_haystack/diagnostics/classifier_transfer.json")


def _make_natural_window(bout_row: dict, window_samples: int, dt_s: float, source: str) -> np.ndarray | None:
    house = int(bout_row["house_id"])
    bs, be = int(bout_row["start_ns"]), int(bout_row["end_ns"])
    win_start, win_end = _bout_window_bounds(bs, be, window_samples, dt_s)
    meter = mains_meter_id(house) if source == "mains" else int(bout_row["meter_id"])
    sig = load_meter_window_grid(house, meter, win_start, win_end, dt_s=dt_s)
    if sig.size == 0:
        return None
    return _centre_pad(sig, window_samples)


def _make_inserted_window(
    bout_row: dict, sampler: BackgroundSampler, window_samples: int,
    dt_s: float, rng: np.random.Generator, source: str,
) -> np.ndarray | None:
    house = int(bout_row["house_id"])
    bs, be = int(bout_row["start_ns"]), int(bout_row["end_ns"])
    ctx_s = window_samples * dt_s
    bg = sampler.sample(
        target=str(bout_row["appliance"]),
        context_length_s=ctx_s, rng=rng,
        same_house_as=house,
        margin_s=min(ctx_s * 0.05, 60.0),
    )
    if bg is None or bg.mains_w.size == 0:
        return None
    if source == "mains":
        bg_signal = bg.mains_w
    else:
        bg_signal = load_meter_window_grid(
            house, int(bout_row["meter_id"]), bg.start_ns, bg.end_ns, dt_s=dt_s,
        )
    sub = load_meter_window_grid(
        house, int(bout_row["meter_id"]), bs, be + int(dt_s * 1e9), dt_s=dt_s,
    )
    if bg_signal.size == 0 or sub.size == 0:
        return None
    out = _centre_pad(bg_signal, window_samples).copy()
    sub = sub[:window_samples]
    centre = window_samples // 2
    pos = max(0, centre - sub.size // 2)
    end = min(window_samples, pos + sub.size)
    out[pos:end] += sub[: end - pos]
    return out


def _collect(
    bouts: pl.DataFrame, sampler: BackgroundSampler,
    appliances: Sequence[str], split: str,
    n_per_class: int, window_samples: int, dt_s: float,
    source: str, mode: str, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the requested split / mode / source."""
    X: list[np.ndarray] = []
    y: list[int] = []
    for cls_idx, app in enumerate(appliances):
        sub = bouts.filter((pl.col("appliance") == app) & (pl.col("split") == split))
        if sub.height == 0:
            continue
        rows = sub.sample(n=min(sub.height, n_per_class * 4),
                          with_replacement=False,
                          seed=int(rng.integers(0, 2**31 - 1))).to_dicts()
        n_taken = 0
        for row in rows:
            if n_taken >= n_per_class:
                break
            if mode == "natural":
                w = _make_natural_window(row, window_samples, dt_s, source)
            elif mode == "inserted":
                w = _make_inserted_window(row, sampler, window_samples, dt_s, rng, source)
            else:
                raise ValueError(f"unknown mode: {mode}")
            if w is None:
                continue
            X.append(w)
            y.append(cls_idx)
            n_taken += 1
    if not X:
        return np.empty((0, window_samples), dtype="float32"), np.empty(0, dtype="int64")
    return (
        np.stack(X).astype("float32"),
        np.asarray(y, dtype="int64"),
    )


def _train_and_eval(
    X_train, y_train, eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    appliance_names: list[str],
) -> dict:
    import xgboost as xgb
    n_class = len(appliance_names)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=n_class,
        n_jobs=-1, verbosity=0,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    out = {"appliances": appliance_names, "n_train": int(len(y_train)),
           "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
           "eval": {}}
    for name, (X_e, y_e) in eval_sets.items():
        if y_e.size == 0:
            out["eval"][name] = {"skipped": True}
            continue
        pred = model.predict(X_e)
        proba = model.predict_proba(X_e)
        cm = confusion_matrix(y_e, pred, labels=list(range(n_class))).tolist()
        per_class_acc = {}
        for i, app in enumerate(appliance_names):
            mask = y_e == i
            per_class_acc[app] = (float(accuracy_score(y_e[mask], pred[mask]))
                                  if mask.sum() else None)
        out["eval"][name] = {
            "n": int(len(y_e)),
            "accuracy": float(accuracy_score(y_e, pred)),
            "macro_f1": float(f1_score(y_e, pred, average="macro")),
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm,
        }
    return out


def _print_table(result: dict, source: str) -> None:
    apps = result["appliances"]
    train_acc = result["train_accuracy"]
    print(f"\n[{source}]  train accuracy = {train_acc:.3f}  (n_train = {result['n_train']})")
    print(f"  {'appliance':<20} test_inserted    test_natural    Δ (natural - inserted)")
    print(f"  {'-' * 70}")
    e_ins = result["eval"].get("test_inserted", {})
    e_nat = result["eval"].get("test_natural", {})
    pcs_ins = e_ins.get("per_class_accuracy", {})
    pcs_nat = e_nat.get("per_class_accuracy", {})
    for app in apps:
        a_ins = pcs_ins.get(app)
        a_nat = pcs_nat.get(app)
        if a_ins is None and a_nat is None:
            continue
        a_ins_s = f"{a_ins:.3f}" if a_ins is not None else "  -  "
        a_nat_s = f"{a_nat:.3f}" if a_nat is not None else "  -  "
        delta = (a_nat - a_ins) if (a_ins is not None and a_nat is not None) else None
        delta_s = f"{delta:+.3f}" if delta is not None else "  -  "
        print(f"  {app:<20} {a_ins_s:<16} {a_nat_s:<16} {delta_s}")
    print(f"  {'-' * 70}")
    print(f"  {'OVERALL':<20} acc={e_ins.get('accuracy', float('nan')):.3f}      "
          f"acc={e_nat.get('accuracy', float('nan')):.3f}      "
          f"Δ={(e_nat.get('accuracy', float('nan')) - e_ins.get('accuracy', float('nan'))):+.3f}")
    print(f"  {'':<20} f1={e_ins.get('macro_f1', float('nan')):.3f}        "
          f"f1={e_nat.get('macro_f1', float('nan')):.3f}        "
          f"Δ={(e_nat.get('macro_f1', float('nan')) - e_ins.get('macro_f1', float('nan'))):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index", type=Path, default=DEFAULT_BOUT_INDEX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-per-class", type=int, default=200,
                    help="Train + test-inserted + test-natural samples per appliance.")
    ap.add_argument("--window-samples", type=int, default=600,
                    help="Fixed window length (samples) for all classes. 600 @ 6s = 1h.")
    ap.add_argument("--appliances", nargs="+", default=None,
                    help="Subset to test. Default: every appliance with >= n_per_class bouts in train AND test.")
    ap.add_argument("--sources", nargs="+", choices=["mains", "submeter"], default=["mains", "submeter"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.bout_index.exists():
        raise SystemExit(f"{args.bout_index} missing.")

    bouts = pl.read_parquet(args.bout_index)
    rng = np.random.default_rng(args.seed)

    # Pick appliances with enough bouts in both train and test
    train_counts = Counter(bouts.filter(pl.col("split") == "train")["appliance"].to_list())
    test_counts = Counter(bouts.filter(pl.col("split") == "test")["appliance"].to_list())
    eligible = sorted([
        a for a in train_counts
        if train_counts[a] >= args.n_per_class and test_counts.get(a, 0) >= max(20, args.n_per_class // 4)
    ])
    if args.appliances:
        eligible = [a for a in args.appliances if a in eligible]
    if len(eligible) < 2:
        raise SystemExit(f"need >= 2 eligible appliances, got {eligible}")

    print(f"Eligible appliances ({len(eligible)}): {eligible}")
    print(f"  per-class counts (train | test):")
    for a in eligible:
        print(f"    {a:<20} {train_counts[a]:>6}  |  {test_counts.get(a,0):>6}")

    sampler = BackgroundSampler(bouts, "train")
    sampler_test = BackgroundSampler(bouts, "test")

    out_per_source: dict[str, dict] = {}
    for source in args.sources:
        print(f"\n=== {source} ===")
        print(" Collecting train (inserted, train split) ...")
        X_tr, y_tr = _collect(bouts, sampler, eligible, "train",
                              args.n_per_class, args.window_samples,
                              NOMINAL_DT_S, source, "inserted", rng)
        n_test_per_class = max(20, args.n_per_class // 4)
        print(f" Collecting test_inserted (test split, n={n_test_per_class}/class) ...")
        X_te_ins, y_te_ins = _collect(bouts, sampler_test, eligible, "test",
                                      n_test_per_class, args.window_samples,
                                      NOMINAL_DT_S, source, "inserted", rng)
        print(f" Collecting test_natural  (test split, n={n_test_per_class}/class) ...")
        X_te_nat, y_te_nat = _collect(bouts, sampler_test, eligible, "test",
                                      n_test_per_class, args.window_samples,
                                      NOMINAL_DT_S, source, "natural", rng)
        result = _train_and_eval(
            X_tr, y_tr,
            {"test_inserted": (X_te_ins, y_te_ins),
             "test_natural":  (X_te_nat, y_te_nat)},
            eligible,
        )
        out_per_source[source] = result
        _print_table(result, source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"window_samples": args.window_samples, "n_per_class": args.n_per_class,
         "appliances": eligible, "results": out_per_source},
        indent=2,
    ))
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
