#!/usr/bin/env python3
"""
Clean insertion-validation report.

Runs the shortcut probe in two modes per appliance per channel:

  PLACEBO  (natural vs natural, arbitrary labels)
    Both classes drawn from the same natural-window pool. Expected AUC ~ 0.5.
    If higher, the windowing/CV/feature pipeline has spurious bias and the
    insertion AUC isn't interpretable.

  INSERTION (natural vs target-OFF BG + bout)
    Class 0: real-mains (or real-submeter) window centred on a real bout.
    Class 1: target-OFF same-house BG window + the bout additively inserted
             at the centre.
    Measures the residual sim-to-real distributional gap after the additive
    insertion has done its work.

Both probes are run on both channels (mains, submeter) so we can see whether
a model that only sees mains (the actual benchmark) faces the same shortcut
risk as a model that could see per-appliance submeters (it cannot, in v1).

Output:
  - Console table comparing PLACEBO vs INSERTION per (channel, appliance)
  - JSON sidecar at the requested --out path with the full per-appliance dicts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.loader import UKD_HAYSTACK_DIR, NOMINAL_DT_S
from src.datasets.uk_dale_haystack.scripts.probe_insertion_shortcut import (
    probe_appliance,
)


PLACEBO_TOLERANCE = 0.10  # placebo must be within 0.5 +/- this to be valid
INSERTION_BLOCKING = 0.80  # plan's hard gate on the insertion probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index",
                    default=str(UKD_HAYSTACK_DIR / "bout_index.parquet"))
    ap.add_argument("--out",
                    default=str(UKD_HAYSTACK_DIR / "diagnostics"
                                / "insertion_validation.json"))
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--appliances", nargs="+", default=None,
                    help="Default: any appliance with >=200 train bouts.")
    ap.add_argument("--sources", nargs="+",
                    choices=["mains", "submeter"], default=["mains", "submeter"])
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
            row = {"appliance": app, "source": source}
            for mode in ("natural_natural", "insertion"):
                rng = np.random.default_rng(args.seed)
                result = probe_appliance(
                    bouts, sampler, app, args.n_per_class, rng,
                    source=source,
                    baseline_mode=mode,
                )
                if result.get("skipped"):
                    row[mode] = {"skipped": True, "reason": result["reason"]}
                    print(f"  {app:<18} {mode:<18} SKIPPED: {result['reason']}")
                    continue
                row[mode] = result
                print(f"  {app:<18} {mode:<18} AUC={result['auc_5fold_mean']:.3f}  "
                      f"window={result['window_samples']}samp/{result['window_seconds']:.0f}s")
            out["per_run"].append(row)
        print()

    print_summary_table(out, args.sources, targets)
    print()
    print_gates(out, args.sources)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {out_path}")


def print_summary_table(out: dict, sources: list[str], appliances: list[str]) -> None:
    print()
    print(" Channel    Appliance         Placebo  Insertion   Δ")
    print(" " + "-" * 60)
    for source in sources:
        for app in appliances:
            row = next((r for r in out["per_run"]
                        if r["appliance"] == app and r["source"] == source), None)
            if row is None:
                continue
            placebo = row.get("natural_natural", {}).get("auc_5fold_mean")
            insert = row.get("insertion", {}).get("auc_5fold_mean")
            if placebo is None or insert is None:
                continue
            delta = insert - placebo
            print(f" {source:<10} {app:<18} {placebo:>5.3f}    {insert:>5.3f}    "
                  f"{delta:+.3f}")


def print_gates(out: dict, sources: list[str]) -> None:
    print(" Gate evaluation:")
    for source in sources:
        rows = [r for r in out["per_run"] if r["source"] == source
                and "natural_natural" in r and "insertion" in r
                and not r["natural_natural"].get("skipped")
                and not r["insertion"].get("skipped")]
        if not rows:
            continue
        placebos = [r["natural_natural"]["auc_5fold_mean"] for r in rows]
        inserts  = [r["insertion"]["auc_5fold_mean"] for r in rows]
        max_placebo_dev = max(abs(p - 0.5) for p in placebos)
        max_insert     = max(inserts)
        placebo_ok = max_placebo_dev <= PLACEBO_TOLERANCE
        insert_ok = max_insert <= INSERTION_BLOCKING
        print(f"  [{source}]  placebo max |AUC-0.5| = {max_placebo_dev:.3f} "
              f"({'PASS' if placebo_ok else 'FAIL'} <= {PLACEBO_TOLERANCE})")
        print(f"  [{source}]  insertion max AUC     = {max_insert:.3f} "
              f"({'PASS' if insert_ok else 'FAIL'} <= {INSERTION_BLOCKING})")


if __name__ == "__main__":
    main()
