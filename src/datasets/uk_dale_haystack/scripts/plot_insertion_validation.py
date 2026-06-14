#!/usr/bin/env python3
"""
Visual validation of validate_insertion.py.

For every (appliance, channel) combo this script generates two folders of
pair-plots, both stacked top-vs-bottom on a shared y-axis:

  insertion/    top = natural (real-channel window centred on a real bout)
                bot = target-OFF BG window of same length + the bout
                      ADDITIVELY inserted at the centre. No mean-shift, no
                      blending -- the insertion is literally
                      `bg[pos:end] += submeter`. The plot annotates
                      pre-bout and post-bout sample values just outside the
                      bout edges so the additive nature is auditable.

  placebo/      top = natural window from one bout
                bot = natural window from a DIFFERENT, randomly-chosen bout
                Both come from the same distribution; XGBoost should not be
                able to tell them apart.

Output: data/uk_dale/inspect/insertion_validation/{mode}/{source}/{appliance}/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.loader import (
    UKD_HAYSTACK_DIR,
    NOMINAL_DT_S,
)
from src.datasets.uk_dale_haystack.scripts.probe_insertion_shortcut import (
    collect_inserted_window,
    collect_natural_window,
)


OUT_DIR = Path("data/uk_dale/inspect/insertion_validation")


def plot_pair(
    top: np.ndarray, bot: np.ndarray,
    bout_centre_idx: int, bout_n_samples: int,
    appliance: str, source: str, mode: str, dt_s: float,
    out_path: Path, idx: int,
    *,
    top_label: str, bot_label: str,
) -> None:
    n = top.shape[0]
    t_min = (np.arange(n) * dt_s) / 60.0
    bout_lo = max(0, bout_centre_idx - bout_n_samples // 2)
    bout_hi = min(n, bout_centre_idx + bout_n_samples // 2)
    bout_x0 = (bout_lo * dt_s) / 60.0
    bout_x1 = (bout_hi * dt_s) / 60.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 5.8), sharex=True, sharey=True)

    for ax, sig, title in ((ax0, top, top_label), (ax1, bot, bot_label)):
        ax.plot(t_min, sig, color="black", lw=0.6)
        ax.axvspan(bout_x0, bout_x1, color="#4caf50", alpha=0.18, linewidth=0)
        ax.set_ylabel("watts")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25)

    # For insertion mode: annotate the additive boundary so the user can
    # confirm there's no mean-shift or smoothing.
    if mode == "insertion" and 0 < bout_lo < n - 1 and 0 < bout_hi < n - 1:
        for label, idx_x in (("pre", bout_lo - 1), ("post", bout_hi)):
            x_min = (idx_x * dt_s) / 60.0
            for ax, sig in ((ax0, top), (ax1, bot)):
                y = float(sig[idx_x])
                ax.scatter([x_min], [y], color="red", s=18, zorder=4)
                ax.annotate(
                    f"{label}={y:.0f}W",
                    xy=(x_min, y),
                    xytext=(x_min + 0.05 * t_min[-1], y),
                    fontsize=7, color="red",
                    arrowprops=dict(arrowstyle="-", color="red", lw=0.6),
                )

    ax1.set_xlabel("minutes from window start (bout centred)")
    fig.suptitle(
        f"{appliance}  --  {source}  --  {mode}  --  sample {idx:02d}  "
        f"(window={n}samp, {n*dt_s:.0f}s)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout-index",
                    default=str(UKD_HAYSTACK_DIR / "bout_index.parquet"))
    ap.add_argument("--appliances", nargs="+", default=None,
                    help="Default: any appliance with >=200 train bouts.")
    ap.add_argument("--sources", nargs="+",
                    choices=["mains", "submeter"], default=["mains", "submeter"])
    ap.add_argument("--modes", nargs="+",
                    choices=["insertion", "placebo"], default=["insertion", "placebo"])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--window-factor", type=float, default=4.0)
    ap.add_argument("--max-window-samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bouts = pl.read_parquet(args.bout_index)
    sampler = BackgroundSampler(bouts, "train")
    rng = np.random.default_rng(args.seed)
    dt_s = NOMINAL_DT_S

    if args.appliances:
        targets = list(args.appliances)
    else:
        targets = []
        for app in bouts.filter(pl.col("split") == "train")["appliance"].unique().to_list():
            n = bouts.filter((pl.col("appliance") == app) & (pl.col("split") == "train")).height
            if n >= 200:
                targets.append(app)
    targets.sort()

    for source in args.sources:
        for app in targets:
            sub = bouts.filter((pl.col("appliance") == app) & (pl.col("split") == "train"))
            if sub.height < 2 * args.n_samples:
                print(f"  [{source}/{app}] SKIP (only {sub.height} bouts)")
                continue

            dur_med_s = float(sub["duration_s"].median())
            window_samples = max(20, int(args.window_factor * dur_med_s / dt_s))
            window_samples = min(window_samples, args.max_window_samples)

            rows = sub.to_dicts()
            rng.shuffle(rows)
            row_iter = iter(rows)

            for mode in args.modes:
                plotted = 0
                attempts = 0
                while plotted < args.n_samples and attempts < args.n_samples * 6:
                    attempts += 1
                    try:
                        row_top = next(row_iter)
                    except StopIteration:
                        break

                    top = collect_natural_window(row_top, window_samples, dt_s, source)
                    if top is None:
                        continue

                    if mode == "insertion":
                        bot = collect_inserted_window(
                            row_top, sampler, window_samples, dt_s, rng, source,
                        )
                        top_label = f"NATURAL  ({source})  --  bout in real recording"
                        bot_label = (
                            f"INSERTED ({source})  --  target-OFF BG + additive needle "
                            "(no mean-shift, no blending)"
                        )
                    else:
                        try:
                            row_bot = next(row_iter)
                        except StopIteration:
                            break
                        bot = collect_natural_window(row_bot, window_samples, dt_s, source)
                        if bot is None:
                            continue
                        top_label = f"NATURAL #1 ({source})  --  bout in real recording"
                        bot_label = f"NATURAL #2 ({source})  --  different bout, same distribution"

                    if bot is None:
                        continue

                    bout_n = max(1, int(round((row_top["end_ns"] - row_top["start_ns"]) / 1e9 / dt_s)))
                    centre = window_samples // 2
                    out_path = OUT_DIR / mode / source / app / f"sample_{plotted:02d}.png"
                    plot_pair(
                        top, bot, centre, bout_n, app, source, mode, dt_s,
                        out_path, plotted,
                        top_label=top_label, bot_label=bot_label,
                    )
                    plotted += 1
                print(f"  [{mode}/{source}/{app}] -> {plotted} pair plots in "
                      f"{OUT_DIR / mode / source / app}")

    print(f"\nDone. Inspect in: {OUT_DIR}")


if __name__ == "__main__":
    main()
