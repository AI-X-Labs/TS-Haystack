"""
Per-sample inspection plots.

Renders the mains trace with each inserted needle as a coloured shaded band
(labelled at the band's top), background's natural other-appliance bouts as
light dashed verticals, and the Q/A text at the figure bottom.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.uk_dale_haystack.core.activity_regimes import (
    ACTIVITY_TO_REGIME,
    REGIMES,
)
from src.datasets.uk_dale_haystack.loader import (
    load_meter_window_grid,
    mains_meter_id,
    NOMINAL_DT_S,
)


REGIME_COLORS = {
    "impulse":    "#d73027",
    "long_cycle": "#1f78b4",
    "cooking":    "#ff7f00",
    "refrig":     "#33a02c",
}


def reconstruct_sample_signal(row: dict[str, Any]) -> np.ndarray:
    """Reconstruct the additive mains for a stored sample row.

    Reads the mains window from the loader, then sums each inserted needle
    (pulled from the source meter at its original bout timestamps) back in.
    """
    house = int(row["background_house_id"])
    start_ns = int(row["background_start_ns"])
    end_ns = int(row["background_end_ns"])
    dt_s = float(row["dt_s"])

    mains = load_meter_window_grid(
        house, mains_meter_id(house), start_ns, end_ns, dt_s=dt_s,
    ).copy()

    needles = json.loads(row["needles_json"])
    for n in needles:
        pos = int(n["insert_position_samples"])
        n_samples = int(n["insert_duration_samples"])
        # Pull the submeter trace from its actual source-time bout window
        dt_ns = int(dt_s * 1e9)
        sub = load_meter_window_grid(
            int(n["source_house_id"]), int(n["source_meter_id"]),
            int(n["source_start_ns"]),
            int(n["source_end_ns"]) + dt_ns,
            dt_s=dt_s,
        )
        # Trim or pad to match insert_duration_samples (anomaly classes may
        # have changed length, e.g. truncated_cycle).
        if sub.shape[0] >= n_samples:
            sub = sub[:n_samples]
        else:
            sub = np.concatenate([sub, np.zeros(n_samples - sub.shape[0], dtype=sub.dtype)])
        # Apply abnormal_peak scaling at reconstruction time (the source bout
        # is nominal; the anomaly only exists in the synthesized sample).
        ap = n.get("anomaly_params") or {}
        if n.get("anomaly_class") == "abnormal_peak" and "scale" in ap:
            ceiling = float(ap.get("ceiling_w", 1e9))
            sub = np.minimum(sub * float(ap["scale"]), ceiling)
        mains[pos:pos + n_samples] += sub
    return mains


def plot_sample_row(
    row: dict[str, Any],
    out_path: Path,
    *,
    mains_w: np.ndarray | None = None,
    title_extra: str = "",
) -> Path:
    """Render one sample's plot to ``out_path``.

    If ``mains_w`` is None, the mains will be reconstructed from the row.
    """
    if mains_w is None:
        mains_w = reconstruct_sample_signal(row)

    dt_s = float(row["dt_s"])
    n = mains_w.shape[0]
    t_min = (np.arange(n) * dt_s) / 60.0

    needles = json.loads(row["needles_json"])
    other_bouts = json.loads(row["other_bouts_json"])

    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.plot(t_min, mains_w, color="black", lw=0.55, label="mains (W)")

    # Inserted needle bands (above 0.05 ymin)
    for nd in needles:
        regime = ACTIVITY_TO_REGIME.get(nd["appliance"], "long_cycle")
        color = REGIME_COLORS.get(regime, "#666")
        x0 = (nd["insert_position_samples"] * dt_s) / 60.0
        x1 = ((nd["insert_position_samples"] + nd["insert_duration_samples"]) * dt_s) / 60.0
        edge = "red" if nd.get("is_anomalous") else color
        hatch = "//" if nd.get("is_anomalous") else None
        ax.axvspan(x0, x1, ymin=0.05, ymax=1.0,
                   color=color, alpha=0.20, edgecolor=edge, linewidth=1.2,
                   hatch=hatch, zorder=1)
        label = nd["appliance"]
        if nd.get("is_anomalous"):
            label += f"\n[anomaly: {nd.get('anomaly_class')}]"
        ax.text(
            (x0 + x1) / 2, ax.get_ylim()[1] * 0.97,
            label, ha="center", va="top", fontsize=7,
            color=edge, weight="bold",
            bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, pad=1),
        )

    # Natural other_bouts -- light dashed verticals
    for ob in other_bouts:
        x0 = (ob["start_sample"] * dt_s) / 60.0
        x1 = (ob["end_sample"] * dt_s) / 60.0
        ax.axvspan(x0, x1, ymin=0.0, ymax=0.04, color="#888", alpha=0.4,
                   linewidth=0)

    ax.set_xlabel("minutes from window start")
    ax.set_ylabel("mains power (W)")
    ax.set_xlim(0, t_min[-1] if n else 1)
    ax.grid(alpha=0.25)

    # Q/A text
    qa = (
        f"Q: {row['question']}\n"
        f"A: {row['answer']}  ({row['answer_type']})"
    )
    fig.text(0.01, 0.01, qa, fontsize=9, ha="left", va="bottom",
             family="monospace",
             bbox=dict(facecolor="#fdfdfd", edgecolor="#999", pad=4))
    title = (
        f"{row['task_type']}  ctx={row['context_length_s']}s  split={row['split']}  "
        f"h{row['background_house_id']}  needles={row['n_needles']}{title_extra}"
    )
    ax.set_title(title)
    fig.subplots_adjust(bottom=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
