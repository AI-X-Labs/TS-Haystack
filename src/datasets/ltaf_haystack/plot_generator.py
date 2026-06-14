# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Plot generator for LTAF-Haystack visual verification.

Produces a single PIL image showing the 2-lead ECG with the sample's
annotations painted on top:

* Rhythm bands — translucent colour bands per rhythm bout.
* Beat markers — per-symbol strip above the signal.
* Highlights — bold outline around the task's answer region (used by
  localization / multi_hop / anomaly_localization).
* Q/A footer — question text + answer at the bottom of the figure.
"""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBeatEvent,
    LTAFBoutRecord,
)


# Colours chosen for colour-blind safety and high-contrast painting on the
# black-on-white ECG trace. Unknown rhythms fall back to gray.
RHYTHM_COLOURS: Dict[str, str] = {
    "NSR":  "#2ca02c",
    "AFIB": "#d62728",
    "SBR":  "#17becf",
    "AB":   "#bcbd22",
    "B":    "#ff7f0e",
    "T":    "#e377c2",
    "SVTA": "#9467bd",
    "VT":   "#8c564b",
    "IVR":  "#1f77b4",
}

BEAT_COLOURS: Dict[str, str] = {
    "N": "#333333",
    "A": "#ff7f0e",
    "V": "#d62728",
    "Q": "#7f7f7f",
}

# (facecolor, edgecolor) per highlight label. "answer" is the canonical
# yellow band; "context" marks a region the question refers to but is not
# itself the answer (e.g. the target bout in `antecedent`). Unknown labels
# fall back to "answer" for backward compatibility.
HIGHLIGHT_COLOURS: Dict[str, Tuple[str, str]] = {
    "answer":  ("#ffeb3b", "#c19500"),
    "context": ("#80cbc4", "#00695c"),
}

def downsample_for_plotting(
    data: np.ndarray, max_points: int = 8000
) -> Tuple[np.ndarray, int]:
    n = data.shape[0]
    if n <= max_points:
        return data, 1
    factor = max(1, n // max_points)
    return data[::factor], factor


def _rhythm_colour(rhythm: str) -> str:
    return RHYTHM_COLOURS.get(rhythm, "#999999")


def create_ecg_plot(
    signals: np.ndarray,
    channel_names: Sequence[str] = ("ECG1", "ECG2"),
    rhythm_bands: Optional[Sequence[LTAFBoutRecord]] = None,
    beat_markers: Optional[Dict[str, Sequence[int]]] = None,
    highlights: Optional[Sequence[Tuple[int, int, str]]] = None,
    title: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    source_hz: int = 128,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 110,
    max_plot_points: int = 8000,
) -> Image.Image:
    """Render an annotated 2-lead ECG plot and return a PIL Image.

    Coordinates (``rhythm_bands``, ``beat_markers``, ``highlights``) are
    all **window-relative sample indices**.
    """
    if signals.ndim != 2:
        raise ValueError(f"signals must be (N, leads); got shape {signals.shape}")
    n_samples, n_channels = signals.shape
    if len(channel_names) != n_channels:
        channel_names = [f"lead_{i+1}" for i in range(n_channels)]

    fig, axes = plt.subplots(
        n_channels + 1,  # +1 for the beat strip above
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": [0.35] + [1.0] * n_channels},
    )
    if n_channels + 1 == 2:
        # matplotlib returns a single Axes when n==1; normalize to a sequence.
        axes = list(axes)

    beat_ax = axes[0]
    lead_axes = axes[1:]

    duration_seconds = n_samples / max(source_hz, 1)

    # Beat strip (symbol markers)
    beat_ax.set_xlim(0, 1)
    beat_ax.set_ylim(-0.5, max(1.0, len(BEAT_COLOURS) - 0.5))
    beat_ax.set_yticks(range(len(BEAT_COLOURS)))
    beat_ax.set_yticklabels(list(BEAT_COLOURS.keys()), fontsize=7)
    beat_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in beat_ax.spines.values():
        spine.set_visible(False)
    if beat_markers:
        y_of = {sym: i for i, sym in enumerate(BEAT_COLOURS.keys())}
        for symbol, samples in beat_markers.items():
            if not samples:
                continue
            xs = np.asarray(samples, dtype=np.float64) / max(1, n_samples - 1)
            ys = np.full_like(xs, y_of.get(symbol, 0), dtype=np.float64)
            beat_ax.scatter(
                xs, ys, s=6, c=BEAT_COLOURS.get(symbol, "#000000"), marker="|"
            )

    # Lead plots
    for ch_idx, (ax, name) in enumerate(zip(lead_axes, channel_names)):
        series, factor = downsample_for_plotting(signals[:, ch_idx], max_plot_points)
        x = np.linspace(0, 1, series.shape[0])
        ax.plot(x, series, color="#111111", linewidth=0.55, antialiased=True)
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

        # Rhythm bands
        if rhythm_bands:
            for bout in rhythm_bands:
                start_frac = bout.start_sample / max(1, n_samples - 1)
                end_frac = bout.end_sample / max(1, n_samples - 1)
                ax.axvspan(
                    start_frac,
                    end_frac,
                    color=_rhythm_colour(bout.activity),
                    alpha=0.18,
                )

        # Answer / context highlights — bold outline, colour-coded by label.
        if highlights:
            for start_s, end_s, label in highlights:
                s_frac = start_s / max(1, n_samples - 1)
                e_frac = end_s / max(1, n_samples - 1)
                face, edge = HIGHLIGHT_COLOURS.get(
                    label, HIGHLIGHT_COLOURS["answer"]
                )
                ax.axvspan(
                    s_frac,
                    e_frac,
                    facecolor=face,
                    alpha=0.22,
                    edgecolor=edge,
                    linewidth=1.8,
                )

    # Final axis labels on the bottom lead
    lead_axes[-1].set_xlabel(
        f"time (normalized across {duration_seconds:.1f} s window)", fontsize=8
    )

    # Title + Q/A footer
    header = title or ""
    if header:
        fig.suptitle(header, fontsize=10, y=0.995)
    if question is not None or answer is not None:
        q = (question or "").strip()
        a = (answer or "").strip()
        text = f"Q: {q}\nA: {a}"
        fig.text(
            0.02, 0.005, text,
            fontsize=8, ha="left", va="bottom", family="monospace",
            wrap=True,
        )
        plt.subplots_adjust(bottom=0.14)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def beats_by_symbol(
    beats: Sequence[LTAFBeatEvent],
) -> Dict[str, List[int]]:
    """Convenience adapter: regroup a beats_timeline into ``{symbol: [samples]}``."""
    out: Dict[str, List[int]] = {s: [] for s in BEAT_COLOURS}
    for b in beats:
        out.setdefault(b.symbol, []).append(int(b.sample))
    return out


__all__ = [
    "create_ecg_plot",
    "beats_by_symbol",
    "downsample_for_plotting",
    "RHYTHM_COLOURS",
    "BEAT_COLOURS",
    "HIGHLIGHT_COLOURS",
]
