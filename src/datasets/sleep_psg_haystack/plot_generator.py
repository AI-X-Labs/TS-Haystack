# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Plot generation for Sleep PSG TS-Haystack visual verification.

Creates multi-channel PSG visualizations with optional needle region
annotations for verifying sample generation quality.

Key Features:
- 13-channel PSG plot (one subplot per channel)
- Optional needle region annotations (shaded areas + labels)
- Downsampling for long time series (> 5000 samples)
- Channel grouping by signal type (EEG, EOG, EMG, Respiratory, Other)
"""

import json
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.datasets.sleep_psg_haystack.loader import CHANNEL_NAMES

# Channel grouping and colors for visual clarity
CHANNEL_GROUPS = {
    "EEG": {"channels": ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"], "color": "#1f77b4"},
    "EOG": {"channels": ["E1-M2"], "color": "#9467bd"},
    "EMG": {"channels": ["Chin1-Chin2"], "color": "#d62728"},
    "Respiratory": {"channels": ["ABD", "CHEST", "AIRFLOW"], "color": "#2ca02c"},
    "SpO2": {"channels": ["SaO2"], "color": "#ff7f0e"},
    "ECG": {"channels": ["ECG"], "color": "#8c564b"},
}


def _get_channel_color(channel_name: str) -> str:
    for group_info in CHANNEL_GROUPS.values():
        if channel_name in group_info["channels"]:
            return group_info["color"]
    return "#333333"


def downsample_for_plotting(data: np.ndarray, max_points: int = 5000) -> Tuple[np.ndarray, int]:
    """Downsample via decimation for efficient plotting."""
    n = len(data)
    if n <= max_points:
        return data, 1
    factor = n // max_points
    return data[::factor], factor


def create_psg_plot(
    signals: np.ndarray,
    channel_names: List[str],
    time_range: Tuple[str, str],
    needles: Optional[List[Dict]] = None,
    figsize: Tuple[int, int] = (16, 20),
    dpi: int = 100,
    annotate_needles: bool = True,
    max_plot_points: int = 5000,
    source_hz: int = 200,
    title: Optional[str] = None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
) -> "Image.Image":
    """
    Create a multi-channel PSG plot with optional needle annotations.

    Args:
        signals: Shape (N_samples, N_channels)
        channel_names: List of channel names
        time_range: (start_time, end_time) strings
        needles: List of needle metadata dicts
        figsize: Figure size
        dpi: Resolution
        annotate_needles: If True, shade needle regions
        max_plot_points: Max points to plot (longer series downsampled)
        source_hz: Sampling frequency
        title: Optional custom title
        question: Optional Q/A text to display
        answer: Optional answer text to display

    Returns:
        PIL Image of the plot
    """
    n_channels = signals.shape[1]
    n_samples = signals.shape[0]

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    time_axis_full = np.linspace(0, 1, n_samples)
    duration_seconds = n_samples / source_hz

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ch_data = signals[:, ch_idx]
        ch_plot, factor = downsample_for_plotting(ch_data, max_plot_points)
        time_axis = np.linspace(0, 1, len(ch_plot))
        color = _get_channel_color(ch_name)

        ax.plot(time_axis, ch_plot, color=color, linewidth=0.4, alpha=0.85)
        ax.set_ylabel(ch_name, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, alpha=0.2)

        # Annotate needle regions with distinct colors per region
        if annotate_needles and needles:
            region_colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
            for ni, needle in enumerate(needles):
                start_frac = needle.get("insert_position_frac", 0)
                dur_samples = needle.get("duration_samples", 0)
                end_frac = start_frac + (dur_samples / n_samples)
                rc = region_colors[ni % len(region_colors)]

                ax.axvspan(start_frac, end_frac, alpha=0.15, color=rc,
                           label=needle.get("activity") if ch_idx == 0 else None)
                ax.axvline(start_frac, color=rc, linestyle="--", alpha=0.5, linewidth=0.8)
                ax.axvline(end_frac, color=rc, linestyle="--", alpha=0.5, linewidth=0.8)

    # X-axis formatting on bottom subplot
    axes[-1].set_xlabel(f"Time ({time_range[0]} to {time_range[1]})", fontsize=9)
    tick_pos = [0, 0.25, 0.5, 0.75, 1.0]
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Title
    if duration_seconds >= 3600:
        dur_str = f"{duration_seconds/3600:.1f} hours"
    elif duration_seconds >= 60:
        dur_str = f"{duration_seconds/60:.1f} minutes"
    else:
        dur_str = f"{duration_seconds:.0f} seconds"

    title_text = title or f"PSG Recording ({dur_str}, {n_samples:,} samples, {source_hz}Hz)"
    axes[0].set_title(title_text, fontsize=10)

    # Legend for annotated regions
    if annotate_needles and needles:
        from matplotlib.patches import Patch
        region_colors = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
        handles = []
        for ni, needle in enumerate(needles):
            rc = region_colors[ni % len(region_colors)]
            label = needle.get("activity", "unknown")
            handles.append(Patch(facecolor=rc, alpha=0.3, label=label))
        if handles:
            axes[0].legend(handles=handles, loc="upper right", fontsize=7)

    # Q/A annotation at the bottom
    if question or answer:
        qa_text = ""
        if question:
            qa_text += f"Q: {question}"
        if answer:
            qa_text += f"\nA: {answer}"
        fig.text(0.5, -0.01, qa_text, ha="center", fontsize=8,
                 style="italic", wrap=True,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    return img


def create_psg_plot_from_sample(
    sample: Dict,
    annotate_needles: bool = True,
    **kwargs,
) -> "Image.Image":
    """
    Create PSG plot from a generated sample dict (as loaded from parquet).

    Extracts per-channel data columns and needle metadata.
    """
    needles = sample.get("needles", "[]")
    if isinstance(needles, str):
        needles = json.loads(needles)

    time_range = (
        sample.get("recording_time_start", "?"),
        sample.get("recording_time_end", "?"),
    )

    # Reconstruct signals array from per-channel columns
    channel_names = sample.get("channel_names", CHANNEL_NAMES)
    if isinstance(channel_names, str):
        channel_names = json.loads(channel_names)

    channels = []
    for name in channel_names:
        if name in sample:
            channels.append(np.array(sample[name]))
    signals = np.column_stack(channels) if channels else np.zeros((100, len(channel_names)))

    return create_psg_plot(
        signals=signals,
        channel_names=channel_names,
        time_range=time_range,
        needles=needles if annotate_needles else None,
        annotate_needles=annotate_needles,
        question=sample.get("question"),
        answer=sample.get("answer"),
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PSG Plot Generator Test")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 60000  # 5 minutes at 200Hz
    n_channels = 13

    # Synthetic PSG-like signals
    signals = np.random.randn(n_samples, n_channels).astype(np.float32) * 10

    # Simulate a "REM" needle with distinct EOG pattern (channel 6)
    needle_start = 20000
    needle_end = 30000
    signals[needle_start:needle_end, 6] += np.sin(
        np.linspace(0, 100 * np.pi, needle_end - needle_start)
    ) * 30  # Rapid eye movements

    needles = [{
        "activity": "REM",
        "insert_position_frac": needle_start / n_samples,
        "duration_samples": needle_end - needle_start,
        "timestamp_start": "0:01:40",
        "timestamp_end": "0:02:30",
    }]

    img = create_psg_plot(
        signals=signals,
        channel_names=CHANNEL_NAMES,
        time_range=("0:00:00", "0:05:00"),
        needles=needles,
        question="Is there REM sleep in this recording?",
        answer="Yes.",
    )

    test_path = "/tmp/test_psg_plot.png"
    img.save(test_path)
    print(f"Saved test plot to: {test_path}")
    print(f"Image size: {img.size}")
    print("=" * 60)
