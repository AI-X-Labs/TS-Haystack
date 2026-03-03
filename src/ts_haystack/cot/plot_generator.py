# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Plot generation for TS-Haystack CoT generation.

This module creates accelerometer visualizations that are passed to the LLM
along with rich metadata for generating grounded chain-of-thought rationales.

Key Features:
- 3-axis accelerometer plot (X, Y, Z subplots)
- Optional needle region annotations (shaded areas)
- Downsampling for long time series (> 10000 samples)
- Clean, publication-quality formatting
"""

import json
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


def downsample_for_plotting(
    data: Union[List[float], np.ndarray],
    max_points: int = 5000,
) -> Tuple[np.ndarray, int]:
    """
    Downsample data for efficient plotting.

    Uses simple decimation (every nth point) for speed.

    Args:
        data: Input data array
        max_points: Maximum points to keep for plotting

    Returns:
        Tuple of (downsampled data, downsample factor)
    """
    data = np.array(data)
    n_points = len(data)

    if n_points <= max_points:
        return data, 1

    factor = n_points // max_points
    return data[::factor], factor


def create_accelerometer_plot(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    z: Union[List[float], np.ndarray],
    time_range: Tuple[str, str],
    needles: Optional[List[Dict]] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    annotate_needles: bool = True,
    max_plot_points: int = 5000,
    source_hz: int = 100,
) -> Image.Image:
    """
    Create a 3-axis accelerometer plot with optional needle annotations.

    The plot is designed to be passed to an LLM for visual analysis.
    Long time series are automatically downsampled for efficient rendering.

    Args:
        x: X-axis accelerometer data
        y: Y-axis accelerometer data
        z: Z-axis accelerometer data
        time_range: Tuple of (start_time, end_time) as strings (e.g., ("6:00 AM", "8:00 AM"))
        needles: List of needle metadata dicts with keys:
                 - activity: str
                 - insert_position_frac: float (0-1)
                 - duration_samples: int
                 - timestamp_start: str
                 - timestamp_end: str
        figsize: Figure size as (width, height) tuple
        dpi: Resolution for the output image
        annotate_needles: If True, shade needle regions and add labels
        max_plot_points: Maximum points to plot (longer series are downsampled)
        source_hz: Source sampling frequency (for time axis calculation)

    Returns:
        PIL Image containing the plot
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Convert to numpy arrays
    x_data = np.array(x)
    y_data = np.array(y)
    z_data = np.array(z)
    n_samples = len(x_data)

    # Downsample if necessary
    x_plot, factor = downsample_for_plotting(x_data, max_plot_points)
    y_plot, _ = downsample_for_plotting(y_data, max_plot_points)
    z_plot, _ = downsample_for_plotting(z_data, max_plot_points)

    # Create time axis (normalized 0-1 for display)
    time_axis = np.linspace(0, 1, len(x_plot))

    # Calculate actual duration for title
    duration_seconds = n_samples / source_hz

    # Plot each axis
    data_series = [
        ('X-axis', x_plot, '#1f77b4'),  # Blue
        ('Y-axis', y_plot, '#ff7f0e'),  # Orange
        ('Z-axis', z_plot, '#2ca02c'),  # Green
    ]

    for ax, (label, data, color) in zip(axes, data_series):
        ax.plot(time_axis, data, color=color, linewidth=0.5, alpha=0.9)
        ax.set_ylabel(f'{label} (g)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate needle regions
        if annotate_needles and needles:
            for needle in needles:
                start_frac = needle.get("insert_position_frac", 0)
                duration_samples = needle.get("duration_samples", 0)
                end_frac = start_frac + (duration_samples / n_samples)

                # Shade the needle region
                ax.axvspan(
                    start_frac, end_frac,
                    alpha=0.2, color='red',
                    label=needle.get("activity", "activity") if ax == axes[0] else None
                )

                # Add vertical lines at boundaries
                ax.axvline(start_frac, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.axvline(end_frac, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    # Format x-axis labels
    axes[-1].set_xlabel(f"Time ({time_range[0]} to {time_range[1]})", fontsize=10)

    # Set x-axis ticks
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = ['0%', '25%', '50%', '75%', '100%']
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels)

    # Title with duration info
    if duration_seconds >= 60:
        duration_str = f"{duration_seconds/60:.1f} minutes"
    else:
        duration_str = f"{duration_seconds:.0f} seconds"
    axes[0].set_title(
        f"Accelerometer Data ({duration_str}, {n_samples:,} samples)",
        fontsize=11
    )

    # Add legend if needles are annotated
    if annotate_needles and needles:
        # Collect unique activities for legend
        activities = list(set(n.get("activity", "unknown") for n in needles))
        if activities:
            axes[0].legend(
                [f"Inserted: {', '.join(activities)}"],
                loc='upper right',
                fontsize=8
            )

    plt.tight_layout()

    # Convert to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()  # Copy to allow buffer to be closed
    buf.close()
    plt.close(fig)

    return img


def create_accelerometer_plot_from_sample(
    sample: Dict,
    annotate_needles: bool = True,
    **kwargs,
) -> Image.Image:
    """
    Create accelerometer plot from a TS-Haystack sample dict.

    Convenience wrapper that extracts data from the sample format.

    Args:
        sample: Dict with keys: x_axis, y_axis, z_axis, recording_time_start,
                recording_time_end, needles (JSON string)
        annotate_needles: If True, shade needle regions
        **kwargs: Additional arguments passed to create_accelerometer_plot

    Returns:
        PIL Image containing the plot
    """
    # Parse needles from JSON string if needed
    needles = sample.get("needles", "[]")
    if isinstance(needles, str):
        needles = json.loads(needles)

    # Get time range
    time_range = (
        sample.get("recording_time_start", "unknown"),
        sample.get("recording_time_end", "unknown"),
    )

    return create_accelerometer_plot(
        x=sample["x_axis"],
        y=sample["y_axis"],
        z=sample["z_axis"],
        time_range=time_range,
        needles=needles if annotate_needles else None,
        annotate_needles=annotate_needles,
        **kwargs,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Plot Generator Test")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 10000  # 100 seconds at 100Hz

    # Simulate sedentary + walking bout
    x = np.random.randn(n_samples) * 0.1
    y = np.random.randn(n_samples) * 0.1
    z = np.random.randn(n_samples) * 0.1 - 1.0  # Gravity

    # Add a "walking" pattern in the middle
    walk_start = 4000
    walk_end = 6000
    x[walk_start:walk_end] += np.sin(np.linspace(0, 40*np.pi, walk_end - walk_start)) * 0.5
    y[walk_start:walk_end] += np.sin(np.linspace(0, 40*np.pi, walk_end - walk_start) + np.pi/2) * 0.3
    z[walk_start:walk_end] += np.sin(np.linspace(0, 80*np.pi, walk_end - walk_start)) * 0.2

    # Create needle metadata
    needles = [
        {
            "activity": "walking",
            "insert_position_frac": walk_start / n_samples,
            "duration_samples": walk_end - walk_start,
            "timestamp_start": "6:40 AM",
            "timestamp_end": "7:00 AM",
        }
    ]

    print("\nCreating plot with needle annotations...")
    img = create_accelerometer_plot(
        x=x.tolist(),
        y=y.tolist(),
        z=z.tolist(),
        time_range=("6:00 AM", "7:40 AM"),
        needles=needles,
        annotate_needles=True,
    )

    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")

    # Save test image
    test_path = "/tmp/test_accelerometer_plot.png"
    img.save(test_path)
    print(f"Saved test plot to: {test_path}")

    print("\n" + "=" * 60)
    print("Plot generator test complete!")
    print("=" * 60)
