# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for StyleTransfer module.

These tests use synthetic data and can run without Capture24 data.
Includes visualization outputs for visual assessment of style transfer coherence.
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core.data_structures import (
    NeedleSample,
    SignalStatistics,
)
from ts_haystack.core.style_transfer import StyleTransfer


# =============================================================================
# Visualization utilities
# =============================================================================

# Output directory for plots (same folder as tests)
PLOT_OUTPUT_DIR = Path(__file__).parent / "plots"


def ensure_plot_dir():
    """Create plot output directory if it doesn't exist."""
    PLOT_OUTPUT_DIR.mkdir(exist_ok=True)


def plot_style_transfer_comparison(
    background: tuple,
    needle: NeedleSample,
    transferred_needle: NeedleSample,
    final_signal: tuple,
    position: int,
    filename: str,
):
    """
    Generate comparison plots for visual assessment of style transfer.

    Creates a multi-panel figure showing:
    - Background signal (x, y, z)
    - Original needle (x, y, z)
    - Transferred needle (x, y, z)
    - Final signal with needle inserted (x, y, z)

    Args:
        background: Tuple of (x, y, z) background arrays
        needle: Original NeedleSample
        transferred_needle: Style-transferred NeedleSample
        final_signal: Tuple of (x, y, z) final arrays after insertion
        position: Insertion position in samples
        filename: Output filename (without extension)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    ensure_plot_dir()

    bg_x, bg_y, bg_z = background
    final_x, final_y, final_z = final_signal

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f"Style Transfer Comparison\n(insertion at sample {position})", fontsize=14)

    # Row 0: Background signal
    axes[0, 0].plot(bg_x, linewidth=0.5, color="blue", alpha=0.7)
    axes[0, 0].set_title("Background X")
    axes[0, 0].set_ylabel("Acceleration (g)")
    axes[0, 0].axvline(position, color="red", linestyle="--", alpha=0.5, label="Insert pos")
    axes[0, 0].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.5)

    axes[0, 1].plot(bg_y, linewidth=0.5, color="green", alpha=0.7)
    axes[0, 1].set_title("Background Y")
    axes[0, 1].axvline(position, color="red", linestyle="--", alpha=0.5)
    axes[0, 1].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.5)

    axes[0, 2].plot(bg_z, linewidth=0.5, color="orange", alpha=0.7)
    axes[0, 2].set_title("Background Z")
    axes[0, 2].axvline(position, color="red", linestyle="--", alpha=0.5)
    axes[0, 2].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.5)

    # Row 1: Original needle
    axes[1, 0].plot(needle.x, linewidth=0.5, color="blue")
    axes[1, 0].set_title(f"Original Needle X ({needle.activity})")
    axes[1, 0].set_ylabel("Acceleration (g)")

    axes[1, 1].plot(needle.y, linewidth=0.5, color="green")
    axes[1, 1].set_title(f"Original Needle Y ({needle.activity})")

    axes[1, 2].plot(needle.z, linewidth=0.5, color="orange")
    axes[1, 2].set_title(f"Original Needle Z ({needle.activity})")

    # Row 2: Transferred needle
    axes[2, 0].plot(transferred_needle.x, linewidth=0.5, color="blue")
    axes[2, 0].set_title("Transferred Needle X")
    axes[2, 0].set_ylabel("Acceleration (g)")

    axes[2, 1].plot(transferred_needle.y, linewidth=0.5, color="green")
    axes[2, 1].set_title("Transferred Needle Y")

    axes[2, 2].plot(transferred_needle.z, linewidth=0.5, color="orange")
    axes[2, 2].set_title("Transferred Needle Z")

    # Row 3: Final signal (zoomed to insertion region)
    margin = 200
    start_idx = max(0, position - margin)
    end_idx = min(len(final_x), position + needle.n_samples + margin)

    axes[3, 0].plot(range(start_idx, end_idx), final_x[start_idx:end_idx], linewidth=0.5, color="blue")
    axes[3, 0].axvline(position, color="red", linestyle="--", alpha=0.7, label="Needle start")
    axes[3, 0].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.7, label="Needle end")
    axes[3, 0].axvspan(position, position + needle.n_samples, alpha=0.1, color="red")
    axes[3, 0].set_title("Final Signal X (zoomed)")
    axes[3, 0].set_ylabel("Acceleration (g)")
    axes[3, 0].set_xlabel("Sample index")
    axes[3, 0].legend(fontsize=8)

    axes[3, 1].plot(range(start_idx, end_idx), final_y[start_idx:end_idx], linewidth=0.5, color="green")
    axes[3, 1].axvline(position, color="red", linestyle="--", alpha=0.7)
    axes[3, 1].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.7)
    axes[3, 1].axvspan(position, position + needle.n_samples, alpha=0.1, color="red")
    axes[3, 1].set_title("Final Signal Y (zoomed)")
    axes[3, 1].set_xlabel("Sample index")

    axes[3, 2].plot(range(start_idx, end_idx), final_z[start_idx:end_idx], linewidth=0.5, color="orange")
    axes[3, 2].axvline(position, color="red", linestyle="--", alpha=0.7)
    axes[3, 2].axvline(position + needle.n_samples, color="red", linestyle="--", alpha=0.7)
    axes[3, 2].axvspan(position, position + needle.n_samples, alpha=0.1, color="red")
    axes[3, 2].set_title("Final Signal Z (zoomed)")
    axes[3, 2].set_xlabel("Sample index")

    plt.tight_layout()
    output_path = PLOT_OUTPUT_DIR / f"{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_blending_comparison(
    background: tuple,
    needle: tuple,
    result_linear: tuple,
    result_cosine: tuple,
    position: int,
    filename: str,
):
    """
    Generate comparison of linear vs cosine blending.

    Args:
        background: Tuple of (x, y, z) background arrays
        needle: Tuple of (x, y, z) needle arrays
        result_linear: Result with linear blending
        result_cosine: Result with cosine blending
        position: Insertion position
        filename: Output filename
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    ensure_plot_dir()

    needle_len = len(needle[0])
    margin = 100
    start_idx = max(0, position - margin)
    end_idx = min(len(background[0]), position + needle_len + margin)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Linear vs Cosine Blending Comparison", fontsize=14)

    for i, (axis_name, color) in enumerate([("X", "blue"), ("Y", "green"), ("Z", "orange")]):
        # Linear blending
        axes[i, 0].plot(
            range(start_idx, end_idx),
            background[i][start_idx:end_idx],
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            label="Background",
        )
        axes[i, 0].plot(
            range(start_idx, end_idx),
            result_linear[i][start_idx:end_idx],
            linewidth=0.8,
            color=color,
            label="Result",
        )
        axes[i, 0].axvline(position, color="red", linestyle="--", alpha=0.5)
        axes[i, 0].axvline(position + needle_len, color="red", linestyle="--", alpha=0.5)
        axes[i, 0].axvspan(position, position + needle_len, alpha=0.1, color="red")
        axes[i, 0].set_title(f"{axis_name} - Linear Blending")
        axes[i, 0].set_ylabel("Acceleration (g)")
        if i == 0:
            axes[i, 0].legend(fontsize=8)

        # Cosine blending
        axes[i, 1].plot(
            range(start_idx, end_idx),
            background[i][start_idx:end_idx],
            linewidth=0.5,
            color="gray",
            alpha=0.5,
            label="Background",
        )
        axes[i, 1].plot(
            range(start_idx, end_idx),
            result_cosine[i][start_idx:end_idx],
            linewidth=0.8,
            color=color,
            label="Result",
        )
        axes[i, 1].axvline(position, color="red", linestyle="--", alpha=0.5)
        axes[i, 1].axvline(position + needle_len, color="red", linestyle="--", alpha=0.5)
        axes[i, 1].axvspan(position, position + needle_len, alpha=0.1, color="red")
        axes[i, 1].set_title(f"{axis_name} - Cosine Blending")
        if i == 0:
            axes[i, 1].legend(fontsize=8)

    axes[2, 0].set_xlabel("Sample index")
    axes[2, 1].set_xlabel("Sample index")

    plt.tight_layout()
    output_path = PLOT_OUTPUT_DIR / f"{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_statistics_comparison(
    needle: NeedleSample,
    transferred: NeedleSample,
    needle_stats: SignalStatistics,
    target_stats: SignalStatistics,
    transferred_stats: SignalStatistics,
    filename: str,
):
    """
    Generate statistics comparison plots.

    Shows histograms and statistics comparison between original needle,
    target, and transferred signal.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    ensure_plot_dir()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Statistics Comparison: Original vs Transferred", fontsize=14)

    axes_labels = ["X", "Y", "Z"]
    colors = ["blue", "green", "orange"]

    for i, (ax_label, color) in enumerate(zip(axes_labels, colors)):
        # Histograms
        original_data = [needle.x, needle.y, needle.z][i]
        transferred_data = [transferred.x, transferred.y, transferred.z][i]

        axes[0, i].hist(original_data, bins=50, alpha=0.5, color="gray", label="Original", density=True)
        axes[0, i].hist(transferred_data, bins=50, alpha=0.5, color=color, label="Transferred", density=True)
        axes[0, i].axvline(needle_stats.mean[i], color="gray", linestyle="--", linewidth=2, label=f"Orig \u03bc={needle_stats.mean[i]:.2f}")
        axes[0, i].axvline(target_stats.mean[i], color="red", linestyle="-", linewidth=2, label=f"Target \u03bc={target_stats.mean[i]:.2f}")
        axes[0, i].axvline(transferred_stats.mean[i], color=color, linestyle=":", linewidth=2, label=f"Trans \u03bc={transferred_stats.mean[i]:.2f}")
        axes[0, i].set_title(f"{ax_label} Distribution")
        axes[0, i].set_xlabel("Acceleration (g)")
        axes[0, i].set_ylabel("Density")
        axes[0, i].legend(fontsize=7)

    # Bar chart of statistics comparison
    x_pos = np.arange(3)
    width = 0.25

    # Means
    axes[1, 0].bar(x_pos - width, needle_stats.mean, width, label="Original", color="gray", alpha=0.7)
    axes[1, 0].bar(x_pos, target_stats.mean, width, label="Target", color="red", alpha=0.7)
    axes[1, 0].bar(x_pos + width, transferred_stats.mean, width, label="Transferred", color="blue", alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(["X", "Y", "Z"])
    axes[1, 0].set_title("Mean Comparison")
    axes[1, 0].set_ylabel("Mean (g)")
    axes[1, 0].legend()

    # Stds
    axes[1, 1].bar(x_pos - width, needle_stats.std, width, label="Original", color="gray", alpha=0.7)
    axes[1, 1].bar(x_pos, target_stats.std, width, label="Target", color="red", alpha=0.7)
    axes[1, 1].bar(x_pos + width, transferred_stats.std, width, label="Transferred", color="blue", alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(["X", "Y", "Z"])
    axes[1, 1].set_title("Std Dev Comparison")
    axes[1, 1].set_ylabel("Std Dev (g)")
    axes[1, 1].legend()

    # Text summary
    axes[1, 2].axis("off")
    summary_text = (
        "Statistics Summary:\n\n"
        f"Original Needle:\n"
        f"  Mean: [{needle_stats.mean[0]:.3f}, {needle_stats.mean[1]:.3f}, {needle_stats.mean[2]:.3f}]\n"
        f"  Std:  [{needle_stats.std[0]:.3f}, {needle_stats.std[1]:.3f}, {needle_stats.std[2]:.3f}]\n\n"
        f"Target Context:\n"
        f"  Mean: [{target_stats.mean[0]:.3f}, {target_stats.mean[1]:.3f}, {target_stats.mean[2]:.3f}]\n"
        f"  Std:  [{target_stats.std[0]:.3f}, {target_stats.std[1]:.3f}, {target_stats.std[2]:.3f}]\n\n"
        f"Transferred Needle:\n"
        f"  Mean: [{transferred_stats.mean[0]:.3f}, {transferred_stats.mean[1]:.3f}, {transferred_stats.mean[2]:.3f}]\n"
        f"  Std:  [{transferred_stats.std[0]:.3f}, {transferred_stats.std[1]:.3f}, {transferred_stats.std[2]:.3f}]"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family="monospace", verticalalignment="center")

    plt.tight_layout()
    output_path = PLOT_OUTPUT_DIR / f"{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def style_transfer_linear():
    """StyleTransfer with linear blending."""
    return StyleTransfer(blend_mode="linear", blend_window_samples=50)


@pytest.fixture
def style_transfer_cosine():
    """StyleTransfer with cosine blending."""
    return StyleTransfer(blend_mode="cosine", blend_window_samples=50)


@pytest.fixture
def sample_needle():
    """Create a synthetic needle sample simulating walking activity."""
    rng = np.random.default_rng(42)
    n_samples = 500

    # Simulate walking-like pattern with some periodicity
    t = np.linspace(0, 5, n_samples)
    x = (0.5 + 0.3 * np.sin(2 * np.pi * 2 * t) + rng.normal(0, 0.05, n_samples)).astype(np.float32)
    y = (-0.2 + 0.2 * np.sin(2 * np.pi * 2 * t + np.pi / 4) + rng.normal(0, 0.05, n_samples)).astype(np.float32)
    z = (0.8 + 0.4 * np.sin(2 * np.pi * 2 * t + np.pi / 2) + rng.normal(0, 0.08, n_samples)).astype(np.float32)

    return NeedleSample(
        source_pid="P001",
        activity="walking",
        start_ms=1000000,
        end_ms=1005000,
        duration_ms=5000,
        x=x,
        y=y,
        z=z,
    )


@pytest.fixture
def sample_background():
    """Create synthetic background arrays simulating sedentary activity."""
    rng = np.random.default_rng(123)
    n_samples = 10000

    # Simulate sedentary/sitting - low variance, near gravity vector
    x = rng.normal(0.0, 0.02, n_samples).astype(np.float32)
    y = rng.normal(0.0, 0.02, n_samples).astype(np.float32)
    z = rng.normal(-1.0, 0.02, n_samples).astype(np.float32)  # Gravity-aligned

    return (x, y, z)


# =============================================================================
# Test SignalStatistics computation
# =============================================================================


class TestComputeStatistics:
    """Tests for StyleTransfer.compute_statistics()."""

    def test_basic_statistics(self, style_transfer_linear):
        """Test basic mean and std computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        stats = style_transfer_linear.compute_statistics(x, y, z)

        assert isinstance(stats, SignalStatistics)
        assert stats.mean.shape == (3,)
        assert stats.std.shape == (3,)
        assert stats.cov.shape == (3, 3)
        assert stats.cholesky.shape == (3, 3)

        np.testing.assert_almost_equal(stats.mean[0], 3.0)
        np.testing.assert_almost_equal(stats.mean[1], 6.0)
        np.testing.assert_almost_equal(stats.mean[2], 0.0)

    def test_handles_zero_std(self, style_transfer_linear):
        """Test that zero std is handled (constant signal)."""
        x = np.ones(100, dtype=np.float32)
        y = np.ones(100, dtype=np.float32) * 2
        z = np.ones(100, dtype=np.float32) * 3

        stats = style_transfer_linear.compute_statistics(x, y, z)

        assert np.all(stats.std >= 1e-8)

    def test_cholesky_decomposition(self, style_transfer_linear):
        """Test that Cholesky decomposition is valid."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000).astype(np.float32)
        y = rng.normal(0, 1, 1000).astype(np.float32)
        z = rng.normal(0, 1, 1000).astype(np.float32)

        stats = style_transfer_linear.compute_statistics(x, y, z)

        reconstructed = stats.cholesky @ stats.cholesky.T
        np.testing.assert_array_almost_equal(reconstructed, stats.cov, decimal=5)

    def test_correlated_signals(self, style_transfer_linear):
        """Test statistics with correlated signals."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n).astype(np.float32)
        y = (x * 0.8 + rng.normal(0, 0.5, n)).astype(np.float32)
        z = rng.normal(-1, 0.5, n).astype(np.float32)

        stats = style_transfer_linear.compute_statistics(x, y, z)

        assert stats.cov[0, 1] > 0


# =============================================================================
# Test Style Transfer
# =============================================================================


class TestTransfer:
    """Tests for StyleTransfer.transfer()."""

    def test_transfer_preserves_shape(self, style_transfer_linear, sample_needle):
        """Test that transfer preserves array shapes."""
        target_stats = SignalStatistics(
            mean=np.array([0.0, 0.0, -1.0]),
            std=np.array([0.1, 0.1, 0.1]),
            cov=np.eye(3) * 0.01,
            cholesky=np.eye(3) * 0.1,
        )

        transferred = style_transfer_linear.transfer(sample_needle, target_stats)

        assert transferred.x.shape == sample_needle.x.shape
        assert transferred.y.shape == sample_needle.y.shape
        assert transferred.z.shape == sample_needle.z.shape

    def test_transfer_changes_statistics(
        self, style_transfer_linear, sample_needle, sample_background
    ):
        """Test that transferred signal has target-like statistics."""
        target_stats = style_transfer_linear.compute_statistics(*sample_background)
        transferred = style_transfer_linear.transfer(sample_needle, target_stats)

        result_stats = style_transfer_linear.compute_statistics(
            transferred.x, transferred.y, transferred.z
        )

        np.testing.assert_array_almost_equal(
            result_stats.mean, target_stats.mean, decimal=1
        )

    def test_transfer_preserves_metadata(self, style_transfer_linear, sample_needle):
        """Test that transfer preserves needle metadata."""
        target_stats = SignalStatistics(
            mean=np.array([0.0, 0.0, -1.0]),
            std=np.array([0.1, 0.1, 0.1]),
            cov=np.eye(3) * 0.01,
            cholesky=np.eye(3) * 0.1,
        )

        transferred = style_transfer_linear.transfer(sample_needle, target_stats)

        assert transferred.source_pid == sample_needle.source_pid
        assert transferred.activity == sample_needle.activity
        assert transferred.start_ms == sample_needle.start_ms
        assert transferred.end_ms == sample_needle.end_ms
        assert transferred.duration_ms == sample_needle.duration_ms

    def test_transfer_dtype_float32(self, style_transfer_linear, sample_needle):
        """Test that transferred arrays are float32."""
        target_stats = SignalStatistics(
            mean=np.array([0.0, 0.0, -1.0]),
            std=np.array([0.1, 0.1, 0.1]),
            cov=np.eye(3) * 0.01,
            cholesky=np.eye(3) * 0.1,
        )

        transferred = style_transfer_linear.transfer(sample_needle, target_stats)

        assert transferred.x.dtype == np.float32
        assert transferred.y.dtype == np.float32
        assert transferred.z.dtype == np.float32


# =============================================================================
# Test Blending Weights
# =============================================================================


class TestBlendWeights:
    """Tests for blend weight generation."""

    def test_linear_blend_weights(self, style_transfer_linear):
        """Test linear blend weights."""
        weights = style_transfer_linear._get_blend_weights(10)

        assert len(weights) == 10
        assert weights[0] == 0.0
        assert weights[-1] == 1.0
        np.testing.assert_array_almost_equal(np.diff(weights), np.diff(weights)[0])

    def test_cosine_blend_weights(self, style_transfer_cosine):
        """Test cosine blend weights."""
        weights = style_transfer_cosine._get_blend_weights(10)

        assert len(weights) == 10
        np.testing.assert_almost_equal(weights[0], 0.0, decimal=5)
        np.testing.assert_almost_equal(weights[-1], 1.0, decimal=5)
        assert weights[len(weights) // 2] >= 0.5

    def test_zero_length_blend(self, style_transfer_linear):
        """Test blend weights for zero length."""
        weights = style_transfer_linear._get_blend_weights(0)
        assert len(weights) == 0

    def test_single_sample_blend(self, style_transfer_linear):
        """Test blend weights for single sample."""
        weights = style_transfer_linear._get_blend_weights(1)
        assert len(weights) == 1
        assert weights[0] == 0.0


# =============================================================================
# Test Insert with Blending
# =============================================================================


class TestInsertWithBlending:
    """Tests for StyleTransfer.insert_with_blending()."""

    def test_basic_insertion(self, style_transfer_linear, sample_background):
        """Test basic needle insertion."""
        needle_len = 100
        needle = (
            np.ones(needle_len, dtype=np.float32) * 10,
            np.ones(needle_len, dtype=np.float32) * 20,
            np.ones(needle_len, dtype=np.float32) * 30,
        )
        position = 5000

        result = style_transfer_linear.insert_with_blending(
            sample_background, needle, position
        )

        assert len(result) == 3
        assert all(len(r) == len(sample_background[0]) for r in result)

    def test_insertion_modifies_correct_region(
        self, style_transfer_linear, sample_background
    ):
        """Test that insertion modifies the correct region."""
        bg_x, bg_y, bg_z = sample_background
        original_x = bg_x.copy()

        needle_len = 100
        needle = (
            np.ones(needle_len, dtype=np.float32) * 999,
            np.ones(needle_len, dtype=np.float32) * 999,
            np.ones(needle_len, dtype=np.float32) * 999,
        )
        position = 5000
        blend_len = style_transfer_linear.blend_window_samples

        result_x, _, _ = style_transfer_linear.insert_with_blending(
            sample_background, needle, position
        )

        np.testing.assert_array_equal(result_x[:position], original_x[:position])

        mid_start = position + blend_len
        mid_end = position + needle_len - blend_len
        if mid_start < mid_end:
            assert np.mean(result_x[mid_start:mid_end]) > 900

        end_pos = position + needle_len
        np.testing.assert_array_equal(result_x[end_pos:], original_x[end_pos:])

    def test_edge_insertion_start(self, style_transfer_linear):
        """Test insertion at the start of background."""
        bg = (
            np.zeros(1000, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
        )
        needle = (
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        )

        result = style_transfer_linear.insert_with_blending(bg, needle, position=0)

        assert len(result[0]) == 1000
        assert result[0][50] > 0

    def test_edge_insertion_end(self, style_transfer_linear):
        """Test insertion near the end of background."""
        bg = (
            np.zeros(1000, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
            np.zeros(1000, dtype=np.float32),
        )
        needle = (
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        )

        result = style_transfer_linear.insert_with_blending(bg, needle, position=950)

        assert len(result[0]) == 1000
        assert result[0][960] > 0

    def test_oversized_needle_trimmed(self, style_transfer_linear):
        """Test that needle larger than remaining space is trimmed."""
        bg_len = 500
        bg = (
            np.zeros(bg_len, dtype=np.float32),
            np.zeros(bg_len, dtype=np.float32),
            np.zeros(bg_len, dtype=np.float32),
        )
        needle = (
            np.ones(200, dtype=np.float32),
            np.ones(200, dtype=np.float32),
            np.ones(200, dtype=np.float32),
        )

        result = style_transfer_linear.insert_with_blending(bg, needle, position=400)

        assert len(result[0]) == bg_len
        assert result[0][450] > 0

    def test_output_dtype_float32(self, style_transfer_linear, sample_background):
        """Test that output is float32."""
        needle = (
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        )

        result = style_transfer_linear.insert_with_blending(
            sample_background, needle, position=5000
        )

        assert result[0].dtype == np.float32
        assert result[1].dtype == np.float32
        assert result[2].dtype == np.float32


# =============================================================================
# Test Local Statistics
# =============================================================================


class TestComputeLocalStatistics:
    """Tests for StyleTransfer.compute_local_statistics()."""

    def test_local_statistics_at_center(self, style_transfer_linear, sample_background):
        """Test computing local statistics at center of background."""
        stats = style_transfer_linear.compute_local_statistics(
            sample_background, position=5000, window_samples=500
        )

        assert isinstance(stats, SignalStatistics)
        assert stats.mean.shape == (3,)

    def test_local_statistics_at_edge_start(
        self, style_transfer_linear, sample_background
    ):
        """Test computing local statistics near start edge."""
        stats = style_transfer_linear.compute_local_statistics(
            sample_background, position=100, window_samples=500
        )

        assert isinstance(stats, SignalStatistics)

    def test_local_statistics_at_edge_end(
        self, style_transfer_linear, sample_background
    ):
        """Test computing local statistics near end edge."""
        stats = style_transfer_linear.compute_local_statistics(
            sample_background, position=9900, window_samples=500
        )

        assert isinstance(stats, SignalStatistics)

    def test_local_vs_global_statistics(self, style_transfer_linear):
        """Test that local statistics differ from global when appropriate."""
        n = 10000
        rng = np.random.default_rng(42)
        bg_x = np.zeros(n, dtype=np.float32)
        bg_x[:5000] = rng.normal(0, 0.1, 5000)
        bg_x[5000:] = rng.normal(5, 0.1, 5000)
        bg_y = np.zeros(n, dtype=np.float32)
        bg_z = np.zeros(n, dtype=np.float32)

        bg = (bg_x, bg_y, bg_z)

        stats_start = style_transfer_linear.compute_local_statistics(
            bg, position=2500, window_samples=500
        )
        stats_end = style_transfer_linear.compute_local_statistics(
            bg, position=7500, window_samples=500
        )

        assert abs(stats_start.mean[0] - stats_end.mean[0]) > 3


# =============================================================================
# Test Blend Mode Selection
# =============================================================================


class TestBlendModeSelection:
    """Tests for blend mode selection."""

    def test_invalid_blend_mode_raises(self):
        """Test that invalid blend mode raises error."""
        st = StyleTransfer(blend_mode="invalid")

        with pytest.raises(ValueError, match="Unknown blend mode"):
            st._get_blend_weights(10)

    def test_linear_mode_works(self):
        """Test linear mode initialization."""
        st = StyleTransfer(blend_mode="linear")
        weights = st._get_blend_weights(10)
        assert len(weights) == 10

    def test_cosine_mode_works(self):
        """Test cosine mode initialization."""
        st = StyleTransfer(blend_mode="cosine")
        weights = st._get_blend_weights(10)
        assert len(weights) == 10


# =============================================================================
# Test Determinism
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_compute_statistics_deterministic(self, style_transfer_linear):
        """Test that compute_statistics is deterministic."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000).astype(np.float32)
        y = rng.normal(0, 1, 1000).astype(np.float32)
        z = rng.normal(0, 1, 1000).astype(np.float32)

        stats1 = style_transfer_linear.compute_statistics(x, y, z)
        stats2 = style_transfer_linear.compute_statistics(x, y, z)

        np.testing.assert_array_equal(stats1.mean, stats2.mean)
        np.testing.assert_array_equal(stats1.std, stats2.std)
        np.testing.assert_array_equal(stats1.cov, stats2.cov)
        np.testing.assert_array_equal(stats1.cholesky, stats2.cholesky)

    def test_transfer_deterministic(self, style_transfer_linear, sample_needle):
        """Test that transfer is deterministic."""
        target_stats = SignalStatistics(
            mean=np.array([0.0, 0.0, -1.0]),
            std=np.array([0.1, 0.1, 0.1]),
            cov=np.eye(3) * 0.01,
            cholesky=np.eye(3) * 0.1,
        )

        transferred1 = style_transfer_linear.transfer(sample_needle, target_stats)
        transferred2 = style_transfer_linear.transfer(sample_needle, target_stats)

        np.testing.assert_array_equal(transferred1.x, transferred2.x)
        np.testing.assert_array_equal(transferred1.y, transferred2.y)
        np.testing.assert_array_equal(transferred1.z, transferred2.z)

    def test_insertion_deterministic(self, style_transfer_linear, sample_background):
        """Test that insertion is deterministic."""
        needle = (
            np.ones(100, dtype=np.float32) * 999,
            np.ones(100, dtype=np.float32) * 999,
            np.ones(100, dtype=np.float32) * 999,
        )
        position = 5000

        result1 = style_transfer_linear.insert_with_blending(
            sample_background, needle, position
        )
        result2 = style_transfer_linear.insert_with_blending(
            sample_background, needle, position
        )

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        np.testing.assert_array_equal(result1[2], result2[2])


# =============================================================================
# Visualization Tests (generate plots for visual inspection)
# =============================================================================


class TestVisualization:
    """Tests that generate visualization plots for manual inspection."""

    def test_full_pipeline_with_plots(
        self, style_transfer_cosine, sample_needle, sample_background
    ):
        """
        Full style transfer pipeline test with visualization.

        Generates plots showing:
        - Background signal
        - Original needle
        - Transferred needle
        - Final signal with insertion
        """
        st = style_transfer_cosine
        position = 5000

        # Compute target statistics from background
        target_stats = st.compute_statistics(*sample_background)

        # Compute original needle statistics
        needle_stats = st.compute_statistics(
            sample_needle.x, sample_needle.y, sample_needle.z
        )

        # Apply style transfer
        transferred = st.transfer(sample_needle, target_stats)

        # Compute transferred statistics
        transferred_stats = st.compute_statistics(
            transferred.x, transferred.y, transferred.z
        )

        # Insert into background
        final_signal = st.insert_with_blending(
            sample_background,
            (transferred.x, transferred.y, transferred.z),
            position,
        )

        # Generate comparison plot
        plot_style_transfer_comparison(
            background=sample_background,
            needle=sample_needle,
            transferred_needle=transferred,
            final_signal=final_signal,
            position=position,
            filename="style_transfer_full_pipeline",
        )

        # Generate statistics comparison plot
        plot_statistics_comparison(
            needle=sample_needle,
            transferred=transferred,
            needle_stats=needle_stats,
            target_stats=target_stats,
            transferred_stats=transferred_stats,
            filename="style_transfer_statistics",
        )

        # Basic assertions
        assert final_signal[0].shape == sample_background[0].shape
        assert transferred.activity == sample_needle.activity

    def test_blending_comparison_with_plots(self, sample_background):
        """
        Compare linear vs cosine blending with visualization.
        """
        st_linear = StyleTransfer(blend_mode="linear", blend_window_samples=50)
        st_cosine = StyleTransfer(blend_mode="cosine", blend_window_samples=50)

        # Create a distinctive needle
        needle_len = 200
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2, needle_len)
        needle = (
            (np.sin(2 * np.pi * 3 * t) * 0.5 + rng.normal(0, 0.02, needle_len)).astype(np.float32),
            (np.cos(2 * np.pi * 3 * t) * 0.5 + rng.normal(0, 0.02, needle_len)).astype(np.float32),
            (np.sin(2 * np.pi * 3 * t + np.pi/4) * 0.5 - 0.5 + rng.normal(0, 0.02, needle_len)).astype(np.float32),
        )
        position = 5000

        result_linear = st_linear.insert_with_blending(sample_background, needle, position)
        result_cosine = st_cosine.insert_with_blending(sample_background, needle, position)

        # Generate comparison plot
        plot_blending_comparison(
            background=sample_background,
            needle=needle,
            result_linear=result_linear,
            result_cosine=result_cosine,
            position=position,
            filename="blending_comparison_linear_vs_cosine",
        )

        # Basic assertions
        assert result_linear[0].shape == sample_background[0].shape
        assert result_cosine[0].shape == sample_background[0].shape

    def test_different_activity_contrasts(self, style_transfer_cosine):
        """
        Test style transfer with high-contrast activities and visualize.

        Simulates inserting a vigorous activity (running) into a sedentary background.
        """
        rng = np.random.default_rng(42)

        # Sedentary background (very low variance)
        bg_len = 10000
        background = (
            rng.normal(0.05, 0.01, bg_len).astype(np.float32),
            rng.normal(0.02, 0.01, bg_len).astype(np.float32),
            rng.normal(-0.98, 0.01, bg_len).astype(np.float32),
        )

        # Vigorous activity needle (high variance, periodic)
        needle_len = 500
        t = np.linspace(0, 5, needle_len)
        needle = NeedleSample(
            source_pid="P002",
            activity="running",
            start_ms=2000000,
            end_ms=2005000,
            duration_ms=5000,
            x=(1.2 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.1, needle_len)).astype(np.float32),
            y=(0.8 * np.cos(2 * np.pi * 3 * t) + rng.normal(0, 0.1, needle_len)).astype(np.float32),
            z=(-0.3 + 0.5 * np.sin(2 * np.pi * 6 * t) + rng.normal(0, 0.1, needle_len)).astype(np.float32),
        )

        position = 5000
        st = style_transfer_cosine

        # Apply style transfer
        target_stats = st.compute_statistics(*background)
        needle_stats = st.compute_statistics(needle.x, needle.y, needle.z)
        transferred = st.transfer(needle, target_stats)
        transferred_stats = st.compute_statistics(transferred.x, transferred.y, transferred.z)

        # Insert
        final_signal = st.insert_with_blending(
            background,
            (transferred.x, transferred.y, transferred.z),
            position,
        )

        # Generate plots
        plot_style_transfer_comparison(
            background=background,
            needle=needle,
            transferred_needle=transferred,
            final_signal=final_signal,
            position=position,
            filename="high_contrast_running_in_sedentary",
        )

        plot_statistics_comparison(
            needle=needle,
            transferred=transferred,
            needle_stats=needle_stats,
            target_stats=target_stats,
            transferred_stats=transferred_stats,
            filename="high_contrast_statistics",
        )

        # Verify transfer worked
        assert final_signal[0].shape == background[0].shape
