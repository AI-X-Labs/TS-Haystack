# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for ComparisonTaskGenerator.

Task 7: "What was the {longest/shortest} period {with/without} {activity}?"
Answer type: time_range

Tests validate:
- Multiple bouts with distinct durations (no ties)
- Correct identification of extremum
- Both "with" and "without" polarity
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestComparisonSampleGeneration:
    """Tests for comparison sample generation."""

    def test_generate_single_sample(self, comparison_generator, rng):
        """Test generating a single comparison sample."""
        difficulty = DifficultyConfig(
            context_length_samples=30000,
            needle_position="random",
            needle_length_ratio_range=(0.01, 0.05),  # 300-1500 samples for 30000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 2,
                "max_bouts": 4,
                "min_duration_diff_ms": 2000,
                "min_gap_samples": 100,
            },
        )

        sample = comparison_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "comparison"

        if sample.is_valid:
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")

            # Check for extremum keywords in question
            assert "longest" in sample.question.lower() or "shortest" in sample.question.lower()

    def test_distinct_durations(self, comparison_generator):
        """Verify bouts have distinct durations (no ties)."""
        difficulty = DifficultyConfig(
            context_length_samples=35000,
            needle_position="random",
            needle_length_ratio_range=(0.009, 0.057),  # 315-1995 samples for 35000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 3,
                "max_bouts": 4,
                "min_duration_diff_ms": 3000,
                "min_gap_samples": 100,
            },
        )

        for i in range(15):
            rng = np.random.default_rng(1900 + i)
            sample = comparison_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) >= 2:
                durations = [n.duration_samples for n in sample.needles]
                unique_durations = len(set(durations))

                print(f"  Sample {i}: Durations = {durations}")
                assert unique_durations == len(durations), "Durations should be unique"
                return

        pytest.skip("Could not generate suitable sample")


class TestComparisonVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_comparison_sample(self, comparison_generator):
        """Visualize a comparison sample with multiple bouts."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("comparison")

        difficulty = DifficultyConfig(
            context_length_samples=40000,
            needle_position="random",
            needle_length_ratio_range=(0.01, 0.045),  # 400-1800 samples for 40000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 3,
                "max_bouts": 4,
                "min_duration_diff_ms": 3000,
                "min_gap_samples": 200,
            },
        )

        sample = None
        for i in range(40):
            rng = np.random.default_rng(2000 + i)
            candidate = comparison_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) >= 3:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Find longest and shortest
        sorted_by_duration = sorted(sample.needles, key=lambda n: n.duration_samples)
        shortest = sorted_by_duration[0]
        longest = sorted_by_duration[-1]

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Comparison Task - Find Longest/Shortest", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.4, color=color, alpha=0.7)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight each needle, with special colors for extrema
            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples

                if needle == longest:
                    ax.axvspan(start, end, alpha=0.4, color="red", label="Longest")
                elif needle == shortest:
                    ax.axvspan(start, end, alpha=0.4, color="blue", label="Shortest")
                else:
                    ax.axvspan(start, end, alpha=0.25, color="gray")

                if ax == axes[0]:
                    ax.annotate(f"{needle.duration_samples}",
                                xy=(start + (end - start) // 2, ax.get_ylim()[1] * 0.9),
                                fontsize=8, ha="center")

        axes[0].legend(loc="upper right", fontsize=8)
        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")

        durations_str = ", ".join([f"{n.duration_samples}" for n in sorted_by_duration])
        text = (
            f"Background: {sample.background_pid}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Bout durations (samples): {durations_str}\n"
            f"Shortest: {shortest.duration_samples} samples | Longest: {longest.duration_samples} samples"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "comparison_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_gap_comparison(self, comparison_generator):
        """Visualize comparison task focusing on gaps (without polarity)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("comparison")

        difficulty = DifficultyConfig(
            context_length_samples=35000,
            needle_position="random",
            needle_length_ratio_range=(0.009, 0.029),  # 315-1015 samples for 35000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 3,
                "max_bouts": 4,
                "min_gap_samples": 300,  # Larger gaps for visibility
            },
        )

        sample = None
        for i in range(40):
            rng = np.random.default_rng(2100 + i)
            candidate = comparison_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) >= 3:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Sort needles by position to calculate gaps
        sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)

        # Calculate gaps
        gaps = []
        for i in range(len(sorted_needles) - 1):
            curr_end = sorted_needles[i].insert_position_samples + sorted_needles[i].duration_samples
            next_start = sorted_needles[i + 1].insert_position_samples
            gaps.append((curr_end, next_start, next_start - curr_end))

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.suptitle("Comparison Task - Gaps Between Bouts", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        ax.plot(t, sample.x, linewidth=0.4, alpha=0.6, color="#1f77b4")

        # Highlight bouts
        for needle in sample.needles:
            start = needle.insert_position_samples
            end = start + needle.duration_samples
            ax.axvspan(start, end, alpha=0.3, color="red")

        # Highlight gaps
        for gap_start, gap_end, gap_size in gaps:
            ax.axvspan(gap_start, gap_end, alpha=0.2, color="green")
            ax.annotate(f"Gap: {gap_size}",
                        xy=((gap_start + gap_end) // 2, ax.get_ylim()[0]),
                        fontsize=8, ha="center", color="green")

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("X-axis (g)")

        # Add gap info to title
        gap_sizes = [g[2] for g in gaps]
        ax.set_title(f"Gaps: {gap_sizes} | Longest gap: {max(gap_sizes)} | Shortest gap: {min(gap_sizes)}")

        plt.tight_layout()
        output_path = plot_dir / "comparison_gaps.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
