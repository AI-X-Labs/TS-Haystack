# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for AntecedentTaskGenerator.

Task 6: "What activity occurred immediately before {target_activity}?"
Answer type: category

Tests validate:
- Two needles inserted adjacent to each other
- Answer is the antecedent (first) activity
- Small gap between needles
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestAntecedentSampleGeneration:
    """Tests for antecedent sample generation."""

    def test_generate_single_sample(self, antecedent_generator, rng):
        """Test generating a single antecedent sample."""
        difficulty = DifficultyConfig(
            context_length_samples=12000,
            needle_position="random",
            needle_length_ratio_range=(0.025, 0.10),  # 300-1200 samples for 12000 context
            background_purity="pure",
            task_specific={
                "adjacency_gap_samples": 10,
                "margin_samples": 100,
            },
        )

        sample = antecedent_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "antecedent"
        assert sample.answer_type == "category"

        if sample.is_valid:
            # Should have exactly 2 needles (antecedent + target)
            assert len(sample.needles) == 2

            # Sort by position to identify antecedent
            sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)
            antecedent = sorted_needles[0]
            target = sorted_needles[1]

            # Answer should be the antecedent activity
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Antecedent: {antecedent.activity}")
            print(f"  Target: {target.activity}")

            # The answer should match the antecedent
            assert antecedent.activity.lower() in sample.answer.lower() or \
                   sample.answer.lower() in antecedent.activity.lower()

    def test_needles_adjacent(self, antecedent_generator):
        """Verify the two needles are inserted adjacently."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.02, 0.08),  # 200-800 samples for 10000 context
            background_purity="pure",
            task_specific={
                "adjacency_gap_samples": 20,
                "margin_samples": 100,
            },
        )

        for i in range(15):
            rng = np.random.default_rng(1600 + i)
            sample = antecedent_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) == 2:
                sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)
                antecedent_end = sorted_needles[0].insert_position_samples + sorted_needles[0].duration_samples
                target_start = sorted_needles[1].insert_position_samples

                gap = target_start - antecedent_end
                print(f"  Gap between needles: {gap} samples")

                # Gap should be small (as configured)
                assert gap >= 0, "Needles should not overlap"
                assert gap <= 100, f"Gap should be small for adjacency, got {gap}"
                return

        pytest.skip("Could not generate suitable sample")


class TestAntecedentVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_antecedent_sample(self, antecedent_generator):
        """Visualize an antecedent sample with adjacent needles."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("antecedent")

        difficulty = DifficultyConfig(
            context_length_samples=12000,
            needle_position="random",
            needle_length_ratio_range=(0.025, 0.125),  # 300-1500 samples for 12000 context
            background_purity="pure",
            task_specific={
                "adjacency_gap_samples": 15,
                "margin_samples": 100,
            },
        )

        # Find suitable sample
        sample = None
        for i in range(40):
            rng = np.random.default_rng(1700 + i)
            candidate = antecedent_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) == 2:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)
        antecedent = sorted_needles[0]
        target = sorted_needles[1]

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Antecedent Task - What Came Before?", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Antecedent (green) and target (red)
            ax.axvspan(antecedent.insert_position_samples,
                       antecedent.insert_position_samples + antecedent.duration_samples,
                       alpha=0.3, color="green")
            ax.axvspan(target.insert_position_samples,
                       target.insert_position_samples + target.duration_samples,
                       alpha=0.3, color="red")

            if ax == axes[0]:
                ax.annotate(f"ANTECEDENT\n{antecedent.activity}",
                            xy=(antecedent.insert_position_samples + antecedent.duration_samples // 2,
                                ax.get_ylim()[1] * 0.85),
                            fontsize=9, ha="center", color="green", fontweight="bold")
                ax.annotate(f"TARGET\n{target.activity}",
                            xy=(target.insert_position_samples + target.duration_samples // 2,
                                ax.get_ylim()[1] * 0.85),
                            fontsize=9, ha="center", color="red", fontweight="bold")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        gap = target.insert_position_samples - (antecedent.insert_position_samples + antecedent.duration_samples)
        text = (
            f"Background: {sample.background_pid}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Antecedent: {antecedent.activity} ({antecedent.timestamp_start})\n"
            f"Target: {target.activity} ({target.timestamp_start})\n"
            f"Gap: {gap} samples"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "antecedent_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_zoomed_transition(self, antecedent_generator):
        """Visualize the transition zone between antecedent and target."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("antecedent")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),  # 300-1000 samples for 10000 context
            background_purity="pure",
            task_specific={"adjacency_gap_samples": 10},
        )

        sample = None
        for i in range(30):
            rng = np.random.default_rng(1800 + i)
            candidate = antecedent_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) == 2:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)
        antecedent = sorted_needles[0]
        target = sorted_needles[1]

        # Calculate zoom window around the transition
        transition_center = antecedent.insert_position_samples + antecedent.duration_samples
        margin = 300
        start_idx = max(0, transition_center - margin)
        end_idx = min(len(sample.x), transition_center + margin)

        # Create zoomed visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f"Antecedent Task - Zoomed Transition\n{antecedent.activity} -> {target.activity}",
                     fontsize=14, fontweight="bold")

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
        for ax, (name, data, color) in zip(axes, [("X", sample.x, colors[0]),
                                                   ("Y", sample.y, colors[1]),
                                                   ("Z", sample.z, colors[2])]):
            t_zoomed = np.arange(start_idx, end_idx)
            ax.plot(t_zoomed, data[start_idx:end_idx], linewidth=1, color=color)
            ax.set_ylabel(f"{name}-axis (g)")

            # Highlight regions
            ax.axvspan(antecedent.insert_position_samples,
                       antecedent.insert_position_samples + antecedent.duration_samples,
                       alpha=0.3, color="green")
            ax.axvspan(target.insert_position_samples,
                       target.insert_position_samples + target.duration_samples,
                       alpha=0.3, color="red")

            # Mark transition point
            ax.axvline(transition_center, color="black", linestyle="--", linewidth=2, alpha=0.7)

            ax.set_xlim(start_idx, end_idx)

        axes[2].set_xlabel("Sample Index")

        plt.tight_layout()
        output_path = plot_dir / "antecedent_transition_zoom.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
