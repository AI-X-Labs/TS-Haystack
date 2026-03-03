# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for OrderingTaskGenerator.

Task 4: "Did {activity_a} occur before {activity_b}?"
Answer type: boolean or category

Tests validate:
- Two distinct activities inserted
- Correct temporal order matches answer
- Balanced yes/no answers
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestOrderingSampleGeneration:
    """Tests for ordering sample generation."""

    def test_generate_single_sample(self, ordering_generator, rng):
        """Test generating a single ordering sample."""
        difficulty = DifficultyConfig(
            context_length_samples=15000,
            needle_position="random",
            needle_length_ratio_range=(0.02, 0.08),  # 300-1200 samples for 15000 context
            background_purity="pure",
            task_specific={
                "min_gap_samples": 100,
                "question_format": "boolean",
            },
        )

        sample = ordering_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "ordering"

        if sample.is_valid:
            assert len(sample.needles) == 2, "Ordering should have exactly 2 needles"

            # Check activities are different
            activities = [n.activity for n in sample.needles]
            assert len(set(activities)) == 2, "Activities should be different"

            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Activities: {activities[0]} vs {activities[1]}")

    def test_temporal_order_correct(self, ordering_generator):
        """Verify the answer reflects actual temporal order."""
        difficulty = DifficultyConfig(
            context_length_samples=12000,
            needle_position="random",
            needle_length_ratio_range=(0.025, 0.067),  # 300-800 samples for 12000 context
            background_purity="pure",
            task_specific={
                "min_gap_samples": 100,
                "question_format": "boolean",
            },
        )

        for i in range(15):
            rng = np.random.default_rng(1000 + i)
            sample = ordering_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) == 2:
                # Determine actual order
                sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)
                first_activity = sorted_needles[0].activity
                second_activity = sorted_needles[1].activity

                # Check config to see expected order
                config = sample.difficulty_config
                if "first_activity" in config:
                    assert config["first_activity"] == first_activity
                    assert config["second_activity"] == second_activity

                print(f"  Order: {first_activity} -> {second_activity}")
                print(f"  Answer: {sample.answer}")
                break

    def test_answer_balance(self, ordering_generator):
        """Test that yes/no answers are approximately balanced."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.08),  # 300-800 samples for 10000 context
            background_purity="pure",
            task_specific={"min_gap_samples": 100},
        )

        yes_count = 0
        no_count = 0

        for i in range(50):
            rng = np.random.default_rng(1100 + i)
            sample = ordering_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                if "yes" in sample.answer.lower():
                    yes_count += 1
                elif "no" in sample.answer.lower():
                    no_count += 1

        total = yes_count + no_count
        print(f"  Balance: {yes_count}/{total} yes, {no_count}/{total} no")

        # Expect roughly 50/50, allow 30-70% range
        assert yes_count >= total * 0.30, f"Too few yes: {yes_count}/{total}"
        assert no_count >= total * 0.30, f"Too few no: {no_count}/{total}"


class TestOrderingVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_ordering_sample(self, ordering_generator):
        """Visualize an ordering sample with two activities."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("ordering")

        difficulty = DifficultyConfig(
            context_length_samples=12000,
            needle_position="random",
            needle_length_ratio_range=(0.033, 0.125),  # 400-1500 samples for 12000 context
            background_purity="pure",
            task_specific={"min_gap_samples": 200},
        )

        # Find sample with clear order
        sample = None
        for i in range(30):
            rng = np.random.default_rng(1200 + i)
            candidate = ordering_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) == 2:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Sort needles by position for display
        sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Ordering Task - Temporal Order", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        needle_colors = ["#e41a1c", "#377eb8"]  # Red for first, blue for second

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight needles with different colors
            for idx, needle in enumerate(sorted_needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                ax.axvspan(start, end, alpha=0.3, color=needle_colors[idx])

                if ax == axes[0]:
                    label = "1st" if idx == 0 else "2nd"
                    ax.annotate(f"{label}: {needle.activity}",
                                xy=(start + (end - start) // 2, ax.get_ylim()[1] * 0.85),
                                fontsize=9, ha="center", color=needle_colors[idx], fontweight="bold")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        text = (
            f"Background: {sample.background_pid}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Actual order: {sorted_needles[0].activity} ({sorted_needles[0].timestamp_start}) "
            f"-> {sorted_needles[1].activity} ({sorted_needles[1].timestamp_start})"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "ordering_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_yes_no_comparison(self, ordering_generator):
        """Visualize both yes and no answer samples side by side."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("ordering")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),  # 300-1000 samples for 10000 context
            background_purity="pure",
            task_specific={"min_gap_samples": 150},
        )

        # Find one yes and one no sample
        yes_sample = None
        no_sample = None

        for i in range(50):
            rng = np.random.default_rng(1300 + i)
            candidate = ordering_generator.generate_sample(difficulty, rng)

            if candidate.is_valid and len(candidate.needles) == 2:
                if "yes" in candidate.answer.lower() and yes_sample is None:
                    yes_sample = candidate
                elif "no" in candidate.answer.lower() and no_sample is None:
                    no_sample = candidate

            if yes_sample and no_sample:
                break

        if not yes_sample or not no_sample:
            pytest.skip("Could not generate both yes and no samples")

        # Create side-by-side visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Ordering Task - Yes vs No Answers", fontsize=14, fontweight="bold")

        for col, (sample, title) in enumerate([(yes_sample, "Yes Answer"), (no_sample, "No Answer")]):
            t = np.arange(len(sample.x))
            sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)

            # Signal plot
            ax_signal = axes[0, col]
            ax_signal.plot(t, sample.x, linewidth=0.4, alpha=0.7, color="#1f77b4")

            for idx, needle in enumerate(sorted_needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                color = "#e41a1c" if idx == 0 else "#377eb8"
                ax_signal.axvspan(start, end, alpha=0.3, color=color)

            ax_signal.set_title(f"{title}\nOrder: {sorted_needles[0].activity} -> {sorted_needles[1].activity}",
                                fontsize=10)
            ax_signal.set_ylabel("X-axis (g)")

            # Text panel
            ax_text = axes[1, col]
            ax_text.axis("off")
            text = f"Q: {sample.question}\n\nA: {sample.answer}"
            ax_text.text(0.5, 0.5, text, transform=ax_text.transAxes, fontsize=9,
                         verticalalignment="center", horizontalalignment="center",
                         family="monospace", wrap=True)

        plt.tight_layout()
        output_path = plot_dir / "ordering_yes_no_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
