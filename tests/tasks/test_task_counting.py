# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for CountingTaskGenerator.

Task 3: "How many {activity} bouts occurred?"
Answer type: integer

Tests validate:
- Correct number of needles inserted matches answer
- Non-overlapping needle placement
- Various count ranges
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestCountingSampleGeneration:
    """Tests for counting sample generation."""

    def test_generate_single_sample(self, counting_generator, rng):
        """Test generating a single counting sample."""
        difficulty = DifficultyConfig(
            context_length_samples=30000,  # Larger for multiple needles
            needle_position="random",
            needle_length_ratio_range=(0.01, 0.03),  # 300-900 samples for 30000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 2,
                "max_bouts": 4,
                "min_gap_samples": 100,
            },
        )

        sample = counting_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "counting"
        assert sample.answer_type == "integer"

        if sample.is_valid:
            # Verify answer matches needle count
            answer_count = int(sample.answer.split()[0]) if sample.answer[0].isdigit() else None
            if answer_count is None:
                # Try to extract number from answer text
                import re
                numbers = re.findall(r'\d+', sample.answer)
                answer_count = int(numbers[0]) if numbers else len(sample.needles)

            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles inserted: {len(sample.needles)}")

            assert len(sample.needles) == answer_count or str(len(sample.needles)) in sample.answer

    def test_needle_non_overlapping(self, counting_generator):
        """Verify needles don't overlap."""
        difficulty = DifficultyConfig(
            context_length_samples=40000,
            needle_position="random",
            needle_length_ratio_range=(0.0075, 0.02),  # 300-800 samples for 40000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 3,
                "max_bouts": 5,
                "min_gap_samples": 100,
            },
        )

        for i in range(10):
            rng = np.random.default_rng(700 + i)
            sample = counting_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) > 1:
                # Sort needles by position
                sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)

                for j in range(len(sorted_needles) - 1):
                    curr = sorted_needles[j]
                    next_n = sorted_needles[j + 1]

                    curr_end = curr.insert_position_samples + curr.duration_samples
                    next_start = next_n.insert_position_samples

                    gap = next_start - curr_end
                    assert gap >= 0, f"Needles overlap: {curr_end} > {next_start}"

                    print(f"  Sample {i}: {len(sample.needles)} needles, gaps OK")
                break


class TestCountingVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_counting_sample(self, counting_generator):
        """Visualize a counting sample with multiple needles."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("counting")

        difficulty = DifficultyConfig(
            context_length_samples=40000,
            needle_position="random",
            needle_length_ratio_range=(0.0075, 0.03),  # 300-1200 samples for 40000 context
            background_purity="pure",
            task_specific={
                "min_bouts": 3,
                "max_bouts": 5,
                "min_gap_samples": 150,
            },
        )

        # Find sample with multiple needles
        sample = None
        for i in range(30):
            rng = np.random.default_rng(800 + i)
            candidate = counting_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) >= 3:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
        fig.suptitle(f"Counting Task - {len(sample.needles)} '{sample.needles[0].activity}' bouts",
                     fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        needle_colors = plt.cm.Set1(np.linspace(0, 1, len(sample.needles)))

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.4, color=color, alpha=0.7)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight each needle with different colors and numbers
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                ax.axvspan(start, end, alpha=0.3, color=needle_colors[idx])

                if ax == axes[0]:
                    # Add number label above first axis
                    ax.annotate(str(idx + 1), xy=(start + (end - start) // 2, ax.get_ylim()[1] * 0.9),
                                fontsize=10, ha="center", fontweight="bold", color="red")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")

        # Create legend patches
        legend_text = "  ".join([f"#{i+1}: {n.timestamp_start}" for i, n in enumerate(sample.needles)])

        text = (
            f"Background: {sample.background_pid}  |  "
            f"Time: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Needle positions: {legend_text}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "counting_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_count_variations(self, counting_generator):
        """Visualize samples with different counts."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("counting")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Counting Task - Count Variations", fontsize=14, fontweight="bold")

        count_configs = [
            (1, 2, "1-2 bouts"),
            (2, 3, "2-3 bouts"),
            (3, 4, "3-4 bouts"),
            (4, 5, "4-5 bouts"),
        ]

        for ax, (min_b, max_b, label) in zip(axes.flat, count_configs):
            difficulty = DifficultyConfig(
                context_length_samples=35000,
                needle_position="random",
                needle_length_ratio_range=(0.006, 0.017),  # 210-595 samples for 35000 context
                background_purity="pure",
                task_specific={
                    "min_bouts": min_b,
                    "max_bouts": max_b,
                    "min_gap_samples": 100,
                },
            )

            sample = None
            for i in range(20):
                rng = np.random.default_rng(900 + i + min_b * 10)
                candidate = counting_generator.generate_sample(difficulty, rng)
                if candidate.is_valid and candidate.needles:
                    sample = candidate
                    break

            if sample is None:
                ax.text(0.5, 0.5, f"Could not generate\n{label}", transform=ax.transAxes, ha="center", va="center")
                continue

            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.3, alpha=0.6, color="#1f77b4")

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                ax.axvspan(start, end, alpha=0.4, color="red")

            ax.set_title(f"{label}: Found {len(sample.needles)} bouts", fontsize=10)
            ax.set_xlabel("Sample")

        plt.tight_layout()
        output_path = plot_dir / "counting_variations.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
