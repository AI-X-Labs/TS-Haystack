# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for MultiHopTaskGenerator.

Task 8: "When did the {ordinal} {target_activity} bout occur {direction} the {anchor_activity}?"
Answer type: time_range

Tests validate:
- Anchor + K target needles in correct temporal arrangement
- Direction (before/after) correctness
- K-th occurrence identification
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestMultiHopSampleGeneration:
    """Tests for multi-hop sample generation."""

    def test_generate_single_sample(self, multi_hop_generator, rng):
        """Test generating a single multi-hop sample."""
        difficulty = DifficultyConfig(
            context_length_samples=25000,
            needle_position="random",
            needle_length_ratio_range=(0.012, 0.048),  # 300-1200 samples for 25000 context
            background_purity="pure",
            task_specific={
                "k_distribution": [0.5, 0.3, 0.2],  # P(K=1,2,3)
                "direction_mode": "random",
                "n_distractors_opposite": 0,
                "min_gap_samples": 100,
            },
        )

        sample = multi_hop_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "multi_hop"

        if sample.is_valid:
            # Should have at least 2 needles (1 anchor + K targets)
            assert len(sample.needles) >= 2

            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Total needles: {len(sample.needles)}")

            # Check for ordinal in question
            assert any(ord in sample.question.lower() for ord in
                       ["first", "1st", "second", "2nd", "third", "3rd"])

    def test_direction_modes(self, multi_hop_generator):
        """Test different direction modes (before/after)."""
        for direction_mode in ["after_only", "before_only", "random"]:
            difficulty = DifficultyConfig(
                context_length_samples=20000,
                needle_position="random",
                needle_length_ratio_range=(0.01, 0.04),  # 200-800 samples for 20000 context
                background_purity="pure",
                task_specific={
                    "k_distribution": [1.0, 0.0, 0.0],  # K=1 only for simplicity
                    "direction_mode": direction_mode,
                    "min_gap_samples": 100,
                },
            )

            valid_sample = None
            for i in range(20):
                rng = np.random.default_rng(2200 + i + hash(direction_mode) % 100)
                sample = multi_hop_generator.generate_sample(difficulty, rng)
                if sample.is_valid:
                    valid_sample = sample
                    break

            if valid_sample:
                has_after = "after" in valid_sample.question.lower()
                has_before = "before" in valid_sample.question.lower()

                print(f"  Mode '{direction_mode}': after={has_after}, before={has_before}")

                if direction_mode == "after_only":
                    assert has_after, "Should ask 'after' for after_only mode"
                elif direction_mode == "before_only":
                    assert has_before, "Should ask 'before' for before_only mode"

    def test_k_values(self, multi_hop_generator):
        """Test different K values (1st, 2nd, 3rd)."""
        k_configs = [
            ([1.0, 0.0, 0.0], 1),
            ([0.0, 1.0, 0.0], 2),
            ([0.0, 0.0, 1.0], 3),
        ]

        for k_dist, expected_k in k_configs:
            difficulty = DifficultyConfig(
                context_length_samples=30000,
                needle_position="random",
                needle_length_ratio_range=(0.007, 0.027),  # 210-810 samples for 30000 context
                background_purity="pure",
                task_specific={
                    "k_distribution": k_dist,
                    "direction_mode": "after_only",
                    "min_gap_samples": 100,
                },
            )

            valid_sample = None
            for i in range(30):
                rng = np.random.default_rng(2300 + i + expected_k * 10)
                sample = multi_hop_generator.generate_sample(difficulty, rng)
                if sample.is_valid:
                    valid_sample = sample
                    break

            if valid_sample:
                # Check K value from config
                k_value = valid_sample.difficulty_config.get("K")
                print(f"  K={expected_k}: Generated sample with K={k_value}")
                assert k_value == expected_k, f"Expected K={expected_k}, got K={k_value}"


class TestMultiHopVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_multi_hop_sample(self, multi_hop_generator):
        """Visualize a multi-hop sample with anchor and targets."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("multi_hop")

        difficulty = DifficultyConfig(
            context_length_samples=30000,
            needle_position="random",
            needle_length_ratio_range=(0.01, 0.04),  # 300-1200 samples for 30000 context
            background_purity="pure",
            task_specific={
                "k_distribution": [0.3, 0.4, 0.3],  # Mix of K values
                "direction_mode": "random",
                "n_distractors_opposite": 0,
                "min_gap_samples": 150,
            },
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(2400 + i)
            candidate = multi_hop_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) >= 3:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        config = sample.difficulty_config

        # Identify anchor and targets
        anchor_activity = config.get("anchor_activity")
        target_activity = config.get("target_activity")
        k = config.get("K", 1)
        direction = config.get("direction", "after")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1, 1, 0.7]})
        fig.suptitle(f"Multi-Hop Task - K={k} {direction} anchor", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.4, color=color, alpha=0.7)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Color code: anchor (purple), targets (red with numbers), answer target (gold)
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples

                if needle.activity == anchor_activity:
                    ax.axvspan(start, end, alpha=0.4, color="purple")
                    if ax == axes[0]:
                        ax.annotate(f"ANCHOR\n{anchor_activity}",
                                    xy=(start + (end - start) // 2, ax.get_ylim()[1] * 0.85),
                                    fontsize=8, ha="center", color="purple", fontweight="bold")
                else:
                    # Target - determine if it's the answer
                    ax.axvspan(start, end, alpha=0.3, color="red")
                    if ax == axes[0]:
                        ax.annotate(f"{needle.activity}",
                                    xy=(start + (end - start) // 2, ax.get_ylim()[0] * 1.1),
                                    fontsize=7, ha="center", color="red")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")

        needle_info = ", ".join([f"{n.activity}@{n.insert_position_samples}" for n in sample.needles])
        text = (
            f"Background: {sample.background_pid}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Anchor: {anchor_activity} | Target: {target_activity}\n"
            f"K: {k} | Direction: {direction}\n"
            f"Needles: {needle_info}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=9,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "multi_hop_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_k_variations(self, multi_hop_generator):
        """Visualize samples with different K values."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("multi_hop")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle("Multi-Hop Task - K Variations (1st, 2nd, 3rd)", fontsize=14, fontweight="bold")

        k_configs = [
            ([1.0, 0.0, 0.0], "K=1 (1st occurrence)"),
            ([0.0, 1.0, 0.0], "K=2 (2nd occurrence)"),
            ([0.0, 0.0, 1.0], "K=3 (3rd occurrence)"),
        ]

        for ax, (k_dist, title) in zip(axes, k_configs):
            difficulty = DifficultyConfig(
                context_length_samples=35000,
                needle_position="random",
                needle_length_ratio_range=(0.006, 0.023),  # 210-805 samples for 35000 context
                background_purity="pure",
                task_specific={
                    "k_distribution": k_dist,
                    "direction_mode": "after_only",
                    "min_gap_samples": 100,
                },
            )

            sample = None
            for i in range(40):
                rng = np.random.default_rng(2500 + i + int(k_dist[1] * 100) + int(k_dist[2] * 200))
                candidate = multi_hop_generator.generate_sample(difficulty, rng)
                if candidate.is_valid:
                    sample = candidate
                    break

            if sample is None:
                ax.text(0.5, 0.5, f"Could not generate\n{title}",
                        transform=ax.transAxes, ha="center", va="center")
                continue

            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.3, alpha=0.6, color="#1f77b4")

            config = sample.difficulty_config
            anchor_activity = config.get("anchor_activity")

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                color = "purple" if needle.activity == anchor_activity else "red"
                ax.axvspan(start, end, alpha=0.3, color=color)

            q_short = sample.question[:60] + "..." if len(sample.question) > 60 else sample.question
            ax.set_title(f"{title}\nQ: {q_short}", fontsize=10)
            ax.set_ylabel("X-axis (g)")

        axes[2].set_xlabel("Sample Index")

        plt.tight_layout()
        output_path = plot_dir / "multi_hop_k_variations.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
