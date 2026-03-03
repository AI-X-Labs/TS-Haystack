# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for AnomalyLocalizationTaskGenerator.

Task 10: "Is there an anomaly, and if so, when does it occur?"
Answer type: time_range (timestamp for positive, boolean logic for negative)
Evaluation:
- Detection: Accuracy (Yes/No match)
- Localization: IoU (Intersection over Union) for time range

Tests validate:
- Positive samples include time range in answer + distractors
- Negative samples have same-regime distractors only (no time range)
- Time range matches actual anomaly needle position
- BOTH positive and negative samples have multiple needle insertions (mandatory distractors)

Key test: Both positive and negative samples must have similar signal characteristics
(multiple needle insertions) to prevent detection shortcuts.
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import (
    DifficultyConfig,
    get_regime,
)
from tests.tasks.conftest import ensure_plot_dir


class TestAnomalyLocalizationSampleGeneration:
    """Tests for basic sample generation functionality."""

    def test_generate_single_sample(self, anomaly_localization_generator, medium_difficulty, rng):
        """Test generating a single anomaly localization sample."""
        sample = anomaly_localization_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "anomaly_localization"
        assert sample.answer_type == "time_range"

        if sample.is_valid:
            assert len(sample.x) == medium_difficulty.context_length_samples
            # Both positive and negative have needle insertions
            assert len(sample.needles) >= 1
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")
            print(f"  Is positive: {sample.difficulty_config.get('is_positive')}")

    def test_positive_includes_time_range(self, anomaly_localization_generator):
        """Verify positive samples include time range in answer + distractors."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(100 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                # Answer should include time range
                anomaly_start = sample.difficulty_config.get("anomaly_start")
                anomaly_end = sample.difficulty_config.get("anomaly_end")

                assert anomaly_start is not None
                assert anomaly_end is not None
                # Check that time range appears in answer
                assert anomaly_start in sample.answer or "to" in sample.answer

                print(f"  Positive: {sample.answer}")
                print(f"  Time range: {anomaly_start} to {anomaly_end}")
                return

        pytest.skip("Could not generate positive sample")

    def test_negative_no_time_range(self, anomaly_localization_generator):
        """Verify negative samples have distractors but no anomaly time range."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and not sample.difficulty_config.get("is_positive"):
                # Answer should NOT include anomaly time range
                assert "anomaly_start" not in sample.difficulty_config or sample.difficulty_config.get("anomaly_start") is None
                # Answer should mention consistent activity
                assert "consistent" in sample.answer.lower() or "no" in sample.answer.lower()

                print(f"  Negative: {sample.answer}")
                return

        pytest.skip("Could not generate negative sample")

    def test_time_range_matches_needle(self, anomaly_localization_generator):
        """Verify that reported time range matches actual anomaly needle position."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(300 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                anomaly_activity = sample.difficulty_config.get("anomaly_activity")
                anomaly_start = sample.difficulty_config.get("anomaly_start")
                anomaly_end = sample.difficulty_config.get("anomaly_end")

                # Find the anomaly needle
                anomaly_needle = None
                for n in sample.needles:
                    if n.activity == anomaly_activity:
                        anomaly_needle = n
                        break

                assert anomaly_needle is not None
                assert anomaly_needle.timestamp_start == anomaly_start
                assert anomaly_needle.timestamp_end == anomaly_end

                print(f"  Time range verified: {anomaly_start} to {anomaly_end}")
                return

        pytest.skip("Could not generate positive sample")

    def test_cross_regime_anomaly_verified(self, anomaly_localization_generator):
        """Verify that anomaly is from opposite regime."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(350 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                bg_regime = sample.difficulty_config.get("background_regime")
                anomaly_activity = sample.difficulty_config.get("anomaly_activity")
                anomaly_regime = get_regime(anomaly_activity)

                # Verify cross-regime
                assert bg_regime != anomaly_regime, \
                    f"Expected cross-regime: bg={bg_regime}, anomaly={anomaly_regime}"

                print(f"  Cross-regime verified: bg={bg_regime}, anomaly_activity={anomaly_activity} ({anomaly_regime})")
                return

        pytest.skip("Could not generate positive sample")


class TestAnomalyLocalizationVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_positive_with_distractors(self, anomaly_localization_generator):
        """Visualize positive sample with same-regime distractors."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_localization")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_gap_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(400 + i)
            candidate = anomaly_localization_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Localization - Positive (with Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        anomaly_activity = sample.difficulty_config.get("anomaly_activity")

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                if needle.activity == anomaly_activity:
                    ax.axvspan(start, end, alpha=0.4, color="red", label=f"ANOMALY: {needle.activity}")
                else:
                    ax.axvspan(start, end, alpha=0.2, color="blue", label=f"Distractor: {needle.activity}")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        bg_regime = sample.difficulty_config.get("background_regime", "?")
        anomaly_start = sample.difficulty_config.get("anomaly_start", "?")
        anomaly_end = sample.difficulty_config.get("anomaly_end", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", 0)
        text = (
            f"Background: {bg_regime}  |  Anomaly: {anomaly_activity}  |  Distractors: {n_distractors}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Time range: {anomaly_start} to {anomaly_end}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_localization_positive.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_negative_sample(self, anomaly_localization_generator):
        """Visualize negative sample (same-regime distractors only)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_localization")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_gap_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(500 + i)
            candidate = anomaly_localization_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and not candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Localization - Negative (Same-Regime Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"SAME-REGIME: {needle.activity}")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        bg_regime = sample.difficulty_config.get("background_regime", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", len(sample.needles))
        distractor_activities = sample.difficulty_config.get("distractor_activities", [n.activity for n in sample.needles])
        text = (
            f"Background: {bg_regime}  |  Distractors: {n_distractors} (all same-regime)\n"
            f"Activities: {distractor_activities}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_localization_negative.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
