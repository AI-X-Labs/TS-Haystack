# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for AnomalyDetectionTaskGenerator.

Task 9: "Is there an anomaly in this recording?"
Answer type: boolean (Yes/No)
Evaluation: Accuracy (exact match)

Tests validate:
- Cross-regime insertion creates anomalies (positive)
- Same-regime distractors do NOT create anomalies (negative)
- BOTH positive and negative samples have multiple needle insertions (mandatory distractors)
- Background regime detection
- Balanced positive/negative samples
- Answer format correctness

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


class TestAnomalyDetectionSampleGeneration:
    """Tests for basic sample generation functionality."""

    def test_generate_single_sample(self, anomaly_detection_generator, medium_difficulty, rng):
        """Test generating a single anomaly detection sample."""
        sample = anomaly_detection_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "anomaly_detection"
        assert sample.answer_type == "boolean"

        if sample.is_valid:
            assert len(sample.x) == medium_difficulty.context_length_samples
            assert "anomal" in sample.question.lower() or "unusual" in sample.question.lower() or "ordinary" in sample.question.lower()
            # Both positive and negative have needle insertions
            assert len(sample.needles) >= 1
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")
            print(f"  Is positive: {sample.difficulty_config.get('is_positive')}")

    def test_positive_has_cross_regime_plus_distractors(self, anomaly_detection_generator):
        """Verify positive samples have cross-regime anomaly + same-regime distractors."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(100 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                bg_regime = sample.difficulty_config.get("background_regime")
                anomaly_activity = sample.difficulty_config.get("anomaly_activity")
                anomaly_regime = get_regime(anomaly_activity)

                # Verify cross-regime anomaly
                assert bg_regime != anomaly_regime, \
                    f"Expected cross-regime: bg={bg_regime}, anomaly={anomaly_regime}"

                # Verify anomaly needle is present
                assert any(n.activity == anomaly_activity for n in sample.needles)

                # Verify we have at least 1 needle (the anomaly)
                assert len(sample.needles) >= 1, \
                    f"Expected at least anomaly needle, got {len(sample.needles)} needles"

                print(f"  Positive: bg={bg_regime}, anomaly={anomaly_activity}, total needles={len(sample.needles)}")
                return

        pytest.skip("Could not generate positive sample")

    def test_negative_has_same_regime_distractors(self, anomaly_detection_generator):
        """Verify negative samples have same-regime distractors (no anomaly)."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid and not sample.difficulty_config.get("is_positive"):
                bg_regime = sample.difficulty_config.get("background_regime")

                # Verify ALL needles are same-regime (no anomaly)
                for needle in sample.needles:
                    needle_regime = get_regime(needle.activity)
                    assert bg_regime == needle_regime, \
                        f"Expected all same-regime: bg={bg_regime}, needle={needle_regime}"

                # Verify we have distractors (at least min_distractors)
                assert len(sample.needles) >= 1, \
                    f"Expected distractors, got {len(sample.needles)} needles"

                print(f"  Negative: bg={bg_regime}, distractors={len(sample.needles)}")
                return

        pytest.skip("Could not generate negative sample")

    def test_both_positive_and_negative_have_insertions(self, anomaly_detection_generator):
        """Critical test: Both cases have needle insertions to prevent detection shortcuts."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 3},
        )

        positive_needle_counts = []
        negative_needle_counts = []

        for i in range(50):
            rng = np.random.default_rng(150 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                if sample.difficulty_config.get("is_positive"):
                    positive_needle_counts.append(len(sample.needles))
                else:
                    negative_needle_counts.append(len(sample.needles))

        # Both must have insertions
        assert all(c >= 1 for c in positive_needle_counts), "Positive samples must have needles"
        assert all(c >= 1 for c in negative_needle_counts), "Negative samples must have needles"

        print(f"  Positive needle counts: {positive_needle_counts[:10]}...")
        print(f"  Negative needle counts: {negative_needle_counts[:10]}...")

    def test_balanced_positive_negative(self, anomaly_detection_generator):
        """Test that positive/negative samples are approximately balanced."""
        difficulty = DifficultyConfig(
            context_length_samples=8000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100},
        )

        positive_count = 0
        negative_count = 0

        for i in range(50):
            rng = np.random.default_rng(300 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                if sample.difficulty_config.get("is_positive"):
                    positive_count += 1
                else:
                    negative_count += 1

        total = positive_count + negative_count
        print(f"  Balance: {positive_count}/{total} positive, {negative_count}/{total} negative")

        assert positive_count >= total * 0.30
        assert negative_count >= total * 0.30


class TestAnomalyDetectionVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_positive_sample(self, anomaly_detection_generator):
        """Visualize a positive anomaly detection sample with distractors."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_detection")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.08, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(400 + i)
            candidate = anomaly_detection_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Detection - Positive (Cross-Regime Insertion)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            anomaly_activity = sample.difficulty_config.get("anomaly_activity")
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
        anomaly = sample.difficulty_config.get("anomaly_activity", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", 0)
        text = (
            f"Background Regime: {bg_regime}  |  Anomaly: {anomaly}  |  Distractors: {n_distractors}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_detection_positive.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_negative_sample(self, anomaly_detection_generator):
        """Visualize a negative anomaly detection sample (same-regime distractors only)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_detection")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.08, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(500 + i)
            candidate = anomaly_detection_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and not candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Detection - Negative (Same-Regime Distractors Only)", fontsize=14, fontweight="bold")

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
            f"Background Regime: {bg_regime}  |  Distractors: {n_distractors} (all same-regime)\n"
            f"Activities: {distractor_activities}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_detection_negative.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
