# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for ExistenceTaskGenerator.

Task 1: "Is there {activity} in this recording?"
Answer type: boolean (Yes/No)

Tests validate:
- Sample generation produces valid GeneratedSample objects
- Question/answer format correctness
- Balanced positive/negative samples
- Distractor insertion from same activity regime
- Visualization for human evaluation
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import (
    DifficultyConfig,
    WILLETTS_ACTIVITY_REGIMES,
    get_regime,
)
from tests.tasks.conftest import ensure_plot_dir


# =============================================================================
# Sample Generation Tests
# =============================================================================


class TestExistenceSampleGeneration:
    """Tests for basic sample generation functionality."""

    def test_generate_single_sample(self, existence_generator, medium_difficulty, rng):
        """Test generating a single existence sample."""
        sample = existence_generator.generate_sample(medium_difficulty, rng)

        # Check basic structure
        assert sample is not None
        assert sample.task_type == "existence"
        assert sample.answer_type == "boolean"

        if sample.is_valid:
            assert len(sample.x) == medium_difficulty.context_length_samples
            assert len(sample.y) == medium_difficulty.context_length_samples
            assert len(sample.z) == medium_difficulty.context_length_samples
            assert len(sample.question) > 0
            assert sample.answer in ["Yes", "No", "Yes.", "No."] or "yes" in sample.answer.lower() or "no" in sample.answer.lower()

            print(f"  Valid: {sample.is_valid}")
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")

    def test_generate_multiple_samples_balance(self, existence_generator):
        """Test that positive/negative samples are approximately balanced."""
        difficulty = DifficultyConfig(
            context_length_samples=5000,
            needle_position="random",
            needle_length_ratio_range=(0.06, 0.20),  # 300-1000 samples for 5000 context
            background_purity="pure",
            task_specific={
                "min_distractors": 1,
                "max_distractors": 3,
                "min_gap_samples": 100,
            },
        )

        n_samples = 50
        positive_count = 0
        negative_count = 0

        for i in range(n_samples):
            rng = np.random.default_rng(42 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                answer_lower = sample.answer.lower()
                if "yes" in answer_lower:
                    positive_count += 1
                elif "no" in answer_lower:
                    negative_count += 1

        total = positive_count + negative_count
        print(f"  Balance: {positive_count}/{total} positive, {negative_count}/{total} negative")

        # Expect roughly 50/50, allow 30-70% range
        assert positive_count >= total * 0.30, f"Too few positive: {positive_count}/{total}"
        assert negative_count >= total * 0.30, f"Too few negative: {negative_count}/{total}"

    def test_needle_insertion_for_positive(self, existence_generator):
        """Test that positive samples with inserted needles have correct needle metadata."""
        difficulty = DifficultyConfig(
            context_length_samples=8000,
            needle_position="random",
            needle_length_ratio_range=(0.04, 0.19),  # 320-1520 samples for 8000 context
            background_purity="pure",
            task_specific={
                "min_distractors": 1,
                "max_distractors": 3,
                "min_gap_samples": 100,
            },
        )

        # Generate samples until we get a positive one with needle
        for i in range(30):
            rng = np.random.default_rng(100 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid and "yes" in sample.answer.lower() and sample.needles:
                # Check that target activity is in one of the needles
                target = sample.difficulty_config.get("target_activity")
                inserted = [n.activity for n in sample.needles]
                assert target in inserted, f"Target {target} not in inserted {inserted}"

                print(f"  Found positive sample with needles:")
                print(f"    Target activity: {target}")
                print(f"    Inserted activities: {inserted}")
                print(f"    N needles: {len(sample.needles)}")
                return

        pytest.skip("Could not generate positive sample with needle insertion")


# =============================================================================
# Distractor Insertion Tests
# =============================================================================


class TestExistenceDistractorInsertion:
    """Tests for distractor insertion functionality."""

    def test_multiple_needles_inserted(self, existence_generator):
        """Verify multiple needles are inserted when configured."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),  # 300-1000 samples
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        multi_needle_count = 0
        total_valid = 0

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                total_valid += 1
                if len(sample.needles) >= 2:
                    multi_needle_count += 1
                    print(f"  Sample {i}: {len(sample.needles)} needles inserted")

        print(f"  Multi-needle samples: {multi_needle_count}/{total_valid}")
        # Most samples should have multiple needles
        assert multi_needle_count >= total_valid * 0.5, \
            f"Expected >= 50% multi-needle, got {multi_needle_count}/{total_valid}"

    def test_needles_from_same_regime(self, existence_generator):
        """Verify all inserted needles are from the same activity regime."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        for i in range(20):
            rng = np.random.default_rng(300 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) >= 2:
                # Check all needles are from the same regime
                regimes = set()
                for needle in sample.needles:
                    try:
                        regime = get_regime(needle.activity)
                        regimes.add(regime)
                    except ValueError:
                        continue

                assert len(regimes) == 1, \
                    f"Expected single regime, got {regimes} for activities {[n.activity for n in sample.needles]}"

                print(f"  Sample {i}: {len(sample.needles)} needles from regime '{list(regimes)[0]}'")
                print(f"    Activities: {[n.activity for n in sample.needles]}")
                return

        pytest.skip("Could not generate sample with multiple needles from same regime")

    def test_negative_from_same_regime(self, existence_generator):
        """Verify negative targets are from same regime as inserted needles."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),
            background_purity="pure",
            task_specific={
                "min_distractors": 1,
                "max_distractors": 3,
                "min_gap_samples": 100,
            },
        )

        for i in range(30):
            rng = np.random.default_rng(400 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid and "no" in sample.answer.lower() and sample.needles:
                target = sample.difficulty_config.get("target_activity")
                selected_regime = sample.difficulty_config.get("selected_regime")
                inserted_activities = sample.difficulty_config.get("inserted_activities", [])

                # Target should be from the same regime as inserted needles
                try:
                    target_regime = get_regime(target)
                    assert target_regime == selected_regime, \
                        f"Target regime '{target_regime}' != selected regime '{selected_regime}'"
                except ValueError:
                    continue

                # Target should NOT be in inserted activities
                assert target not in inserted_activities, \
                    f"Negative target '{target}' should not be in inserted {inserted_activities}"

                print(f"  Negative sample found:")
                print(f"    Target: {target} (regime: {target_regime})")
                print(f"    Inserted: {inserted_activities}")
                print(f"    Selected regime: {selected_regime}")
                return

        pytest.skip("Could not generate negative sample with distractor insertion")

    def test_needles_non_overlapping(self, existence_generator):
        """Verify inserted needles don't overlap."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.10),
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        for i in range(20):
            rng = np.random.default_rng(500 + i)
            sample = existence_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) >= 2:
                # Sort needles by position
                sorted_needles = sorted(sample.needles, key=lambda n: n.insert_position_samples)

                # Check no overlaps
                for j in range(len(sorted_needles) - 1):
                    current_end = sorted_needles[j].insert_position_samples + sorted_needles[j].duration_samples
                    next_start = sorted_needles[j + 1].insert_position_samples
                    assert current_end <= next_start, \
                        f"Needles overlap: {sorted_needles[j].activity} ends at {current_end}, " \
                        f"{sorted_needles[j + 1].activity} starts at {next_start}"

                print(f"  Sample {i}: {len(sample.needles)} non-overlapping needles")
                return

        pytest.skip("Could not generate sample with multiple needles")


# =============================================================================
# Visualization Tests
# =============================================================================


class TestExistenceVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_positive_sample(self, existence_generator):
        """Visualize a positive existence sample."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("existence")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.20),  # 500-2000 samples for 10000 context
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        # Find a good positive sample
        sample = None
        for i in range(50):
            rng = np.random.default_rng(200 + i)
            candidate = existence_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and "yes" in candidate.answer.lower() and candidate.needles:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Existence Task - Positive Sample (with Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        needle_colors = ["red", "blue", "green", "purple", "orange"]

        # Plot each axis
        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight needle regions with different colors
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                nc = needle_colors[idx % len(needle_colors)]
                ax.axvspan(start, end, alpha=0.3, color=nc, label=needle.activity)

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        target = sample.difficulty_config.get("target_activity", "?")
        inserted = sample.difficulty_config.get("inserted_activities", [])
        regime = sample.difficulty_config.get("selected_regime", "?")
        text = (
            f"Background PID: {sample.background_pid}  |  "
            f"Time: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Regime: {regime}  |  Target: {target}  |  Inserted: {inserted}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "existence_positive_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_negative_sample(self, existence_generator):
        """Visualize a negative existence sample."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("existence")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.15),  # 300-1500 samples for 10000 context
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 3,
                "min_gap_samples": 100,
            },
        )

        # Find a negative sample
        sample = None
        for i in range(50):
            rng = np.random.default_rng(300 + i)
            candidate = existence_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and "no" in candidate.answer.lower() and candidate.needles:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Existence Task - Negative Sample (with Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        needle_colors = ["red", "blue", "green", "purple", "orange"]

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight distractor needles
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                nc = needle_colors[idx % len(needle_colors)]
                ax.axvspan(start, end, alpha=0.3, color=nc, label=needle.activity)

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        target = sample.difficulty_config.get("target_activity", "?")
        inserted = sample.difficulty_config.get("inserted_activities", [])
        regime = sample.difficulty_config.get("selected_regime", "?")
        text = (
            f"Background PID: {sample.background_pid}  |  "
            f"Time: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Regime: {regime}  |  Target (NOT inserted): {target}  |  Inserted: {inserted}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "existence_negative_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_multiple_samples(self, existence_generator):
        """Create overview of multiple existence samples."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("existence")

        difficulty = DifficultyConfig(
            context_length_samples=8000,
            needle_position="random",
            needle_length_ratio_range=(0.04, 0.19),  # 320-1520 samples for 8000 context
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 3,
                "min_gap_samples": 100,
            },
        )

        # Generate multiple samples
        samples = []
        for i in range(12):
            rng = np.random.default_rng(400 + i)
            sample = existence_generator.generate_sample(difficulty, rng)
            if sample.is_valid:
                samples.append(sample)
            if len(samples) >= 6:
                break

        if len(samples) < 4:
            pytest.skip("Could not generate enough samples")

        # Create grid
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle("Existence Task - Sample Overview (with Distractors)", fontsize=14, fontweight="bold")

        needle_colors = ["red", "blue", "green", "purple", "orange"]

        for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.3, alpha=0.7, color="#1f77b4")
            ax.plot(t, sample.y, linewidth=0.3, alpha=0.7, color="#2ca02c")
            ax.plot(t, sample.z, linewidth=0.3, alpha=0.7, color="#ff7f0e")

            for nidx, needle in enumerate(sample.needles):
                nc = needle_colors[nidx % len(needle_colors)]
                ax.axvspan(needle.insert_position_samples,
                           needle.insert_position_samples + needle.duration_samples,
                           alpha=0.3, color=nc)

            q_short = sample.question[:45] + "..." if len(sample.question) > 45 else sample.question
            a_short = sample.answer[:25] + "..." if len(sample.answer) > 25 else sample.answer
            n_needles = len(sample.needles)
            ax.set_title(f"Q: {q_short}\nA: {a_short} ({n_needles} needles)", fontsize=8, wrap=True)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        output_path = plot_dir / "existence_samples_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
