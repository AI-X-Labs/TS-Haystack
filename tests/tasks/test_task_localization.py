# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for LocalizationTaskGenerator.

Task 2: "When did the {activity} bout occur?"
Answer type: time_range (timestamp start-end)

Tests validate:
- Sample generation with correct timestamp answers
- Needle insertion at various positions
- Timestamp formatting correctness
- Distractor insertion from same activity regime
- Multiple needles with single target selection
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


class TestLocalizationSampleGeneration:
    """Tests for localization sample generation."""

    def test_generate_single_sample(self, localization_generator, medium_difficulty, rng):
        """Test generating a single localization sample."""
        sample = localization_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "localization"

        if sample.is_valid:
            assert len(sample.needles) >= 1, "Localization should have at least one needle"

            # Find the target needle
            target_activity = sample.difficulty_config.get("target_activity")
            target_needle = None
            for needle in sample.needles:
                if needle.activity == target_activity:
                    target_needle = needle
                    break

            assert target_needle is not None, f"Target {target_activity} not found in needles"

            # Check timestamps in answer
            assert target_needle.timestamp_start in sample.answer or target_needle.timestamp_end in sample.answer

            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Target: {target_activity} @ {target_needle.timestamp_start}-{target_needle.timestamp_end}")
            print(f"  Total needles: {len(sample.needles)}")

    def test_needle_positions(self, localization_generator):
        """Test different needle position modes."""
        for position_mode in ["beginning", "middle", "end", "random"]:
            difficulty = DifficultyConfig(
                context_length_samples=10000,
                needle_position=position_mode,
                needle_length_ratio_range=(0.03, 0.15),  # 300-1500 samples for 10000 context
                background_purity="pure",
                task_specific={
                    "margin_samples": 100,
                    "min_distractors": 2,
                    "max_distractors": 4,
                    "min_gap_samples": 100,
                },
            )

            rng = np.random.default_rng(42)
            sample = localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.needles:
                # Look at target needle position
                target_idx = sample.difficulty_config.get("target_needle_index", 0)
                if target_idx < len(sample.needles):
                    target_needle = sample.needles[target_idx]
                    pos_frac = target_needle.insert_position_frac

                    print(f"  Position mode '{position_mode}': target at {pos_frac:.2%}")


# =============================================================================
# Distractor Insertion Tests
# =============================================================================


class TestLocalizationDistractorInsertion:
    """Tests for distractor insertion functionality."""

    def test_multiple_needles_single_target(self, localization_generator):
        """Verify multiple needles inserted but answer refers to specific one."""
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

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) >= 2:
                # Check that target activity is mentioned in question
                target = sample.difficulty_config.get("target_activity")
                assert target in sample.question, \
                    f"Target '{target}' should be in question: {sample.question}"

                # Check that answer contains target needle timestamps
                target_idx = sample.difficulty_config.get("target_needle_index")
                target_needle = sample.needles[target_idx]
                assert target_needle.timestamp_start in sample.answer or \
                       target_needle.timestamp_end in sample.answer, \
                    f"Answer should contain target timestamps"

                print(f"  Sample with {len(sample.needles)} needles:")
                print(f"    Target: {target} (index {target_idx})")
                print(f"    All activities: {[n.activity for n in sample.needles]}")
                print(f"    Q: {sample.question[:60]}...")
                print(f"    A: {sample.answer[:60]}...")
                return

        pytest.skip("Could not generate sample with multiple needles")

    def test_needles_from_same_regime(self, localization_generator):
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
            sample = localization_generator.generate_sample(difficulty, rng)

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

    def test_target_timestamp_in_answer(self, localization_generator):
        """Verify answer contains correct target needle timestamps."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.03, 0.12),
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        for i in range(20):
            rng = np.random.default_rng(400 + i)
            sample = localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and len(sample.needles) >= 2:
                target_idx = sample.difficulty_config.get("target_needle_index", 0)
                target_needle = sample.needles[target_idx]

                # Verify timestamps
                assert target_needle.timestamp_start in sample.answer, \
                    f"Start time '{target_needle.timestamp_start}' not in answer: {sample.answer}"

                print(f"  Target timestamps verified:")
                print(f"    Start: {target_needle.timestamp_start}")
                print(f"    End: {target_needle.timestamp_end}")
                print(f"    Answer: {sample.answer}")
                return

        pytest.skip("Could not generate sample with multiple needles")

    def test_needles_non_overlapping(self, localization_generator):
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
            sample = localization_generator.generate_sample(difficulty, rng)

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

                print(f"  Sample {i}: {len(sample.needles)} non-overlapping needles verified")
                return

        pytest.skip("Could not generate sample with multiple needles")


# =============================================================================
# Visualization Tests
# =============================================================================


class TestLocalizationVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_localization_sample(self, localization_generator):
        """Visualize a localization sample with distractors highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("localization")

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

        # Generate sample
        sample = None
        for i in range(30):
            rng = np.random.default_rng(500 + i)
            candidate = localization_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and len(candidate.needles) >= 2:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        target_idx = sample.difficulty_config.get("target_needle_index", 0)
        target_needle = sample.needles[target_idx]

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle(f"Localization Task - Find '{target_needle.activity}' (with Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        distractor_colors = ["gray", "lightblue", "lightgreen", "lightyellow"]

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight all needles (distractors in gray, target in red)
            distractor_idx = 0
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples

                if idx == target_idx:
                    # Target in red
                    ax.axvspan(start, end, alpha=0.4, color="red", label=f"TARGET: {needle.activity}")
                    ax.axvline(start, color="red", linestyle="--", alpha=0.7, linewidth=1)
                    ax.axvline(end, color="red", linestyle="--", alpha=0.7, linewidth=1)
                else:
                    # Distractors in muted colors
                    dc = distractor_colors[distractor_idx % len(distractor_colors)]
                    ax.axvspan(start, end, alpha=0.3, color=dc, label=f"distractor: {needle.activity}")
                    distractor_idx += 1

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Add position annotation on first axis
        axes[0].annotate(
            f"TARGET\n{target_needle.activity}\n{target_needle.timestamp_start}-{target_needle.timestamp_end}",
            xy=(target_needle.insert_position_samples + target_needle.duration_samples // 2, axes[0].get_ylim()[1]),
            fontsize=9, ha="center", color="red", fontweight="bold"
        )

        # Text panel
        axes[3].axis("off")
        all_activities = [n.activity for n in sample.needles]
        regime = sample.difficulty_config.get("selected_regime", "?")
        text = (
            f"Background: {sample.background_pid}  |  "
            f"Time Range: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Regime: {regime}  |  Target: {target_needle.activity}  |  All needles: {all_activities}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "localization_sample_with_distractors.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_position_modes(self, localization_generator):
        """Visualize samples with different position modes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("localization")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Localization Task - Position Modes (with Distractors)", fontsize=14, fontweight="bold")

        position_modes = ["beginning", "middle", "end", "random"]

        for ax, mode in zip(axes.flat, position_modes):
            difficulty = DifficultyConfig(
                context_length_samples=8000,
                needle_position=mode,
                needle_length_ratio_range=(0.04, 0.15),  # 320-1200 samples for 8000 context
                background_purity="pure",
                task_specific={
                    "margin_samples": 100,
                    "min_distractors": 2,
                    "max_distractors": 3,
                    "min_gap_samples": 100,
                },
            )

            sample = None
            for i in range(20):
                rng = np.random.default_rng(600 + i + hash(mode) % 100)
                candidate = localization_generator.generate_sample(difficulty, rng)
                if candidate.is_valid and candidate.needles:
                    sample = candidate
                    break

            if sample is None:
                ax.text(0.5, 0.5, f"Could not generate\n{mode} sample",
                        transform=ax.transAxes, ha="center", va="center")
                continue

            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.4, alpha=0.7, color="#1f77b4")

            target_idx = sample.difficulty_config.get("target_needle_index", 0)
            for idx, needle in enumerate(sample.needles):
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                color = "red" if idx == target_idx else "gray"
                alpha = 0.4 if idx == target_idx else 0.2
                ax.axvspan(start, end, alpha=alpha, color=color)

            target_needle = sample.needles[target_idx]
            ax.set_title(f"Position: {mode}\nTarget at {target_needle.insert_position_frac*100:.1f}% ({len(sample.needles)} needles)", fontsize=10)
            ax.set_xlabel("Sample")
            ax.set_ylabel("X-axis (g)")

        plt.tight_layout()
        output_path = plot_dir / "localization_position_modes.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_multiple_samples(self, localization_generator):
        """Create overview of multiple localization samples."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("localization")

        difficulty = DifficultyConfig(
            context_length_samples=8000,
            needle_position="random",
            needle_length_ratio_range=(0.04, 0.15),
            background_purity="pure",
            task_specific={
                "min_distractors": 2,
                "max_distractors": 4,
                "min_gap_samples": 100,
            },
        )

        # Generate multiple samples
        samples = []
        for i in range(15):
            rng = np.random.default_rng(700 + i)
            sample = localization_generator.generate_sample(difficulty, rng)
            if sample.is_valid and len(sample.needles) >= 2:
                samples.append(sample)
            if len(samples) >= 6:
                break

        if len(samples) < 4:
            pytest.skip("Could not generate enough samples")

        # Create grid
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle("Localization Task - Sample Overview (with Distractors)", fontsize=14, fontweight="bold")

        for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.3, alpha=0.7, color="#1f77b4")
            ax.plot(t, sample.y, linewidth=0.3, alpha=0.7, color="#2ca02c")
            ax.plot(t, sample.z, linewidth=0.3, alpha=0.7, color="#ff7f0e")

            target_idx = sample.difficulty_config.get("target_needle_index", 0)
            for nidx, needle in enumerate(sample.needles):
                color = "red" if nidx == target_idx else "gray"
                alpha = 0.4 if nidx == target_idx else 0.2
                ax.axvspan(needle.insert_position_samples,
                           needle.insert_position_samples + needle.duration_samples,
                           alpha=alpha, color=color)

            target_activity = sample.difficulty_config.get("target_activity", "?")
            a_short = sample.answer[:35] + "..." if len(sample.answer) > 35 else sample.answer
            n_needles = len(sample.needles)
            ax.set_title(f"Target: {target_activity}\nA: {a_short}\n({n_needles} needles)", fontsize=8, wrap=True)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        output_path = plot_dir / "localization_samples_overview.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
