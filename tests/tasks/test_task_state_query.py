# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for StateQueryTaskGenerator.

Task 5: "What was the overall activity level when {needle_activity} occurred?"
Answer type: category (the global state, not the needle activity)

Tests validate:
- Mixed background with multiple states
- Needle inserted within a global state
- Answer is the global state, not the needle
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import DifficultyConfig
from tests.tasks.conftest import ensure_plot_dir


class TestStateQuerySampleGeneration:
    """Tests for state query sample generation."""

    def test_generate_single_sample(self, state_query_generator, rng):
        """Test generating a single state query sample."""
        difficulty = DifficultyConfig(
            context_length_samples=15000,
            needle_position="random",
            needle_length_ratio_range=(0.02, 0.067),  # 300-1000 samples for 15000 context
            background_purity="mixed",  # Required for state query
            task_specific={
                "min_global_states": 2,
                "max_global_states": 4,
            },
        )

        sample = state_query_generator.generate_sample(difficulty, rng)

        assert sample is not None
        assert sample.task_type == "state_query"
        assert sample.answer_type == "category"

        if sample.is_valid:
            # Answer should be the global state, not the needle activity
            assert len(sample.needles) >= 1
            needle_activity = sample.needles[0].activity

            # Answer should NOT be the needle activity (it's the surrounding state)
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needle activity: {needle_activity}")
            print(f"  Global state (answer): {sample.answer}")

    def test_needle_within_state_bounds(self, state_query_generator):
        """Verify needle is inserted within a valid global state."""
        difficulty = DifficultyConfig(
            context_length_samples=12000,
            needle_position="random",
            needle_length_ratio_range=(0.017, 0.067),  # 200-800 samples for 12000 context
            background_purity="mixed",
            task_specific={
                "min_global_states": 2,
                "max_global_states": 5,
            },
        )

        for i in range(15):
            rng = np.random.default_rng(1400 + i)
            sample = state_query_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.needles:
                config = sample.difficulty_config

                # Check if global_activity matches answer
                if "global_activity" in config:
                    # The answer should be the global state where needle was inserted
                    assert sample.answer.lower() in config.get("global_activity", "").lower() or \
                           config.get("global_activity", "").lower() in sample.answer.lower()

                print(f"  Sample {i}: Needle in '{config.get('global_activity', 'unknown')}' state")
                return

        pytest.skip("Could not generate suitable sample")


class TestStateQueryVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_state_query_sample(self, state_query_generator):
        """Visualize state query with global states and needle."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("state_query")

        difficulty = DifficultyConfig(
            context_length_samples=15000,
            needle_position="random",
            needle_length_ratio_range=(0.02, 0.08),  # 300-1200 samples for 15000 context
            background_purity="mixed",
            task_specific={
                "min_global_states": 2,
                "max_global_states": 4,
            },
        )

        # Find suitable sample
        sample = None
        for i in range(40):
            rng = np.random.default_rng(1500 + i)
            candidate = state_query_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.needles:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        needle = sample.needles[0]
        config = sample.difficulty_config

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1, 1, 0.7]})
        fig.suptitle("State Query Task - Cross-Scale Integration", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        # Get global timeline if available
        global_timeline = config.get("global_timeline", [])

        # Assign colors to states
        state_colors = {}
        color_palette = plt.cm.Set3(np.linspace(0, 1, 10))
        for i, (_, _, activity) in enumerate(global_timeline):
            if activity not in state_colors:
                state_colors[activity] = color_palette[len(state_colors) % 10]

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Show global states as background colors
            for start_frac, end_frac, activity in global_timeline:
                start_idx = int(start_frac * len(sample.x))
                end_idx = int(end_frac * len(sample.x))
                ax.axvspan(start_idx, end_idx, alpha=0.15, color=state_colors.get(activity, "gray"))

            # Highlight needle with red
            start = needle.insert_position_samples
            end = start + needle.duration_samples
            ax.axvspan(start, end, alpha=0.4, color="red", label=f"Needle: {needle.activity}")
            ax.axvline(start, color="red", linestyle="--", alpha=0.7)
            ax.axvline(end, color="red", linestyle="--", alpha=0.7)

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Legend for states
        if global_timeline:
            patches = [mpatches.Patch(color=state_colors[act], alpha=0.3, label=act)
                       for act in sorted(state_colors.keys())]
            patches.append(mpatches.Patch(color="red", alpha=0.4, label=f"Needle: {needle.activity}"))
            axes[0].legend(handles=patches, loc="upper right", fontsize=8, ncol=2)

        # Text panel
        axes[3].axis("off")

        global_state = config.get("global_activity", sample.answer)
        text = (
            f"Background: {sample.background_pid}  |  "
            f"Time: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Needle: {needle.activity} (local event)\n"
            f"Global state at needle position: {global_state}\n"
            f"(Model must identify GLOBAL state, not local needle activity)"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "state_query_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
