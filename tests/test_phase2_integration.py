# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Integration tests for Phase 2 components.

Tests the full pipeline: Background sampling -> Needle sampling -> Style transfer -> Insertion.

Requires:
- Capture24 sensor data extracted
- Phase 1 artifacts built (timelines, bout index, transition matrix)
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import (
    BackgroundSampler,
    BoutIndexer,
    NeedleSampler,
    PromptTemplateBank,
    SeedManager,
    StyleTransfer,
    TimelineBuilder,
    TransitionMatrix,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def timelines():
    """Load all participant timelines."""
    return TimelineBuilder.load_all_timelines()


@pytest.fixture(scope="module")
def bout_index():
    """Load the bout index."""
    return BoutIndexer.load_index()


@pytest.fixture(scope="module")
def transition_matrix():
    """Load the transition matrix."""
    return TransitionMatrix.load()


@pytest.fixture(scope="module")
def background_sampler(timelines, bout_index):
    """Create BackgroundSampler."""
    return BackgroundSampler(timelines, bout_index, source_hz=100)


@pytest.fixture(scope="module")
def needle_sampler(bout_index, transition_matrix):
    """Create NeedleSampler."""
    return NeedleSampler(bout_index, transition_matrix, source_hz=100)


@pytest.fixture(scope="module")
def style_transfer():
    """Create StyleTransfer."""
    return StyleTransfer(blend_mode="cosine", blend_window_samples=50)


@pytest.fixture(scope="module")
def template_bank():
    """Create PromptTemplateBank."""
    return PromptTemplateBank()


@pytest.fixture(scope="module")
def seed_manager():
    """Create SeedManager."""
    return SeedManager(master_seed=42)


# =============================================================================
# Plot output directory
# =============================================================================

PLOT_OUTPUT_DIR = Path(__file__).parent / "plots"


def ensure_plot_dir():
    """Create plot output directory if it doesn't exist."""
    PLOT_OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for the complete Phase 2 pipeline."""

    def test_existence_task_pipeline(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
        seed_manager,
    ):
        """
        Test existence task pipeline: sample background, check if needle needed,
        optionally insert needle, generate Q/A.
        """
        rng = seed_manager.get_sample_rng("existence", 10000, "test", 0)

        # Sample pure background
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )
        assert background is not None

        # Decide positive/negative (existence)
        is_positive = rng.random() < 0.5

        if is_positive:
            # Positive case: Insert a needle of a different activity
            needle = needle_sampler.sample_needle_for_context(
                context_activities=background.activities_present,
                min_duration_ms=3000,
                rng=rng,
            )

            if needle is not None:
                target_activity = needle.activity

                # Trim needle to fit
                trimmed = needle.trim(min(needle.n_samples, 500))

                # Apply style transfer
                local_stats = style_transfer.compute_local_statistics(
                    (background.x, background.y, background.z),
                    position=5000,
                )
                transferred = style_transfer.transfer(trimmed, local_stats)

                # Insert
                final_x, final_y, final_z = style_transfer.insert_with_blending(
                    (background.x, background.y, background.z),
                    (transferred.x, transferred.y, transferred.z),
                    position=5000,
                )

                assert len(final_x) == len(background.x)
            else:
                # Fall back to activity in background
                target_activity = list(background.activities_present)[0]
        else:
            # Negative case: Ask about activity not in background
            all_activities = set(needle_sampler.get_available_activities())
            absent_activities = all_activities - background.activities_present
            target_activity = list(absent_activities)[0] if absent_activities else "unknown"

        # Generate Q/A
        question, answer = template_bank.sample(
            "existence",
            rng,
            activity=target_activity,
            exists=is_positive,
        )

        assert len(question) > 0
        assert len(answer) > 0
        assert target_activity in question
        print(f"  Q: {question}")
        print(f"  A: {answer}")

    def test_localization_task_pipeline(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
        seed_manager,
    ):
        """
        Test localization task pipeline: sample background, insert needle,
        record position, generate Q/A with timestamps.
        """
        rng = seed_manager.get_sample_rng("localization", 10000, "test", 0)

        # Sample pure background
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )
        assert background is not None

        # Sample needle from different activity
        needle = needle_sampler.sample_needle_for_context(
            context_activities=background.activities_present,
            min_duration_ms=3000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        # Trim and insert
        trimmed = needle.trim(min(needle.n_samples, 500))
        position = rng.integers(1000, 8000)

        local_stats = style_transfer.compute_local_statistics(
            (background.x, background.y, background.z),
            position=position,
        )
        transferred = style_transfer.transfer(trimmed, local_stats)

        final_x, final_y, final_z = style_transfer.insert_with_blending(
            (background.x, background.y, background.z),
            (transferred.x, transferred.y, transferred.z),
            position=position,
        )

        # Compute timestamps (simplified)
        start_time = background.recording_time_context[0]
        end_time = background.recording_time_context[1]

        # Generate Q/A
        question, answer = template_bank.sample(
            "localization",
            rng,
            activity=needle.activity,
            start=start_time,
            end=end_time,
        )

        assert len(question) > 0
        assert len(answer) > 0
        assert needle.activity in question
        print(f"  Q: {question}")
        print(f"  A: {answer}")

    def test_counting_task_pipeline(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
        seed_manager,
    ):
        """
        Test counting task pipeline: sample background, insert multiple needles,
        count them, generate Q/A.
        """
        rng = seed_manager.get_sample_rng("counting", 50000, "test", 0)

        # Sample larger background to fit multiple needles
        background = background_sampler.sample_background(
            context_length_samples=50000,
            purity="pure",
            rng=rng,
        )
        assert background is not None

        # Determine target activity
        all_activities = set(needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present

        if not candidate_activities:
            pytest.skip("No candidate activities for counting")

        target_activity = list(candidate_activities)[0]

        # Insert N needles
        n_needles = rng.integers(2, 5)
        needle_length = 500
        min_gap = 100

        current_signal = (background.x.copy(), background.y.copy(), background.z.copy())
        inserted_count = 0
        occupied_ranges = []

        for i in range(n_needles):
            # Sample needle
            needle = needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=3000,
                exclude_pids={background.pid},
                rng=rng,
            )

            if needle is None:
                continue

            trimmed = needle.trim(needle_length)

            # Find valid position
            position = None
            for _ in range(50):
                candidate = rng.integers(0, len(current_signal[0]) - needle_length)

                # Check conflicts
                valid = True
                for occ_start, occ_end in occupied_ranges:
                    if not (candidate + needle_length + min_gap <= occ_start or candidate >= occ_end + min_gap):
                        valid = False
                        break

                if valid:
                    position = candidate
                    break

            if position is None:
                continue

            # Apply style transfer and insert
            local_stats = style_transfer.compute_local_statistics(
                current_signal, position=position
            )
            transferred = style_transfer.transfer(trimmed, local_stats)

            current_signal = style_transfer.insert_with_blending(
                current_signal,
                (transferred.x, transferred.y, transferred.z),
                position=position,
            )

            occupied_ranges.append((position, position + needle_length))
            inserted_count += 1

        # Generate Q/A
        question, answer = template_bank.sample(
            "counting",
            rng,
            activity=target_activity,
            count=inserted_count,
        )

        assert len(question) > 0
        assert len(answer) > 0
        assert target_activity in question
        assert str(inserted_count) in answer
        print(f"  Inserted {inserted_count} {target_activity} bouts")
        print(f"  Q: {question}")
        print(f"  A: {answer}")

    def test_ordering_task_pipeline(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
        seed_manager,
    ):
        """
        Test ordering task pipeline: insert two activities, record order,
        generate Q/A about temporal ordering.
        """
        rng = seed_manager.get_sample_rng("ordering", 20000, "test", 0)

        # Get two distinct activities
        all_activities = list(needle_sampler.get_available_activities())
        if len(all_activities) < 2:
            pytest.skip("Need at least 2 activities")

        activity_a, activity_b = rng.choice(all_activities, size=2, replace=False)

        # Sample background excluding both activities
        background = background_sampler.sample_background(
            context_length_samples=20000,
            purity="pure",
            excluded_activities={activity_a, activity_b},
            rng=rng,
        )

        if background is None:
            pytest.skip("Could not sample suitable background")

        # Sample both needles
        needle_a = needle_sampler.sample_needle(
            activity=activity_a, min_duration_ms=3000, rng=rng
        )
        needle_b = needle_sampler.sample_needle(
            activity=activity_b, min_duration_ms=3000, rng=rng
        )

        if needle_a is None or needle_b is None:
            pytest.skip("Could not sample needles")

        # Trim
        trimmed_a = needle_a.trim(400)
        trimmed_b = needle_b.trim(400)

        # Decide order
        a_first = rng.random() < 0.5

        if a_first:
            pos_a, pos_b = 2000, 10000
            first_activity, second_activity = activity_a, activity_b
        else:
            pos_a, pos_b = 10000, 2000
            first_activity, second_activity = activity_b, activity_a

        # Insert both
        current_signal = (background.x.copy(), background.y.copy(), background.z.copy())

        for needle, pos in sorted([(trimmed_a, pos_a), (trimmed_b, pos_b)], key=lambda x: x[1]):
            local_stats = style_transfer.compute_local_statistics(current_signal, position=pos)
            transferred = style_transfer.transfer(needle, local_stats)
            current_signal = style_transfer.insert_with_blending(
                current_signal,
                (transferred.x, transferred.y, transferred.z),
                position=pos,
            )

        # Generate Q/A
        question, answer = template_bank.sample(
            "ordering",
            rng,
            activity_a=activity_a,
            activity_b=activity_b,
            a_before_b=a_first,
            first_activity=first_activity,
            second_activity=second_activity,
        )

        assert len(question) > 0
        assert len(answer) > 0
        print(f"  Order: {first_activity} -> {second_activity}")
        print(f"  Q: {question}")
        print(f"  A: {answer}")


# =============================================================================
# Visualization Tests
# =============================================================================


class TestPipelineVisualization:
    """Generate comprehensive visualizations of the full pipeline."""

    def test_visualize_full_pipeline(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        seed_manager,
    ):
        """
        Generate a comprehensive visualization of the full Phase 2 pipeline.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        rng = seed_manager.get_sample_rng("visualization", 10000, "test", 0)

        # Step 1: Sample background
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )
        assert background is not None

        # Step 2: Sample needle
        needle = needle_sampler.sample_needle_for_context(
            context_activities=background.activities_present,
            min_duration_ms=5000,
            rng=rng,
        )
        if needle is None:
            pytest.skip("No suitable needle")

        # Step 3: Trim needle
        trimmed = needle.trim(500)

        # Step 4: Compute statistics
        position = 5000
        needle_stats = style_transfer.compute_statistics(
            trimmed.x, trimmed.y, trimmed.z
        )
        local_stats = style_transfer.compute_local_statistics(
            (background.x, background.y, background.z),
            position=position,
        )

        # Step 5: Apply style transfer
        transferred = style_transfer.transfer(trimmed, local_stats)
        transferred_stats = style_transfer.compute_statistics(
            transferred.x, transferred.y, transferred.z
        )

        # Step 6: Insert with blending
        final_x, final_y, final_z = style_transfer.insert_with_blending(
            (background.x, background.y, background.z),
            (transferred.x, transferred.y, transferred.z),
            position=position,
        )

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle(
            f"Phase 2 Full Pipeline Visualization\n"
            f"Background: {list(background.activities_present)[0]} ({background.pid}) | "
            f"Needle: {needle.activity} ({needle.source_pid})",
            fontsize=14,
        )

        # Row 1: Original Background
        ax1 = fig.add_subplot(6, 3, 1)
        ax1.plot(background.x, linewidth=0.5, color="blue", alpha=0.7)
        ax1.set_title(f"1. Background X ({list(background.activities_present)[0]})")
        ax1.set_ylabel("Accel (g)")
        ax1.axvline(position, color="red", linestyle="--", alpha=0.5, label="Insert pos")
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(6, 3, 2)
        ax2.plot(background.y, linewidth=0.5, color="green", alpha=0.7)
        ax2.set_title("1. Background Y")
        ax2.axvline(position, color="red", linestyle="--", alpha=0.5)

        ax3 = fig.add_subplot(6, 3, 3)
        ax3.plot(background.z, linewidth=0.5, color="orange", alpha=0.7)
        ax3.set_title("1. Background Z")
        ax3.axvline(position, color="red", linestyle="--", alpha=0.5)

        # Row 2: Original Needle
        ax4 = fig.add_subplot(6, 3, 4)
        ax4.plot(needle.x, linewidth=0.5, color="blue")
        ax4.set_title(f"2. Original Needle X ({needle.activity}, {needle.n_samples} samples)")
        ax4.set_ylabel("Accel (g)")

        ax5 = fig.add_subplot(6, 3, 5)
        ax5.plot(needle.y, linewidth=0.5, color="green")
        ax5.set_title("2. Original Needle Y")

        ax6 = fig.add_subplot(6, 3, 6)
        ax6.plot(needle.z, linewidth=0.5, color="orange")
        ax6.set_title("2. Original Needle Z")

        # Row 3: Trimmed Needle
        ax7 = fig.add_subplot(6, 3, 7)
        ax7.plot(trimmed.x, linewidth=0.5, color="blue")
        ax7.set_title(f"3. Trimmed Needle X ({trimmed.n_samples} samples)")
        ax7.set_ylabel("Accel (g)")

        ax8 = fig.add_subplot(6, 3, 8)
        ax8.plot(trimmed.y, linewidth=0.5, color="green")
        ax8.set_title("3. Trimmed Needle Y")

        ax9 = fig.add_subplot(6, 3, 9)
        ax9.plot(trimmed.z, linewidth=0.5, color="orange")
        ax9.set_title("3. Trimmed Needle Z")

        # Row 4: Transferred Needle
        ax10 = fig.add_subplot(6, 3, 10)
        ax10.plot(transferred.x, linewidth=0.5, color="blue")
        ax10.set_title("4. Transferred Needle X (style-matched)")
        ax10.set_ylabel("Accel (g)")

        ax11 = fig.add_subplot(6, 3, 11)
        ax11.plot(transferred.y, linewidth=0.5, color="green")
        ax11.set_title("4. Transferred Needle Y")

        ax12 = fig.add_subplot(6, 3, 12)
        ax12.plot(transferred.z, linewidth=0.5, color="orange")
        ax12.set_title("4. Transferred Needle Z")

        # Row 5: Final Signal (zoomed)
        margin = 300
        start_idx = max(0, position - margin)
        end_idx = min(len(final_x), position + trimmed.n_samples + margin)

        ax13 = fig.add_subplot(6, 3, 13)
        ax13.plot(range(start_idx, end_idx), final_x[start_idx:end_idx], linewidth=0.5, color="blue")
        ax13.axvline(position, color="red", linestyle="--", alpha=0.7)
        ax13.axvline(position + trimmed.n_samples, color="red", linestyle="--", alpha=0.7)
        ax13.axvspan(position, position + trimmed.n_samples, alpha=0.1, color="red")
        ax13.set_title("5. Final Signal X (zoomed)")
        ax13.set_ylabel("Accel (g)")

        ax14 = fig.add_subplot(6, 3, 14)
        ax14.plot(range(start_idx, end_idx), final_y[start_idx:end_idx], linewidth=0.5, color="green")
        ax14.axvline(position, color="red", linestyle="--", alpha=0.7)
        ax14.axvline(position + trimmed.n_samples, color="red", linestyle="--", alpha=0.7)
        ax14.axvspan(position, position + trimmed.n_samples, alpha=0.1, color="red")
        ax14.set_title("5. Final Signal Y (zoomed)")

        ax15 = fig.add_subplot(6, 3, 15)
        ax15.plot(range(start_idx, end_idx), final_z[start_idx:end_idx], linewidth=0.5, color="orange")
        ax15.axvline(position, color="red", linestyle="--", alpha=0.7)
        ax15.axvline(position + trimmed.n_samples, color="red", linestyle="--", alpha=0.7)
        ax15.axvspan(position, position + trimmed.n_samples, alpha=0.1, color="red")
        ax15.set_title("5. Final Signal Z (zoomed)")

        # Row 6: Statistics comparison
        ax16 = fig.add_subplot(6, 3, 16)
        x_pos = np.arange(3)
        width = 0.25
        ax16.bar(x_pos - width, needle_stats.mean, width, label="Original", color="gray", alpha=0.7)
        ax16.bar(x_pos, local_stats.mean, width, label="Target", color="red", alpha=0.7)
        ax16.bar(x_pos + width, transferred_stats.mean, width, label="Transferred", color="blue", alpha=0.7)
        ax16.set_xticks(x_pos)
        ax16.set_xticklabels(["X", "Y", "Z"])
        ax16.set_title("6. Mean Comparison")
        ax16.set_ylabel("Mean (g)")
        ax16.legend(fontsize=8)

        ax17 = fig.add_subplot(6, 3, 17)
        ax17.bar(x_pos - width, needle_stats.std, width, label="Original", color="gray", alpha=0.7)
        ax17.bar(x_pos, local_stats.std, width, label="Target", color="red", alpha=0.7)
        ax17.bar(x_pos + width, transferred_stats.std, width, label="Transferred", color="blue", alpha=0.7)
        ax17.set_xticks(x_pos)
        ax17.set_xticklabels(["X", "Y", "Z"])
        ax17.set_title("6. Std Comparison")
        ax17.set_ylabel("Std (g)")
        ax17.legend(fontsize=8)

        # Summary text
        ax18 = fig.add_subplot(6, 3, 18)
        ax18.axis("off")
        summary = (
            f"Pipeline Summary:\n\n"
            f"Background:\n"
            f"  Participant: {background.pid}\n"
            f"  Activity: {list(background.activities_present)[0]}\n"
            f"  Length: {background.n_samples} samples\n\n"
            f"Needle:\n"
            f"  Source: {needle.source_pid}\n"
            f"  Activity: {needle.activity}\n"
            f"  Original: {needle.n_samples} samples\n"
            f"  Trimmed: {trimmed.n_samples} samples\n\n"
            f"Insertion:\n"
            f"  Position: {position}\n"
            f"  Blend window: {style_transfer.blend_window_samples}\n"
            f"  Mode: {style_transfer.blend_mode}"
        )
        ax18.text(0.1, 0.9, summary, fontsize=10, family="monospace", verticalalignment="top")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "phase2_full_pipeline.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    def test_visualize_multiple_insertions(
        self,
        background_sampler,
        needle_sampler,
        style_transfer,
        seed_manager,
    ):
        """Visualize multiple needle insertions (for counting task)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        rng = seed_manager.get_sample_rng("counting_viz", 50000, "test", 0)

        # Sample large background
        background = background_sampler.sample_background(
            context_length_samples=50000,
            purity="pure",
            rng=rng,
        )
        assert background is not None

        # Get candidate activity
        all_activities = set(needle_sampler.get_available_activities())
        candidates = all_activities - background.activities_present
        if not candidates:
            pytest.skip("No candidate activities")

        target_activity = list(candidates)[0]

        # Insert multiple needles
        n_needles = 4
        needle_length = 500
        positions = [5000, 15000, 25000, 40000]

        current_signal = (background.x.copy(), background.y.copy(), background.z.copy())
        inserted_needles = []

        for pos in positions:
            needle = needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=3000,
                rng=rng,
            )
            if needle is None:
                continue

            trimmed = needle.trim(needle_length)
            local_stats = style_transfer.compute_local_statistics(current_signal, position=pos)
            transferred = style_transfer.transfer(trimmed, local_stats)

            current_signal = style_transfer.insert_with_blending(
                current_signal,
                (transferred.x, transferred.y, transferred.z),
                position=pos,
            )
            inserted_needles.append((pos, trimmed.n_samples))

        # Visualize
        fig, axes = plt.subplots(4, 1, figsize=(18, 12))
        fig.suptitle(
            f"Multiple Needle Insertions (Counting Task)\n"
            f"Background: {list(background.activities_present)[0]} | "
            f"Needles: {target_activity} (n={len(inserted_needles)})",
            fontsize=14,
        )

        # Original background
        axes[0].plot(background.x, linewidth=0.3, color="gray", alpha=0.7, label="Original")
        axes[0].set_title("Original Background X")
        axes[0].set_ylabel("Accel (g)")

        # Final signal with annotations
        axes[1].plot(current_signal[0], linewidth=0.3, color="blue", alpha=0.7)
        for pos, length in inserted_needles:
            axes[1].axvspan(pos, pos + length, alpha=0.3, color="red")
            axes[1].axvline(pos, color="red", linestyle="--", alpha=0.5)
        axes[1].set_title(f"Final Signal X with {len(inserted_needles)} {target_activity} insertions")
        axes[1].set_ylabel("Accel (g)")

        axes[2].plot(current_signal[1], linewidth=0.3, color="green", alpha=0.7)
        for pos, length in inserted_needles:
            axes[2].axvspan(pos, pos + length, alpha=0.3, color="red")
        axes[2].set_title("Final Signal Y")
        axes[2].set_ylabel("Accel (g)")

        axes[3].plot(current_signal[2], linewidth=0.3, color="orange", alpha=0.7)
        for pos, length in inserted_needles:
            axes[3].axvspan(pos, pos + length, alpha=0.3, color="red")
        axes[3].set_title("Final Signal Z")
        axes[3].set_ylabel("Accel (g)")
        axes[3].set_xlabel("Sample index")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "phase2_multiple_insertions.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")
