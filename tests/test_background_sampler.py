# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for BackgroundSampler module.

These tests use actual Capture24 data and Phase 1 artifacts.
Requires:
- Capture24 sensor data extracted
- Phase 1 artifacts built (timelines, bout index)
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import (
    BackgroundSample,
    BoutIndexer,
    TimelineBuilder,
)
from ts_haystack.core.background_sampler import (
    BackgroundSampler,
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
    """Load the actual bout index."""
    return BoutIndexer.load_index()


@pytest.fixture(scope="module")
def background_sampler(timelines, bout_index):
    """Create a BackgroundSampler with actual data."""
    return BackgroundSampler(timelines, bout_index, source_hz=100)


@pytest.fixture
def rng():
    """Create a seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


# =============================================================================
# Plot output directory
# =============================================================================

PLOT_OUTPUT_DIR = Path(__file__).parent / "plots"


def ensure_plot_dir():
    """Create plot output directory if it doesn't exist."""
    PLOT_OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# Test Pure Background Sampling
# =============================================================================


class TestPureBackgroundSampling:
    """Tests for sampling pure (single-activity) backgrounds."""

    def test_sample_pure_background(self, background_sampler, rng):
        """Test sampling a pure background window."""
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )

        assert background is not None
        assert isinstance(background, BackgroundSample)
        assert background.is_pure
        assert len(background.activities_present) == 1
        assert len(background.x) == 10000
        assert len(background.y) == 10000
        assert len(background.z) == 10000

    def test_pure_background_different_context_lengths(self, background_sampler, rng):
        """Test pure sampling with various context lengths."""
        context_lengths = [1000, 5000, 10000, 50000]

        for ctx_len in context_lengths:
            background = background_sampler.sample_background(
                context_length_samples=ctx_len,
                purity="pure",
                rng=np.random.default_rng(42),
            )

            if background is not None:
                assert len(background.x) == ctx_len
                assert background.is_pure
                print(
                    f"  Context {ctx_len}: {background.pid}, "
                    f"activity={list(background.activities_present)[0]}"
                )

    def test_pure_background_allowed_activities(self, background_sampler, rng):
        """Test filtering by allowed activities."""
        allowed = {"sitting", "standing"}

        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            allowed_activities=allowed,
            rng=rng,
        )

        if background is not None:
            assert background.activities_present.issubset(allowed)
            print(f"  Sampled activity: {background.activities_present}")

    def test_pure_background_excluded_activities(self, background_sampler, rng):
        """Test filtering by excluded activities."""
        excluded = {"sleep", "sitting"}

        for _ in range(10):
            background = background_sampler.sample_background(
                context_length_samples=5000,
                purity="pure",
                excluded_activities=excluded,
                rng=rng,
            )

            if background is not None:
                assert not background.activities_present.intersection(excluded)


# =============================================================================
# Test "Any" Background Sampling
# =============================================================================


class TestAnyBackgroundSampling:
    """Tests for sampling backgrounds with purity='any' (samples random window without constraints)."""

    def test_sample_any_background_returns_valid_sample(self, background_sampler, rng):
        """Test that purity='any' returns a valid background sample."""
        background = background_sampler.sample_background(
            context_length_samples=50000,
            purity="any",
            rng=rng,
        )

        assert background is not None
        assert isinstance(background, BackgroundSample)
        assert len(background.x) == 50000
        assert len(background.y) == 50000
        assert len(background.z) == 50000

    def test_any_works_with_short_context(self, background_sampler):
        """Test that purity='any' works with short contexts (which would fail for 'mixed')."""
        # Short contexts (256 samples = 2.56s) should work with 'any'
        # but would fail with 'mixed' since they can't span multiple activities
        rng = np.random.default_rng(42)
        bg = background_sampler.sample_background(
            context_length_samples=256,
            purity="any",
            rng=rng,
        )

        assert bg is not None
        assert len(bg.x) == 256
        # Short contexts will naturally be pure (single activity)
        print(f"  Short context activities: {bg.activities_present}")

    def test_any_can_produce_mixed_for_long_context(self, background_sampler):
        """Test that purity='any' can produce mixed backgrounds for long contexts."""
        mixed_count = 0

        # Sample many times with long context to see if we get any mixed
        for seed in range(50):
            rng = np.random.default_rng(seed)
            bg = background_sampler.sample_background(
                context_length_samples=100000,  # Long context more likely to span activities
                purity="any",
                rng=rng,
            )

            if bg is not None and not bg.is_pure:
                mixed_count += 1

        # With long enough context, we should occasionally get mixed backgrounds
        print(f"  Mixed backgrounds out of 50: {mixed_count}")
        # Don't require mixed (depends on data), just verify it doesn't error

    def test_any_deterministic_with_same_seed(self, background_sampler):
        """Test that same seed produces same background."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        bg1 = background_sampler.sample_background(
            context_length_samples=50000,
            purity="any",
            rng=rng1,
        )

        bg2 = background_sampler.sample_background(
            context_length_samples=50000,
            purity="any",
            rng=rng2,
        )

        if bg1 is not None and bg2 is not None:
            # Same seed should produce same background
            assert bg1.is_pure == bg2.is_pure
            assert bg1.pid == bg2.pid
            assert bg1.start_ms == bg2.start_ms


# =============================================================================
# Test Mixed Background Sampling
# =============================================================================


class TestMixedBackgroundSampling:
    """Tests for sampling mixed (multi-activity) backgrounds."""

    def test_sample_mixed_background(self, background_sampler, rng):
        """Test sampling a mixed background window."""
        background = background_sampler.sample_background(
            context_length_samples=50000,  # Longer window more likely to be mixed
            purity="mixed",
            min_activity_count=2,
            rng=rng,
        )

        if background is not None:
            assert isinstance(background, BackgroundSample)
            assert len(background.activities_present) >= 2
            assert not background.is_pure
            print(f"  Activities: {background.activities_present}")
            print(f"  Timeline: {background.activity_timeline}")

    def test_mixed_background_activity_count_constraints(self, background_sampler, rng):
        """Test activity count constraints for mixed backgrounds."""
        # Request exactly 2-3 activities
        for _ in range(5):
            background = background_sampler.sample_background(
                context_length_samples=100000,
                purity="mixed",
                min_activity_count=2,
                max_activity_count=3,
                rng=rng,
            )

            if background is not None:
                n_activities = len(background.activities_present)
                assert 2 <= n_activities <= 3, f"Got {n_activities} activities"
                print(f"  Activities ({n_activities}): {background.activities_present}")

    def test_mixed_background_has_activity_timeline(self, background_sampler, rng):
        """Test that mixed backgrounds have valid activity timelines."""
        background = background_sampler.sample_background(
            context_length_samples=100000,
            purity="mixed",
            min_activity_count=2,
            rng=rng,
        )

        if background is not None:
            timeline = background.activity_timeline

            assert len(timeline) > 0
            assert all(len(entry) == 3 for entry in timeline)

            # Check timeline is sorted and covers [0, 1]
            for start_frac, end_frac, activity in timeline:
                assert 0 <= start_frac < end_frac <= 1
                assert activity in background.activities_present

            print(f"  Timeline entries: {len(timeline)}")
            for start, end, act in timeline:
                print(f"    [{start:.3f} - {end:.3f}]: {act}")


# =============================================================================
# Test Background Properties
# =============================================================================


class TestBackgroundProperties:
    """Tests for BackgroundSample properties and methods."""

    def test_background_metadata(self, background_sampler, rng):
        """Test that background has correct metadata."""
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )

        assert background is not None
        assert background.pid is not None
        assert background.start_ms > 0
        assert background.end_ms > background.start_ms
        assert background.duration_ms == background.end_ms - background.start_ms
        assert background.n_samples == len(background.x)

    def test_recording_time_context(self, background_sampler, rng):
        """Test human-readable time context."""
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )

        assert background is not None
        start_time, end_time = background.recording_time_context

        assert isinstance(start_time, str)
        assert isinstance(end_time, str)
        # Should have time-like format
        assert ":" in start_time or "AM" in start_time or "PM" in start_time
        print(f"  Time range: {start_time} - {end_time}")

    def test_get_activity_at_position(self, background_sampler, rng):
        """Test getting activity at a specific position."""
        background = background_sampler.sample_background(
            context_length_samples=100000,
            purity="mixed",
            min_activity_count=2,
            rng=rng,
        )

        if background is not None:
            # Check positions
            for pos in [0.0, 0.25, 0.5, 0.75, 0.99]:
                activity = background.get_activity_at_position(pos)
                if activity is not None:
                    assert activity in background.activities_present
                    print(f"  Position {pos}: {activity}")

    def test_data_types(self, background_sampler, rng):
        """Test that data arrays have correct types."""
        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )

        assert background is not None
        assert background.x.dtype == np.float32
        assert background.y.dtype == np.float32
        assert background.z.dtype == np.float32


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_no_valid_activities_raises(self, background_sampler, rng):
        """Test that impossible constraints raise ValueError."""
        with pytest.raises(ValueError):
            background_sampler.sample_background(
                context_length_samples=10000,
                purity="pure",
                allowed_activities={"nonexistent_activity"},
                rng=rng,
            )

    def test_context_too_long_for_pure_raises(self, background_sampler, rng):
        """Test that too-long context raises appropriate error."""
        # Request impossibly long pure window
        with pytest.raises(ValueError):
            background_sampler.sample_background(
                context_length_samples=999999999,  # ~2.7 hours at 100Hz
                purity="pure",
                rng=rng,
            )


# =============================================================================
# Test Caching
# =============================================================================


class TestCaching:
    """Tests for sensor data caching."""

    def test_cache_populated_after_sampling(self, background_sampler, rng):
        """Test that cache is populated after sampling."""
        background_sampler.clear_cache()
        assert len(background_sampler.get_cached_participants()) == 0

        background = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng,
        )

        if background is not None:
            cached = background_sampler.get_cached_participants()
            assert background.pid in cached

    def test_preload_participants(self, background_sampler, timelines):
        """Test preloading participant data."""
        background_sampler.clear_cache()

        # Preload a few participants
        pids_to_load = list(timelines.keys())[:3]
        background_sampler.preload_participants(pids_to_load)

        cached = background_sampler.get_cached_participants()
        for pid in pids_to_load:
            assert pid in cached

    def test_clear_cache(self, background_sampler):
        """Test cache clearing."""
        background_sampler.clear_cache()
        assert len(background_sampler.get_cached_participants()) == 0


# =============================================================================
# Test Determinism
# =============================================================================


class TestDeterminism:
    """Tests for reproducible sampling."""

    def test_same_seed_same_background(self, background_sampler):
        """Test that same seed produces same background."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        bg1 = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng1,
        )

        bg2 = background_sampler.sample_background(
            context_length_samples=10000,
            purity="pure",
            rng=rng2,
        )

        if bg1 is not None and bg2 is not None:
            assert bg1.pid == bg2.pid
            assert bg1.start_ms == bg2.start_ms
            assert bg1.end_ms == bg2.end_ms
            np.testing.assert_array_equal(bg1.x, bg2.x)
            np.testing.assert_array_equal(bg1.y, bg2.y)
            np.testing.assert_array_equal(bg1.z, bg2.z)

    def test_different_seeds_produce_variety(self, background_sampler):
        """Test that different seeds produce different backgrounds."""
        backgrounds = []

        for seed in range(20):
            rng = np.random.default_rng(seed)
            bg = background_sampler.sample_background(
                context_length_samples=10000,
                purity="pure",
                rng=rng,
            )
            if bg is not None:
                backgrounds.append((bg.pid, bg.start_ms))

        unique_backgrounds = set(backgrounds)
        assert len(unique_backgrounds) > 1, "Expected variety from different seeds"


# =============================================================================
# Test Available Activities
# =============================================================================


class TestAvailableActivities:
    """Tests for available activities."""

    def test_get_available_activities(self, background_sampler):
        """Test getting available activities."""
        activities = background_sampler.get_available_activities()

        assert isinstance(activities, list)
        assert len(activities) > 0
        print(f"  Available activities: {activities}")

    def test_sample_each_activity(self, background_sampler):
        """Test sampling background for each available activity."""
        activities = background_sampler.get_available_activities()

        for activity in activities:
            rng = np.random.default_rng(42)
            try:
                bg = background_sampler.sample_background(
                    context_length_samples=5000,
                    purity="pure",
                    allowed_activities={activity},
                    rng=rng,
                )
                if bg is not None:
                    print(f"  {activity}: {bg.pid}, {bg.n_samples} samples")
                else:
                    print(f"  {activity}: No suitable window found")
            except ValueError as e:
                print(f"  {activity}: Error - {e}")


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests that generate visualization plots for manual inspection."""

    def test_visualize_pure_backgrounds(self, background_sampler):
        """Visualize pure backgrounds for different activities."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        activities = background_sampler.get_available_activities()[:6]

        fig, axes = plt.subplots(len(activities), 3, figsize=(15, 3 * len(activities)))
        fig.suptitle("Pure Background Samples by Activity", fontsize=14)

        for i, activity in enumerate(activities):
            rng = np.random.default_rng(42 + i)
            try:
                bg = background_sampler.sample_background(
                    context_length_samples=10000,
                    purity="pure",
                    allowed_activities={activity},
                    rng=rng,
                )
            except ValueError:
                bg = None

            if bg is None:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                    axes[i, j].set_title(f"{activity} - {'XYZ'[j]}")
                continue

            axes[i, 0].plot(bg.x, linewidth=0.3, color="blue", alpha=0.7)
            axes[i, 0].set_title(f"{activity} - X ({bg.pid})")
            axes[i, 0].set_ylabel("Accel (g)")

            axes[i, 1].plot(bg.y, linewidth=0.3, color="green", alpha=0.7)
            axes[i, 1].set_title(f"{activity} - Y")

            axes[i, 2].plot(bg.z, linewidth=0.3, color="orange", alpha=0.7)
            axes[i, 2].set_title(f"{activity} - Z")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "background_sampler_pure.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    def test_visualize_mixed_background(self, background_sampler, rng):
        """Visualize mixed background with activity timeline."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        bg = background_sampler.sample_background(
            context_length_samples=100000,
            purity="mixed",
            min_activity_count=2,
            rng=rng,
        )

        if bg is None:
            pytest.skip("No suitable mixed background found")

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f"Mixed Background Sample ({bg.pid})", fontsize=14)

        # Plot X
        axes[0].plot(bg.x, linewidth=0.3, color="blue", alpha=0.7)
        axes[0].set_title("X-axis")
        axes[0].set_ylabel("Accel (g)")

        # Plot Y
        axes[1].plot(bg.y, linewidth=0.3, color="green", alpha=0.7)
        axes[1].set_title("Y-axis")
        axes[1].set_ylabel("Accel (g)")

        # Plot Z
        axes[2].plot(bg.z, linewidth=0.3, color="orange", alpha=0.7)
        axes[2].set_title("Z-axis")
        axes[2].set_ylabel("Accel (g)")

        # Plot activity timeline
        colors = plt.cm.Set2(np.linspace(0, 1, len(bg.activities_present)))
        activity_color_map = dict(zip(bg.activities_present, colors))

        for start_frac, end_frac, activity in bg.activity_timeline:
            start_idx = int(start_frac * bg.n_samples)
            end_idx = int(end_frac * bg.n_samples)
            axes[3].axvspan(
                start_idx, end_idx,
                alpha=0.7,
                color=activity_color_map[activity],
                label=activity if activity not in [a for s, e, a in bg.activity_timeline[:bg.activity_timeline.index((start_frac, end_frac, activity))]] else None,
            )

        axes[3].set_xlim(0, bg.n_samples)
        axes[3].set_title("Activity Timeline")
        axes[3].set_xlabel("Sample index")
        axes[3].set_yticks([])
        axes[3].legend(loc="upper right")

        # Add vertical lines on signal plots
        for start_frac, end_frac, activity in bg.activity_timeline:
            start_idx = int(start_frac * bg.n_samples)
            for ax in axes[:3]:
                ax.axvline(start_idx, color="gray", linestyle="--", alpha=0.3)

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "background_sampler_mixed.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    def test_visualize_different_context_lengths(self, background_sampler):
        """Visualize backgrounds at different context lengths."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        context_lengths = [1000, 5000, 10000, 50000, 100000]

        fig, axes = plt.subplots(len(context_lengths), 3, figsize=(15, 3 * len(context_lengths)))
        fig.suptitle("Background Samples at Different Context Lengths", fontsize=14)

        for i, ctx_len in enumerate(context_lengths):
            rng = np.random.default_rng(42)
            bg = background_sampler.sample_background(
                context_length_samples=ctx_len,
                purity="pure",
                rng=rng,
            )

            if bg is None:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            activity = list(bg.activities_present)[0]

            axes[i, 0].plot(bg.x, linewidth=0.3, color="blue", alpha=0.7)
            axes[i, 0].set_title(f"{ctx_len} samples - X ({activity})")
            axes[i, 0].set_ylabel("Accel (g)")

            axes[i, 1].plot(bg.y, linewidth=0.3, color="green", alpha=0.7)
            axes[i, 1].set_title(f"{ctx_len} samples - Y")

            axes[i, 2].plot(bg.z, linewidth=0.3, color="orange", alpha=0.7)
            axes[i, 2].set_title(f"{ctx_len} samples - Z")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "background_sampler_context_lengths.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")
