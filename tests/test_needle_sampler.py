# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for NeedleSampler module.

These tests use actual Capture24 data and Phase 1 artifacts.
Requires:
- Capture24 sensor data extracted
- Phase 1 artifacts built (timelines, bout index, transition matrix)
"""

from pathlib import Path

import numpy as np
import pytest

from ts_haystack.core import (
    BoutIndexer,
    NeedleSample,
    TransitionMatrix,
)
from ts_haystack.core.needle_sampler import NeedleSampler


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def bout_index():
    """Load the actual bout index."""
    return BoutIndexer.load_index()


@pytest.fixture(scope="module")
def transition_matrix():
    """Load the actual transition matrix."""
    return TransitionMatrix.load()


@pytest.fixture(scope="module")
def needle_sampler(bout_index, transition_matrix):
    """Create a NeedleSampler with actual data."""
    return NeedleSampler(bout_index, transition_matrix, source_hz=100)


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
# Test Basic Sampling
# =============================================================================


class TestBasicSampling:
    """Tests for basic needle sampling functionality."""

    def test_sample_needle_walking(self, needle_sampler, rng):
        """Test sampling a walking needle."""
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=5000,
            rng=rng,
        )

        assert needle is not None
        assert isinstance(needle, NeedleSample)
        assert needle.activity == "walking"
        assert needle.duration_ms >= 5000
        assert len(needle.x) > 0
        assert len(needle.y) == len(needle.x)
        assert len(needle.z) == len(needle.x)

    def test_sample_needle_all_activities(self, needle_sampler, rng):
        """Test sampling needles for all available activities."""
        activities = needle_sampler.get_available_activities()

        for activity in activities:
            needle = needle_sampler.sample_needle(
                activity=activity,
                min_duration_ms=1000,  # Low threshold to ensure availability
                rng=rng,
            )

            if needle is not None:
                assert needle.activity == activity
                assert len(needle.x) > 0
                print(f"  {activity}: {needle.n_samples} samples, {needle.duration_ms}ms")

    def test_sample_needle_with_duration_constraint(self, needle_sampler, rng):
        """Test that duration constraints are respected."""
        min_duration = 10000  # 10 seconds

        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=min_duration,
            rng=rng,
        )

        if needle is not None:
            assert needle.duration_ms >= min_duration

    def test_sample_needle_excludes_participant(self, needle_sampler, rng):
        """Test that excluded participants are respected."""
        # Get a participant that has walking bouts
        bouts = needle_sampler.bout_index.get_bouts_for_activity("walking")
        if not bouts:
            pytest.skip("No walking bouts available")

        # Get all unique participants with walking
        pids_with_walking = set(b.pid for b in bouts)
        exclude_pid = list(pids_with_walking)[0]

        # Sample with exclusion
        for _ in range(10):
            needle = needle_sampler.sample_needle(
                activity="walking",
                min_duration_ms=1000,
                exclude_pids={exclude_pid},
                rng=rng,
            )

            if needle is not None:
                assert needle.source_pid != exclude_pid

    def test_sample_returns_none_for_impossible_constraints(self, needle_sampler, rng):
        """Test that sampling returns None when constraints can't be met."""
        # Request impossibly long duration
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=999999999,  # ~11.5 days
            rng=rng,
        )

        assert needle is None


# =============================================================================
# Test Context-Aware Sampling
# =============================================================================


class TestContextAwareSampling:
    """Tests for context-aware needle sampling."""

    def test_sample_for_context_excludes_present_activities(
        self, needle_sampler, rng
    ):
        """Test that sampled needle activity is not in context."""
        context_activities = {"sitting", "standing"}

        for _ in range(10):
            needle = needle_sampler.sample_needle_for_context(
                context_activities=context_activities,
                min_duration_ms=5000,
                use_transition_probs=False,
                rng=rng,
            )

            if needle is not None:
                assert needle.activity not in context_activities

    def test_sample_for_context_with_transition_probs(self, needle_sampler, rng):
        """Test sampling with transition probability weighting."""
        context_activities = {"sitting"}

        needles_sampled = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            needle = needle_sampler.sample_needle_for_context(
                context_activities=context_activities,
                min_duration_ms=1000,
                use_transition_probs=True,
                rng=rng,
            )

            if needle is not None:
                needles_sampled.append(needle.activity)
                assert needle.activity not in context_activities

        # Should have some diversity in sampled activities
        if needles_sampled:
            unique_activities = set(needles_sampled)
            print(f"  Sampled activities: {unique_activities}")
            print(f"  Distribution: {[(a, needles_sampled.count(a)) for a in unique_activities]}")

    def test_sample_for_context_uniform(self, needle_sampler, rng):
        """Test uniform sampling without transition weighting."""
        context_activities = {"sleep"}

        needles_sampled = []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            needle = needle_sampler.sample_needle_for_context(
                context_activities=context_activities,
                min_duration_ms=1000,
                use_transition_probs=False,
                rng=rng,
            )

            if needle is not None:
                needles_sampled.append(needle.activity)

        if needles_sampled:
            unique_activities = set(needles_sampled)
            print(f"  Uniform sampled activities: {unique_activities}")


# =============================================================================
# Test Needle Trimming
# =============================================================================


class TestNeedleTrimming:
    """Tests for NeedleSample.trim() method."""

    def test_trim_reduces_length(self, needle_sampler, rng):
        """Test that trimming reduces needle length."""
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=10000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        original_length = needle.n_samples
        target_length = original_length // 2

        trimmed = needle.trim(target_length)

        assert trimmed.n_samples == target_length
        assert len(trimmed.x) == target_length
        assert len(trimmed.y) == target_length
        assert len(trimmed.z) == target_length

    def test_trim_preserves_center(self, needle_sampler, rng):
        """Test that trimming preserves the center of the needle."""
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=10000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        original_length = needle.n_samples
        target_length = original_length // 2

        trimmed = needle.trim(target_length)

        # Check that trimmed data comes from center
        start_idx = (original_length - target_length) // 2
        end_idx = start_idx + target_length

        np.testing.assert_array_equal(trimmed.x, needle.x[start_idx:end_idx])
        np.testing.assert_array_equal(trimmed.y, needle.y[start_idx:end_idx])
        np.testing.assert_array_equal(trimmed.z, needle.z[start_idx:end_idx])

    def test_trim_updates_timestamps(self, needle_sampler, rng):
        """Test that trimming updates timestamps correctly."""
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=10000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        original_duration = needle.duration_ms
        target_length = needle.n_samples // 2

        trimmed = needle.trim(target_length)

        # Duration should be approximately halved
        assert trimmed.duration_ms < original_duration
        assert trimmed.duration_ms > 0

    def test_trim_no_op_for_larger_target(self, needle_sampler, rng):
        """Test that trim is no-op when target > current length."""
        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=5000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        original_length = needle.n_samples
        trimmed = needle.trim(original_length * 2)

        assert trimmed.n_samples == original_length


# =============================================================================
# Test Activity Stats
# =============================================================================


class TestActivityStats:
    """Tests for activity statistics retrieval."""

    def test_get_available_activities(self, needle_sampler):
        """Test getting list of available activities."""
        activities = needle_sampler.get_available_activities()

        assert isinstance(activities, list)
        assert len(activities) > 0
        print(f"  Available activities: {activities}")

    def test_get_activity_stats(self, needle_sampler):
        """Test getting statistics for each activity."""
        activities = needle_sampler.get_available_activities()

        for activity in activities:
            stats = needle_sampler.get_activity_stats(activity)

            if stats is not None:
                assert "count" in stats
                assert "mean_duration_ms" in stats
                assert "min_duration_ms" in stats
                assert "max_duration_ms" in stats
                print(
                    f"  {activity}: count={stats['count']}, "
                    f"mean={stats['mean_duration_ms']:.0f}ms, "
                    f"range=[{stats['min_duration_ms']}-{stats['max_duration_ms']}]ms"
                )

    def test_count_available_bouts(self, needle_sampler):
        """Test counting available bouts with various filters."""
        activity = "walking"

        # Count all
        count_all = needle_sampler.count_available_bouts(activity)
        print(f"  All walking bouts: {count_all}")

        # Count with duration filter
        count_5s = needle_sampler.count_available_bouts(activity, min_duration_ms=5000)
        count_30s = needle_sampler.count_available_bouts(activity, min_duration_ms=30000)
        count_60s = needle_sampler.count_available_bouts(activity, min_duration_ms=60000)

        print(f"  Walking bouts >= 5s: {count_5s}")
        print(f"  Walking bouts >= 30s: {count_30s}")
        print(f"  Walking bouts >= 60s: {count_60s}")

        assert count_all >= count_5s >= count_30s >= count_60s


# =============================================================================
# Test Caching
# =============================================================================


class TestCaching:
    """Tests for sensor data caching."""

    def test_cache_populated_after_sampling(self, needle_sampler, rng):
        """Test that cache is populated after sampling."""
        needle_sampler.clear_cache()
        assert len(needle_sampler.get_cached_participants()) == 0

        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=5000,
            rng=rng,
        )

        if needle is not None:
            cached = needle_sampler.get_cached_participants()
            assert needle.source_pid in cached

    def test_clear_cache(self, needle_sampler):
        """Test cache clearing."""
        needle_sampler.clear_cache()
        assert len(needle_sampler.get_cached_participants()) == 0


# =============================================================================
# Test Determinism
# =============================================================================


class TestDeterminism:
    """Tests for reproducible sampling."""

    def test_same_seed_same_needle(self, needle_sampler):
        """Test that same seed produces same needle."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        needle1 = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=5000,
            rng=rng1,
        )

        needle2 = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=5000,
            rng=rng2,
        )

        if needle1 is not None and needle2 is not None:
            assert needle1.source_pid == needle2.source_pid
            assert needle1.start_ms == needle2.start_ms
            assert needle1.end_ms == needle2.end_ms
            np.testing.assert_array_equal(needle1.x, needle2.x)

    def test_different_seeds_can_produce_different_needles(self, needle_sampler):
        """Test that different seeds can produce different needles."""
        needles = []

        for seed in range(20):
            rng = np.random.default_rng(seed)
            needle = needle_sampler.sample_needle(
                activity="walking",
                min_duration_ms=5000,
                rng=rng,
            )
            if needle is not None:
                needles.append((needle.source_pid, needle.start_ms))

        # Should see some variety
        unique_needles = set(needles)
        assert len(unique_needles) > 1, "Expected different needles from different seeds"


# =============================================================================
# Visualization Tests
# =============================================================================


class TestVisualization:
    """Tests that generate visualization plots for manual inspection."""

    def test_visualize_sampled_needles(self, needle_sampler, rng):
        """Visualize sampled needles for different activities."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        activities = needle_sampler.get_available_activities()[:6]  # Limit to 6

        fig, axes = plt.subplots(len(activities), 3, figsize=(15, 3 * len(activities)))
        fig.suptitle("Sampled Needles by Activity (X, Y, Z)", fontsize=14)

        for i, activity in enumerate(activities):
            needle = needle_sampler.sample_needle(
                activity=activity,
                min_duration_ms=5000,
                rng=np.random.default_rng(42 + i),
            )

            if needle is None:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, "No data", ha="center", va="center")
                    axes[i, j].set_title(f"{activity} - {'XYZ'[j]}")
                continue

            axes[i, 0].plot(needle.x, linewidth=0.5, color="blue")
            axes[i, 0].set_title(f"{activity} - X ({needle.n_samples} samples)")
            axes[i, 0].set_ylabel("Accel (g)")

            axes[i, 1].plot(needle.y, linewidth=0.5, color="green")
            axes[i, 1].set_title(f"{activity} - Y")

            axes[i, 2].plot(needle.z, linewidth=0.5, color="orange")
            axes[i, 2].set_title(f"{activity} - Z")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "needle_sampler_activities.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")

    def test_visualize_needle_trimming(self, needle_sampler, rng):
        """Visualize needle trimming effect."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        ensure_plot_dir()

        needle = needle_sampler.sample_needle(
            activity="walking",
            min_duration_ms=20000,
            rng=rng,
        )

        if needle is None:
            pytest.skip("No suitable needle found")

        # Trim to different lengths
        trim_fractions = [1.0, 0.75, 0.5, 0.25]

        fig, axes = plt.subplots(len(trim_fractions), 3, figsize=(15, 3 * len(trim_fractions)))
        fig.suptitle("Needle Trimming Effect (walking activity)", fontsize=14)

        for i, frac in enumerate(trim_fractions):
            target_len = int(needle.n_samples * frac)
            trimmed = needle.trim(target_len)

            axes[i, 0].plot(trimmed.x, linewidth=0.5, color="blue")
            axes[i, 0].set_title(f"X - {frac:.0%} ({trimmed.n_samples} samples)")
            axes[i, 0].set_ylabel("Accel (g)")

            axes[i, 1].plot(trimmed.y, linewidth=0.5, color="green")
            axes[i, 1].set_title(f"Y - {frac:.0%}")

            axes[i, 2].plot(trimmed.z, linewidth=0.5, color="orange")
            axes[i, 2].set_title(f"Z - {frac:.0%}")

        plt.tight_layout()
        output_path = PLOT_OUTPUT_DIR / "needle_sampler_trimming.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {output_path}")
