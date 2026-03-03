# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Needle sampler for TS-Haystack.

This module provides functionality to sample needle bouts from the
cross-participant bout index for insertion into background windows.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl

from ts_haystack.capture24.loader import (
    load_participant_sensor_data,
)
from ts_haystack.core.data_structures import (
    BoutIndex,
    BoutRef,
    NeedleSample,
)
from ts_haystack.core.transition_matrix import (
    TransitionMatrix,
)


class NeedleSampler:
    """
    Samples needle bouts from the cross-participant bout index.

    Uses bout-index-based sampling for efficiency:
    - Query bout index for bouts >= min_duration_ms
    - Sample a bout from that pool
    - Load sensor data and trim to desired length

    Only filters by min_duration_ms to maximize the candidate pool and
    reduce probability of sampling the same bout repeatedly.

    The sampler receives RNG from callers (task generators) following the seed
    strategy where SeedManager provides deterministic RNGs to task generators.

    Example:
        >>> bout_index = BoutIndexer.load_index()
        >>> transition_matrix = TransitionMatrix.load()
        >>> sampler = NeedleSampler(bout_index, transition_matrix, source_hz=100)
        >>> rng = np.random.default_rng(42)  # In practice, from SeedManager
        >>> needle = sampler.sample_needle(
        ...     activity="walking",
        ...     min_duration_ms=5000,
        ...     rng=rng,
        ... )
        >>> # Trim to exact length if needed
        >>> needle = needle.trim(n_samples=500)
    """

    def __init__(
        self,
        bout_index: BoutIndex,
        transition_matrix: Optional[TransitionMatrix] = None,
        source_hz: int = 100,
    ):
        """
        Initialize the needle sampler.

        Args:
            bout_index: Cross-participant bout index
            transition_matrix: Optional transition matrix for activity sampling
            source_hz: Source data sampling frequency in Hz
        """
        self.bout_index = bout_index
        self.transition_matrix = transition_matrix
        self.source_hz = source_hz

        # Cache for loaded sensor data (pid -> DataFrame)
        self._sensor_cache: Dict[str, pl.DataFrame] = {}

    def sample_needle(
        self,
        activity: str,
        min_duration_ms: int = 0,
        exclude_pids: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[NeedleSample]:
        """
        Sample a needle bout for the given activity.

        Only filters by min_duration_ms to maximize the candidate pool.
        Use NeedleSample.trim() to adjust to exact desired length after sampling.

        Args:
            activity: Target activity type
            min_duration_ms: Minimum bout duration in milliseconds
            exclude_pids: Participant IDs to exclude from sampling
            rng: Random generator (should come from SeedManager via task generator)

        Returns:
            NeedleSample with sensor data, or None if no matching bout found
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get candidate bouts from index (only filter by min duration)
        candidates = self.bout_index.get_bouts_for_activity(
            activity,
            min_duration_ms=min_duration_ms,
            exclude_pids=exclude_pids,
        )

        if not candidates:
            return None

        # Uniform random selection from all valid candidates
        bout_ref = candidates[rng.integers(0, len(candidates))]

        # Load sensor data for this bout
        return self._load_needle_from_bout(bout_ref)

    def sample_needle_for_context(
        self,
        context_activities: Set[str],
        min_duration_ms: int = 0,
        exclude_pids: Optional[Set[str]] = None,
        use_transition_probs: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[NeedleSample]:
        """
        Sample a needle that's appropriate for a given background context.

        Uses transition matrix to select a plausible activity based on what's
        in the background, then samples a needle for that activity.

        Args:
            context_activities: Activities present in the background
            min_duration_ms: Minimum bout duration
            exclude_pids: Participant IDs to exclude
            use_transition_probs: If True, use transition matrix for activity selection
            rng: Random generator

        Returns:
            NeedleSample, or None if no valid needle found
        """
        if rng is None:
            rng = np.random.default_rng()

        # Determine candidate needle activities (NOT in context)
        all_activities = set(self.bout_index.activities)
        candidate_activities = all_activities - context_activities

        if not candidate_activities:
            return None

        # Select needle activity
        if use_transition_probs and self.transition_matrix is not None:
            # Weight by plausibility given context activities
            activity = self._sample_activity_by_transition(
                context_activities, candidate_activities, rng
            )
        else:
            # Uniform random
            activity = list(candidate_activities)[
                rng.integers(0, len(candidate_activities))
            ]

        if activity is None:
            return None

        return self.sample_needle(
            activity=activity,
            min_duration_ms=min_duration_ms,
            exclude_pids=exclude_pids,
            rng=rng,
        )

    def _sample_activity_by_transition(
        self,
        context_activities: Set[str],
        candidate_activities: Set[str],
        rng: np.random.Generator,
    ) -> Optional[str]:
        """
        Sample an activity weighted by transition probabilities from context.

        Computes average transition probability from each context activity to
        each candidate activity, then samples proportionally.
        """
        if self.transition_matrix is None:
            return list(candidate_activities)[rng.integers(0, len(candidate_activities))]

        candidates = list(candidate_activities)
        weights = np.zeros(len(candidates))

        for i, candidate in enumerate(candidates):
            # Average transition prob from all context activities
            probs = []
            for ctx_activity in context_activities:
                prob = self.transition_matrix.get_transition_prob(ctx_activity, candidate)
                probs.append(prob)
            weights[i] = np.mean(probs) if probs else 0.0

        # If all weights are zero, fall back to uniform
        total = weights.sum()
        if total == 0:
            return candidates[rng.integers(0, len(candidates))]

        # Normalize and sample
        probs = weights / total
        idx = rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _load_needle_from_bout(self, bout_ref: BoutRef) -> Optional[NeedleSample]:
        """
        Load sensor data for a bout and return as NeedleSample.
        """
        # Load sensor data from cache or disk
        if bout_ref.pid not in self._sensor_cache:
            try:
                self._sensor_cache[bout_ref.pid] = load_participant_sensor_data(
                    bout_ref.pid, downsample_hz=self.source_hz
                )
            except FileNotFoundError:
                return None

        df = self._sensor_cache[bout_ref.pid]

        # Filter to bout time range
        bout_df = df.filter(
            (pl.col("timestamp_ms") >= bout_ref.start_ms)
            & (pl.col("timestamp_ms") < bout_ref.end_ms)
        )

        if len(bout_df) == 0:
            return None

        # Extract arrays
        x = bout_df["x"].to_numpy().astype(np.float32)
        y = bout_df["y"].to_numpy().astype(np.float32)
        z = bout_df["z"].to_numpy().astype(np.float32)

        return NeedleSample(
            source_pid=bout_ref.pid,
            activity=bout_ref.activity,
            start_ms=bout_ref.start_ms,
            end_ms=bout_ref.end_ms,
            duration_ms=bout_ref.duration_ms,
            x=x,
            y=y,
            z=z,
        )

    def sample_needles_for_regime(
        self,
        regime_activities: Set[str],
        n_needles: int,
        min_duration_ms: int = 0,
        exclude_pids: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[NeedleSample]:
        """
        Sample multiple needles from a set of activities (typically same regime).

        Used for distractor insertion in existence/localization tasks. Samples
        needles from different activities within the regime to prevent models
        from detecting needles by variance change alone.

        Args:
            regime_activities: Set of activity labels to sample from
            n_needles: Number of needles to sample (may return fewer if unavailable)
            min_duration_ms: Minimum bout duration
            exclude_pids: Participant IDs to exclude
            rng: Random generator

        Returns:
            List of NeedleSample objects (may be fewer than n_needles if
            not enough valid bouts are available)
        """
        if rng is None:
            rng = np.random.default_rng()

        needles = []
        activities_list = list(regime_activities)
        rng.shuffle(activities_list)

        # Try to get one needle per activity first (for diversity)
        for activity in activities_list:
            if len(needles) >= n_needles:
                break

            needle = self.sample_needle(
                activity=activity,
                min_duration_ms=min_duration_ms,
                exclude_pids=exclude_pids,
                rng=rng,
            )
            if needle is not None:
                needles.append(needle)

        # If still need more, sample from any activity in the regime
        attempts = 0
        max_attempts = n_needles * 3
        while len(needles) < n_needles and attempts < max_attempts:
            activity = activities_list[rng.integers(0, len(activities_list))]
            needle = self.sample_needle(
                activity=activity,
                min_duration_ms=min_duration_ms,
                exclude_pids=exclude_pids,
                rng=rng,
            )
            if needle is not None:
                needles.append(needle)
            attempts += 1

        return needles

    def get_available_activities(self) -> List[str]:
        """Return list of all activities in the bout index."""
        return self.bout_index.activities

    def get_activity_stats(self, activity: str) -> Optional[Dict]:
        """Get statistics for an activity from the bout index."""
        if activity in self.bout_index.activity_stats:
            return self.bout_index.activity_stats[activity].to_dict()
        return None

    def count_available_bouts(
        self,
        activity: str,
        min_duration_ms: int = 0,
        exclude_pids: Optional[Set[str]] = None,
    ) -> int:
        """Count bouts matching the given criteria."""
        bouts = self.bout_index.get_bouts_for_activity(
            activity,
            min_duration_ms=min_duration_ms,
            exclude_pids=exclude_pids,
        )
        return len(bouts)

    def clear_cache(self) -> None:
        """Clear the sensor data cache to free memory."""
        self._sensor_cache.clear()

    def get_cached_participants(self) -> List[str]:
        """Return list of participants currently in cache."""
        return list(self._sensor_cache.keys())

    def preload_participants(self, pids: List[str]) -> None:
        """
        Preload sensor data for specified participants into cache.

        Useful for reducing I/O during intensive sampling.
        """
        for pid in pids:
            if pid not in self._sensor_cache:
                try:
                    self._sensor_cache[pid] = load_participant_sensor_data(
                        pid, downsample_hz=self.source_hz
                    )
                except FileNotFoundError:
                    pass
