# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Background sampler for TS-Haystack.

This module provides functionality to sample contiguous background windows
from participant sensor recordings for needle insertion.

Uses the bout index for efficient, guaranteed sampling.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import polars as pl

from ts_haystack.capture24.loader import (
    load_participant_sensor_data,
)
from ts_haystack.core.data_structures import (
    BackgroundSample,
    BoutIndex,
    BoutRecord,
    BoutRef,
    ParticipantTimeline,
)


class BackgroundSampler:
    """
    Samples background windows from participant recordings using the bout index.

    Uses bout-index-based sampling for efficiency:
    - For "pure" backgrounds: samples from bouts long enough to contain the window
    - For "mixed" backgrounds: samples windows spanning multiple bouts

    The sampler receives RNG from callers (task generators) following the seed
    strategy where SeedManager provides deterministic RNGs to task generators.

    Example:
        >>> timelines = TimelineBuilder.load_all_timelines()
        >>> bout_index = BoutIndexer.load_index()
        >>> sampler = BackgroundSampler(timelines, bout_index, source_hz=100)
        >>> rng = np.random.default_rng(42)  # In practice, from SeedManager
        >>> background = sampler.sample_background(
        ...     context_length_samples=10000,
        ...     purity="pure",
        ...     rng=rng,
        ... )
    """

    def __init__(
        self,
        timelines: Dict[str, ParticipantTimeline],
        bout_index: BoutIndex,
        source_hz: int = 100,
    ):
        """
        Initialize the background sampler.

        Args:
            timelines: Dictionary mapping participant ID to timeline
            bout_index: Cross-participant bout index for efficient sampling
            source_hz: Source data sampling frequency in Hz
        """
        self.timelines = timelines
        self.bout_index = bout_index
        self.source_hz = source_hz
        self.participant_ids = sorted(timelines.keys())

        # Cache for loaded sensor data (pid -> DataFrame)
        self._sensor_cache: Dict[str, pl.DataFrame] = {}

    def sample_background(
        self,
        context_length_samples: int,
        purity: Literal["pure", "mixed", "any"] = "pure",
        allowed_activities: Optional[Set[str]] = None,
        excluded_activities: Optional[Set[str]] = None,
        min_activity_count: int = 1,
        max_activity_count: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> BackgroundSample:
        """
        Sample a background window from the participant recordings.

        Args:
            context_length_samples: Window size in samples
            purity: "pure" = single activity, "mixed" = multiple activities,
                   "any" = samples a random window without activity count constraints
                         (naturally pure for short contexts, potentially mixed for longer)
            allowed_activities: Only sample from windows with these activities
            excluded_activities: Never sample from windows with these activities
            min_activity_count: Minimum distinct activities in window
            max_activity_count: Maximum distinct activities in window
            rng: Random generator (should come from SeedManager via task generator)

        Returns:
            BackgroundSample with sensor data and metadata

        Raises:
            ValueError: If no valid window can be found
        """
        if rng is None:
            rng = np.random.default_rng()

        context_duration_ms = int(context_length_samples * 1000 / self.source_hz)

        if purity == "pure":
            return self._sample_pure_background(
                context_length_samples=context_length_samples,
                context_duration_ms=context_duration_ms,
                allowed_activities=allowed_activities,
                excluded_activities=excluded_activities,
                rng=rng,
            )
        elif purity == "mixed":
            return self._sample_mixed_background(
                context_length_samples=context_length_samples,
                context_duration_ms=context_duration_ms,
                allowed_activities=allowed_activities,
                excluded_activities=excluded_activities,
                min_activity_count=max(min_activity_count, 2),
                max_activity_count=max_activity_count,
                rng=rng,
            )
        else:  # purity == "any"
            # Sample a random window without activity count constraints
            # This naturally produces pure backgrounds for short contexts
            # and potentially mixed backgrounds for longer contexts
            return self._sample_mixed_background(
                context_length_samples=context_length_samples,
                context_duration_ms=context_duration_ms,
                allowed_activities=allowed_activities,
                excluded_activities=excluded_activities,
                min_activity_count=1,  # No constraint - accept any window
                max_activity_count=max_activity_count,
                rng=rng,
            )

    def _sample_pure_background(
        self,
        context_length_samples: int,
        context_duration_ms: int,
        allowed_activities: Optional[Set[str]],
        excluded_activities: Optional[Set[str]],
        rng: np.random.Generator,
        max_attempts: int = 10,
    ) -> BackgroundSample:
        """
        Sample a pure background (single activity) using bout index.

        Strategy:
        1. Get all activities that pass filters
        2. Sample an activity
        3. Get bouts for that activity that are long enough
        4. Sample a bout
        5. Sample a window position within the bout
        6. If loading fails, retry with a different bout (up to max_attempts)
        """
        # Determine candidate activities
        all_activities = set(self.bout_index.activities)

        if allowed_activities is not None:
            candidate_activities = all_activities & allowed_activities
        else:
            candidate_activities = all_activities

        if excluded_activities is not None:
            candidate_activities = candidate_activities - excluded_activities

        if not candidate_activities:
            raise ValueError(
                f"No candidate activities after filtering. "
                f"allowed={allowed_activities}, excluded={excluded_activities}"
            )

        # Find activities with bouts long enough
        activities_with_valid_bouts = []
        for activity in candidate_activities:
            bouts = self.bout_index.get_bouts_for_activity(
                activity, min_duration_ms=context_duration_ms
            )
            if bouts:
                activities_with_valid_bouts.append(activity)

        if not activities_with_valid_bouts:
            raise ValueError(
                f"No activities have bouts >= {context_duration_ms}ms. "
                f"Try shorter context_length_samples."
            )

        # Track failed PIDs to avoid retrying them
        failed_pids: Set[str] = set()

        for attempt in range(max_attempts):
            # Sample an activity
            activity = activities_with_valid_bouts[
                rng.integers(0, len(activities_with_valid_bouts))
            ]

            # Get valid bouts for this activity, excluding failed PIDs
            all_valid_bouts = self.bout_index.get_bouts_for_activity(
                activity, min_duration_ms=context_duration_ms
            )
            valid_bouts = [b for b in all_valid_bouts if b.pid not in failed_pids]

            if not valid_bouts:
                # All bouts for this activity have failed PIDs, try another activity
                continue

            # Sample a bout
            bout_ref = valid_bouts[rng.integers(0, len(valid_bouts))]

            # Sample window position within bout
            max_offset_ms = bout_ref.duration_ms - context_duration_ms
            offset_ms = rng.integers(0, max_offset_ms + 1) if max_offset_ms > 0 else 0

            start_ms = bout_ref.start_ms + offset_ms
            end_ms = start_ms + context_duration_ms

            # Load sensor data
            x, y, z = self._load_sensor_window(
                bout_ref.pid, start_ms, end_ms, context_length_samples
            )

            if x is None:
                # Loading failed for this participant, mark as failed and retry
                failed_pids.add(bout_ref.pid)
                continue

            # Build activity timeline (entire window is one activity)
            activity_timeline = [(0.0, 1.0, activity)]

            # Format time range
            recording_time_context = self._format_time_range(start_ms, end_ms)

            return BackgroundSample(
                pid=bout_ref.pid,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=context_duration_ms,
                x=x,
                y=y,
                z=z,
                activities_present={activity},
                activity_timeline=activity_timeline,
                recording_time_context=recording_time_context,
            )

        # All attempts failed
        raise ValueError(
            f"Failed to sample pure background after {max_attempts} attempts. "
            f"Failed PIDs: {failed_pids}"
        )

    def _sample_mixed_background(
        self,
        context_length_samples: int,
        context_duration_ms: int,
        allowed_activities: Optional[Set[str]],
        excluded_activities: Optional[Set[str]],
        min_activity_count: int,
        max_activity_count: Optional[int],
        rng: np.random.Generator,
    ) -> BackgroundSample:
        """
        Sample a mixed background (multiple activities) using timelines.

        Strategy:
        1. Find participants with recordings long enough
        2. For each candidate, find windows spanning multiple bouts
        3. Sample a valid window
        """
        # Shuffle participants for random selection
        shuffled_pids = list(self.participant_ids)
        rng.shuffle(shuffled_pids)

        for pid in shuffled_pids:
            timeline = self.timelines[pid]

            # Skip if recording too short
            if timeline.total_duration_ms < context_duration_ms:
                continue

            # Try to find a valid mixed window in this participant
            result = self._try_sample_mixed_from_timeline(
                pid=pid,
                timeline=timeline,
                context_length_samples=context_length_samples,
                context_duration_ms=context_duration_ms,
                allowed_activities=allowed_activities,
                excluded_activities=excluded_activities,
                min_activity_count=min_activity_count,
                max_activity_count=max_activity_count,
                rng=rng,
            )

            if result is not None:
                return result

        raise ValueError(
            f"Could not find mixed background with {min_activity_count}+ activities. "
            f"Try relaxing constraints."
        )

    def _try_sample_mixed_from_timeline(
        self,
        pid: str,
        timeline: ParticipantTimeline,
        context_length_samples: int,
        context_duration_ms: int,
        allowed_activities: Optional[Set[str]],
        excluded_activities: Optional[Set[str]],
        min_activity_count: int,
        max_activity_count: Optional[int],
        rng: np.random.Generator,
        max_attempts: int = 20,
    ) -> Optional[BackgroundSample]:
        """
        Try to sample a mixed background from a specific participant's timeline.
        """
        for _ in range(max_attempts):
            # Sample a random start position
            max_start_ms = timeline.recording_end_ms - context_duration_ms
            if max_start_ms <= timeline.recording_start_ms:
                return None

            start_ms = rng.integers(timeline.recording_start_ms, max_start_ms + 1)
            end_ms = start_ms + context_duration_ms

            # Find activities in this window
            window_bouts = self._get_bouts_in_range(timeline, start_ms, end_ms)

            if not window_bouts:
                continue

            # Extract unique activities
            activities_present = set(b.activity for b in window_bouts)

            # Apply activity filters
            if allowed_activities is not None:
                if not activities_present.issubset(allowed_activities):
                    continue

            if excluded_activities is not None:
                if activities_present.intersection(excluded_activities):
                    continue

            # Check activity count constraints
            n_activities = len(activities_present)
            if n_activities < min_activity_count:
                continue
            if max_activity_count is not None and n_activities > max_activity_count:
                continue

            # Found a valid window - load sensor data
            x, y, z = self._load_sensor_window(
                pid, start_ms, end_ms, context_length_samples
            )

            if x is None:
                continue

            # Build activity timeline
            activity_timeline = self._build_activity_timeline(
                window_bouts, start_ms, end_ms
            )

            # Format time range
            recording_time_context = self._format_time_range(start_ms, end_ms)

            return BackgroundSample(
                pid=pid,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=context_duration_ms,
                x=x,
                y=y,
                z=z,
                activities_present=activities_present,
                activity_timeline=activity_timeline,
                recording_time_context=recording_time_context,
            )

        return None

    def _get_bouts_in_range(
        self,
        timeline: ParticipantTimeline,
        start_ms: int,
        end_ms: int,
    ) -> List[BoutRecord]:
        """
        Get all bouts that overlap with the given time range.
        """
        bouts = []
        for bout in timeline.timeline:
            # Check if bout overlaps with range
            if bout.end_ms > start_ms and bout.start_ms < end_ms:
                bouts.append(bout)
        return bouts

    def _build_activity_timeline(
        self,
        bouts: List[BoutRecord],
        window_start_ms: int,
        window_end_ms: int,
    ) -> List[Tuple[float, float, str]]:
        """
        Build activity timeline as list of (start_frac, end_frac, activity) tuples.

        Fractions are relative to the window (0.0 = window start, 1.0 = window end).
        """
        window_duration = window_end_ms - window_start_ms
        timeline = []

        for bout in sorted(bouts, key=lambda b: b.start_ms):
            # Clip bout to window boundaries
            clipped_start = max(bout.start_ms, window_start_ms)
            clipped_end = min(bout.end_ms, window_end_ms)

            # Convert to fractions
            start_frac = (clipped_start - window_start_ms) / window_duration
            end_frac = (clipped_end - window_start_ms) / window_duration

            timeline.append((start_frac, end_frac, bout.activity))

        return timeline

    def _load_sensor_window(
        self,
        pid: str,
        start_ms: int,
        end_ms: int,
        expected_samples: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load sensor data for a time window.

        Returns (x, y, z) arrays or (None, None, None) if loading fails.
        """
        # Load from cache or disk
        if pid not in self._sensor_cache:
            try:
                self._sensor_cache[pid] = load_participant_sensor_data(
                    pid, downsample_hz=self.source_hz
                )
            except FileNotFoundError:
                return None, None, None

        df = self._sensor_cache[pid]

        # Filter to time range
        window_df = df.filter(
            (pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") < end_ms)
        )

        if len(window_df) == 0:
            return None, None, None

        # Extract arrays
        x = window_df["x"].to_numpy().astype(np.float32)
        y = window_df["y"].to_numpy().astype(np.float32)
        z = window_df["z"].to_numpy().astype(np.float32)

        # Adjust to expected length if needed
        if len(x) < expected_samples:
            # Pad with edge values
            pad_length = expected_samples - len(x)
            x = np.pad(x, (0, pad_length), mode="edge")
            y = np.pad(y, (0, pad_length), mode="edge")
            z = np.pad(z, (0, pad_length), mode="edge")
        elif len(x) > expected_samples:
            # Trim to expected length
            x = x[:expected_samples]
            y = y[:expected_samples]
            z = z[:expected_samples]

        return x, y, z

    def _format_time_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> Tuple[str, str]:
        """
        Convert Unix timestamps to human-readable time strings.

        Returns tuple of (start_time, end_time) with full precision including
        seconds and milliseconds for distinguishable timestamps even with
        short needle durations.

        Example: ("6:00:00.000 AM", "6:01:40.500 AM")
        """
        start_dt = datetime.fromtimestamp(start_ms / 1000)
        end_dt = datetime.fromtimestamp(end_ms / 1000)

        def format_dt(dt: datetime) -> str:
            """Format datetime with seconds and milliseconds."""
            base = dt.strftime("%I:%M:%S")
            ms = dt.microsecond // 1000
            am_pm = dt.strftime("%p")
            result = f"{base}.{ms:03d} {am_pm}"
            return result.lstrip("0") or "0" + result[1:]

        return format_dt(start_dt), format_dt(end_dt)

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
            if pid not in self._sensor_cache and pid in self.participant_ids:
                try:
                    self._sensor_cache[pid] = load_participant_sensor_data(
                        pid, downsample_hz=self.source_hz
                    )
                except FileNotFoundError:
                    pass

    def get_available_activities(self) -> List[str]:
        """Return list of all activities in the bout index."""
        return self.bout_index.activities
