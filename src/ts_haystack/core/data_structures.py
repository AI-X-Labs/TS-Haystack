# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Core data structures for TS-Haystack benchmark.

This module defines all dataclasses used throughout the TS-Haystack pipeline,
from timeline extraction to task generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Timeline & Bout Structures
# =============================================================================


@dataclass
class BoutRecord:
    """
    A single activity bout within a participant's timeline.

    A bout is a contiguous period where the participant performed the same activity.

    Attributes:
        start_ms: Start timestamp in milliseconds (Unix epoch)
        end_ms: End timestamp in milliseconds (Unix epoch)
        activity: Activity label (e.g., "walking", "sleep")
        duration_ms: Duration in milliseconds (end_ms - start_ms)
    """

    start_ms: int
    end_ms: int
    activity: str
    duration_ms: int

    @property
    def center_ms(self) -> int:
        """Return the center timestamp of the bout."""
        return (self.start_ms + self.end_ms) // 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "activity": self.activity,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoutRecord":
        """Create from dictionary."""
        return cls(
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            activity=d["activity"],
            duration_ms=d["duration_ms"],
        )


@dataclass
class ParticipantTimeline:
    """
    Complete activity timeline for one participant.

    Contains both a chronological list of bouts and an activity-indexed view
    for efficient querying.

    Attributes:
        participant_id: Participant identifier (e.g., "P001")
        total_duration_ms: Total recording duration in milliseconds
        recording_start_ms: First timestamp in the recording
        recording_end_ms: Last timestamp in the recording
        timeline: Chronologically ordered list of bouts
        bouts_by_activity: Bouts indexed by activity label
    """

    participant_id: str
    total_duration_ms: int
    recording_start_ms: int
    recording_end_ms: int
    timeline: List[BoutRecord]
    bouts_by_activity: Dict[str, List[BoutRecord]]

    @property
    def num_bouts(self) -> int:
        """Total number of bouts in the timeline."""
        return len(self.timeline)

    @property
    def activities_present(self) -> Set[str]:
        """Set of unique activities in this timeline."""
        return set(self.bouts_by_activity.keys())

    def get_bout_at_time(self, timestamp_ms: int) -> Optional[BoutRecord]:
        """
        Find the bout containing the given timestamp.

        Args:
            timestamp_ms: Timestamp to query

        Returns:
            BoutRecord if found, None otherwise
        """
        for bout in self.timeline:
            if bout.start_ms <= timestamp_ms < bout.end_ms:
                return bout
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "participant_id": self.participant_id,
            "total_duration_ms": self.total_duration_ms,
            "recording_start_ms": self.recording_start_ms,
            "recording_end_ms": self.recording_end_ms,
            "timeline": [b.to_dict() for b in self.timeline],
            "bouts_by_activity": {
                act: [b.to_dict() for b in bouts]
                for act, bouts in self.bouts_by_activity.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParticipantTimeline":
        """Create from dictionary."""
        timeline = [BoutRecord.from_dict(b) for b in d["timeline"]]
        bouts_by_activity = {
            act: [BoutRecord.from_dict(b) for b in bouts]
            for act, bouts in d["bouts_by_activity"].items()
        }
        return cls(
            participant_id=d["participant_id"],
            total_duration_ms=d["total_duration_ms"],
            recording_start_ms=d["recording_start_ms"],
            recording_end_ms=d["recording_end_ms"],
            timeline=timeline,
            bouts_by_activity=bouts_by_activity,
        )


# =============================================================================
# Bout Index Structures
# =============================================================================


@dataclass
class BoutRef:
    """
    Reference to a bout in the cross-participant index.

    Used for efficient needle sampling without loading full participant data.

    Attributes:
        pid: Participant ID
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        duration_ms: Duration in milliseconds
        activity: Activity label
    """

    pid: str
    start_ms: int
    end_ms: int
    duration_ms: int
    activity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pid": self.pid,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "activity": self.activity,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoutRef":
        """Create from dictionary."""
        return cls(
            pid=d["pid"],
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            duration_ms=d["duration_ms"],
            activity=d["activity"],
        )


@dataclass
class ActivityStats:
    """
    Statistics for an activity across all participants.

    Attributes:
        activity: Activity label
        count: Total number of bouts
        total_duration_ms: Sum of all bout durations
        mean_duration_ms: Mean bout duration
        min_duration_ms: Minimum bout duration
        max_duration_ms: Maximum bout duration
        participant_count: Number of participants with this activity
    """

    activity: str
    count: int
    total_duration_ms: int
    mean_duration_ms: float
    min_duration_ms: int
    max_duration_ms: int
    participant_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "activity": self.activity,
            "count": self.count,
            "total_duration_ms": self.total_duration_ms,
            "mean_duration_ms": self.mean_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "participant_count": self.participant_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActivityStats":
        """Create from dictionary."""
        return cls(
            activity=d["activity"],
            count=d["count"],
            total_duration_ms=d["total_duration_ms"],
            mean_duration_ms=d["mean_duration_ms"],
            min_duration_ms=d["min_duration_ms"],
            max_duration_ms=d["max_duration_ms"],
            participant_count=d.get("participant_count", 0),
        )


@dataclass
class BoutIndex:
    """
    Cross-participant bout index for efficient needle sampling.

    Aggregates all bouts from all participants, indexed by activity.

    Attributes:
        by_activity: Bouts grouped by activity label
        activity_stats: Statistics per activity
    """

    by_activity: Dict[str, List[BoutRef]]
    activity_stats: Dict[str, ActivityStats]

    @property
    def activities(self) -> List[str]:
        """List of all activities in the index."""
        return sorted(self.by_activity.keys())

    @property
    def total_bouts(self) -> int:
        """Total number of bouts across all activities."""
        return sum(len(bouts) for bouts in self.by_activity.values())

    def get_bouts_for_activity(
        self,
        activity: str,
        min_duration_ms: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
        exclude_pids: Optional[Set[str]] = None,
    ) -> List[BoutRef]:
        """
        Get bouts for an activity with optional filtering.

        Args:
            activity: Activity label
            min_duration_ms: Minimum duration filter
            max_duration_ms: Maximum duration filter
            exclude_pids: Participant IDs to exclude

        Returns:
            Filtered list of BoutRef
        """
        bouts = self.by_activity.get(activity, [])

        if min_duration_ms is not None:
            bouts = [b for b in bouts if b.duration_ms >= min_duration_ms]

        if max_duration_ms is not None:
            bouts = [b for b in bouts if b.duration_ms <= max_duration_ms]

        if exclude_pids is not None:
            bouts = [b for b in bouts if b.pid not in exclude_pids]

        return bouts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (stats only)."""
        return {
            "activities": self.activities,
            "total_bouts": self.total_bouts,
            "activity_stats": {
                act: stats.to_dict() for act, stats in self.activity_stats.items()
            },
        }


# =============================================================================
# Configuration Structures
# =============================================================================


@dataclass
class DifficultyConfig:
    """
    Difficulty configuration for sample generation.

    Controls various axes of difficulty for generated samples.

    Attributes:
        context_length_samples: Window size in samples (1K, 10K, 100K, 1M)
        needle_position: Position mode for needle insertion
        needle_length_distribution: How to sample needle lengths
        needle_length_ratio_range: Min/max needle duration as fraction of context length
        distractor_density: Distractor configuration
        distractor_count: Number of distractors
        background_purity: Whether background is single or mixed activity
        min_annotation_coverage: Minimum fraction of background with activity annotations
                                (filters out samples with large unlabeled gaps)
        task_specific: Task-specific configuration overrides
    """

    context_length_samples: int
    needle_position: str = "random"  # "beginning", "middle", "end", "random"
    needle_length_distribution: str = "uniform"  # "uniform", "activity_specific"
    needle_length_ratio_range: Tuple[float, float] = (0.02, 0.10)  # 2% to 10% of context
    distractor_density: str = "none"  # "none", "low", "high"
    distractor_count: int = 0
    background_purity: str = "pure"  # "pure", "mixed", or "any" (random window, adapts to context)
    min_annotation_coverage: float = 0.6  # 60% minimum annotation coverage
    task_specific: Dict[str, Any] = field(default_factory=dict)

    def get_needle_length_range_samples(self) -> Tuple[int, int]:
        """
        Compute needle length range in samples from ratio.

        Returns:
            Tuple of (min_samples, max_samples) for needle length
        """
        min_ratio, max_ratio = self.needle_length_ratio_range
        return (
            int(min_ratio * self.context_length_samples),
            int(max_ratio * self.context_length_samples),
        )

    def get_needle_length_range_ms(self, source_hz: int = 100) -> Tuple[int, int]:
        """
        Compute needle length range in milliseconds from ratio.

        Args:
            source_hz: Source data sampling frequency in Hz

        Returns:
            Tuple of (min_ms, max_ms) for needle duration
        """
        min_samples, max_samples = self.get_needle_length_range_samples()
        return (
            int(min_samples * 1000 / source_hz),
            int(max_samples * 1000 / source_hz),
        )

    def get_effective_margin_samples(self) -> int:
        """
        Compute effective margin based on ratio and max cap.

        Uses ratio-based computation with a maximum cap to allow adaptive
        margins for short context lengths while maintaining reasonable bounds
        for longer contexts.

        Config parameters (in task_specific):
            margin_ratio: Margin as fraction of context (default: 0.02 = 2%)
            margin_max_samples: Maximum margin in samples (default: 100)

        Returns:
            Effective margin in samples: min(context_length * ratio, max_cap)
        """
        ratio = self.task_specific.get("margin_ratio", 0.02)
        max_cap = self.task_specific.get("margin_max_samples", 100)
        return min(int(self.context_length_samples * ratio), max_cap)

    def get_effective_min_gap_samples(self) -> int:
        """
        Compute effective minimum gap based on ratio and max cap.

        Uses ratio-based computation with a maximum cap to allow adaptive
        gaps for short context lengths while maintaining reasonable bounds
        for longer contexts.

        Config parameters (in task_specific):
            min_gap_ratio: Gap as fraction of context (default: 0.02 = 2%)
            min_gap_max_samples: Maximum gap in samples (default: 100)

        Returns:
            Effective min gap in samples: min(context_length * ratio, max_cap)
        """
        ratio = self.task_specific.get("min_gap_ratio", 0.02)
        max_cap = self.task_specific.get("min_gap_max_samples", 100)
        return min(int(self.context_length_samples * ratio), max_cap)

    def get_effective_min_state_duration_samples(self) -> int:
        """
        Compute effective min state duration for State Query task.

        Uses ratio-based computation with a maximum cap to allow adaptive
        state durations for short context lengths while maintaining reasonable
        bounds for longer contexts.

        Config parameters (in task_specific):
            min_state_duration_ratio: State duration as fraction of context (default: 0.20 = 20%)
            min_state_duration_max_samples: Maximum duration in samples (default: 500)

        Returns:
            Effective min state duration in samples: min(context_length * ratio, max_cap)
        """
        ratio = self.task_specific.get("min_state_duration_ratio", 0.20)
        max_cap = self.task_specific.get("min_state_duration_max_samples", 500)
        return min(int(self.context_length_samples * ratio), max_cap)

    def get_effective_min_duration_diff_ms(self, source_hz: int = 100) -> int:
        """
        Compute effective min duration diff for Comparison task.

        Uses ratio-based computation with a maximum cap to allow adaptive
        duration differences for short context lengths while maintaining
        reasonable bounds for longer contexts.

        Config parameters (in task_specific):
            min_duration_diff_ratio: Duration diff as fraction of context (default: 0.02 = 2%)
            min_duration_diff_max_ms: Maximum diff in milliseconds (default: 2000)

        Returns:
            Effective min duration diff in milliseconds: min(context_length * ratio * ms_per_sample, max_cap)
        """
        ratio = self.task_specific.get("min_duration_diff_ratio", 0.02)
        max_cap = self.task_specific.get("min_duration_diff_max_ms", 2000)
        ratio_based_samples = int(self.context_length_samples * ratio)
        ratio_based_ms = int(ratio_based_samples * 1000 / source_hz)
        return min(ratio_based_ms, max_cap)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_length_samples": self.context_length_samples,
            "needle_position": self.needle_position,
            "needle_length_distribution": self.needle_length_distribution,
            "needle_length_ratio_range": list(self.needle_length_ratio_range),
            "distractor_density": self.distractor_density,
            "distractor_count": self.distractor_count,
            "background_purity": self.background_purity,
            "min_annotation_coverage": self.min_annotation_coverage,
            "task_specific": self.task_specific,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DifficultyConfig":
        """Create from dictionary."""
        return cls(
            context_length_samples=d["context_length_samples"],
            needle_position=d.get("needle_position", "random"),
            needle_length_distribution=d.get("needle_length_distribution", "uniform"),
            needle_length_ratio_range=tuple(d.get("needle_length_ratio_range", (0.02, 0.10))),
            distractor_density=d.get("distractor_density", "none"),
            distractor_count=d.get("distractor_count", 0),
            background_purity=d.get("background_purity", "pure"),
            min_annotation_coverage=d.get("min_annotation_coverage", 0.6),
            task_specific=d.get("task_specific", {}),
        )


@dataclass
class TaskConfig:
    """
    Configuration for a task generator.

    Attributes:
        task_name: Unique task identifier
        samples_per_split: Number of samples per split
        context_lengths: Context lengths to generate
        difficulty_levels: List of difficulty configurations
        label_scheme: Label scheme to use
        seed: Random seed for reproducibility
    """

    task_name: str
    samples_per_split: Dict[str, int]  # {"train": 10000, "val": 1000, "test": 1000}
    context_lengths: List[int]  # [1000, 10000, 100000, 1000000]
    difficulty_levels: List[DifficultyConfig] = field(default_factory=list)
    label_scheme: str = "WillettsSpecific2018"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_name": self.task_name,
            "samples_per_split": self.samples_per_split,
            "context_lengths": self.context_lengths,
            "difficulty_levels": [d.to_dict() for d in self.difficulty_levels],
            "label_scheme": self.label_scheme,
            "seed": self.seed,
        }


# =============================================================================
# Sample Generation Structures (for Phase 2+)
# =============================================================================


@dataclass
class InsertedNeedle:
    """
    Record of an inserted needle with position information.

    Attributes:
        activity: Activity label of the needle
        source_pid: Source participant ID
        source_start_ms: Original start timestamp in source
        source_end_ms: Original end timestamp in source
        insert_position_samples: Position in output sequence (sample index)
        insert_position_frac: Fractional position (0-1)
        duration_samples: Duration in samples
        duration_ms: Duration in milliseconds
        timestamp_start: Human-readable start time
        timestamp_end: Human-readable end time
    """

    activity: str
    source_pid: str
    source_start_ms: int
    source_end_ms: int
    insert_position_samples: int
    insert_position_frac: float
    duration_samples: int
    duration_ms: int
    timestamp_start: str
    timestamp_end: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "activity": self.activity,
            "source_pid": self.source_pid,
            "source_start_ms": self.source_start_ms,
            "source_end_ms": self.source_end_ms,
            "insert_position_samples": self.insert_position_samples,
            "insert_position_frac": self.insert_position_frac,
            "duration_samples": self.duration_samples,
            "duration_ms": self.duration_ms,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InsertedNeedle":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class GeneratedSample:
    """
    A complete generated sample for any task.

    Contains sensor data, metadata, question/answer, and validation info.

    Attributes:
        x, y, z: Sensor data arrays
        task_type: Task identifier
        context_length_samples: Window size in samples
        background_pid: Source participant for background
        recording_time_range: Human-readable time range
        question: Generated question string
        answer: Ground truth answer string
        answer_type: Type of answer (boolean, timestamp, integer, etc.)
        needles: List of inserted needles
        difficulty_config: Difficulty configuration used
        is_valid: Whether sample passed validation
        validation_notes: Notes from validation
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    task_type: str
    context_length_samples: int
    background_pid: str
    recording_time_range: Tuple[str, str]
    question: str
    answer: str
    answer_type: str  # "boolean", "timestamp", "integer", "category", "time_range"
    needles: List[InsertedNeedle]
    difficulty_config: Dict[str, Any]
    is_valid: bool
    validation_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for parquet serialization."""
        return {
            "x_axis": self.x.tolist(),
            "y_axis": self.y.tolist(),
            "z_axis": self.z.tolist(),
            "task_type": self.task_type,
            "context_length_samples": self.context_length_samples,
            "background_pid": self.background_pid,
            "recording_time_start": self.recording_time_range[0],
            "recording_time_end": self.recording_time_range[1],
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "needles": [n.to_dict() for n in self.needles],
            "difficulty_config": self.difficulty_config,
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes,
        }


# =============================================================================
# Sampling Structures (Phase 2)
# =============================================================================


@dataclass
class SignalStatistics:
    """
    Statistics for style transfer computation.

    Used to match the statistical properties of a needle signal to a target context.

    Attributes:
        mean: Per-axis mean values, shape (3,)
        std: Per-axis standard deviations, shape (3,)
        cov: Covariance matrix, shape (3, 3)
        cholesky: Cholesky decomposition of covariance, shape (3, 3)
    """

    mean: np.ndarray
    std: np.ndarray
    cov: np.ndarray
    cholesky: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "cov": self.cov.tolist(),
            "cholesky": self.cholesky.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalStatistics":
        """Create from dictionary."""
        return cls(
            mean=np.array(d["mean"]),
            std=np.array(d["std"]),
            cov=np.array(d["cov"]),
            cholesky=np.array(d["cholesky"]),
        )


@dataclass
class NeedleSample:
    """
    A sampled needle bout with sensor data.

    Contains the actual accelerometer data extracted from a bout,
    ready for style transfer and insertion into a background.

    Attributes:
        source_pid: Participant ID where this needle was sampled from
        activity: Activity label of the needle
        start_ms: Original start timestamp in source recording
        end_ms: Original end timestamp in source recording
        duration_ms: Duration in milliseconds
        x: X-axis accelerometer data
        y: Y-axis accelerometer data
        z: Z-axis accelerometer data
    """

    source_pid: str
    activity: str
    start_ms: int
    end_ms: int
    duration_ms: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of samples in the needle."""
        return len(self.x)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_pid": self.source_pid,
            "activity": self.activity,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeedleSample":
        """Create from dictionary."""
        return cls(
            source_pid=d["source_pid"],
            activity=d["activity"],
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            duration_ms=d["duration_ms"],
            x=np.array(d["x"]),
            y=np.array(d["y"]),
            z=np.array(d["z"]),
        )

    def trim(self, n_samples: int) -> "NeedleSample":
        """
        Return a new NeedleSample trimmed to the specified number of samples.

        Trims from the center of the needle to preserve the most characteristic signal.
        """
        if n_samples >= self.n_samples:
            return self

        # Trim from center
        start_idx = (self.n_samples - n_samples) // 2
        end_idx = start_idx + n_samples

        # Compute new timestamps
        sample_duration_ms = self.duration_ms / self.n_samples
        new_start_ms = self.start_ms + int(start_idx * sample_duration_ms)
        new_end_ms = self.start_ms + int(end_idx * sample_duration_ms)

        return NeedleSample(
            source_pid=self.source_pid,
            activity=self.activity,
            start_ms=new_start_ms,
            end_ms=new_end_ms,
            duration_ms=new_end_ms - new_start_ms,
            x=self.x[start_idx:end_idx].copy(),
            y=self.y[start_idx:end_idx].copy(),
            z=self.z[start_idx:end_idx].copy(),
        )


@dataclass
class BackgroundSample:
    """
    A sampled background window with sensor data.

    Contains the accelerometer data for a contiguous time window,
    along with metadata about the activities present in the window.

    Attributes:
        pid: Participant ID
        start_ms: Start timestamp in milliseconds (Unix epoch)
        end_ms: End timestamp in milliseconds (Unix epoch)
        duration_ms: Duration in milliseconds
        x: X-axis accelerometer data
        y: Y-axis accelerometer data
        z: Z-axis accelerometer data
        activities_present: Set of unique activities in this window
        activity_timeline: List of (start_frac, end_frac, activity) tuples
                          describing the activity composition within the window
        recording_time_context: Human-readable time range tuple (start_time, end_time)
                               e.g., ("6:00 AM", "8:00 AM")
    """

    pid: str
    start_ms: int
    end_ms: int
    duration_ms: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    activities_present: Set[str]
    activity_timeline: List[Tuple[float, float, str]]
    recording_time_context: Tuple[str, str]

    @property
    def n_samples(self) -> int:
        """Number of samples in the background."""
        return len(self.x)

    @property
    def is_pure(self) -> bool:
        """Return True if background contains only one activity."""
        return len(self.activities_present) == 1

    @property
    def annotation_coverage(self) -> float:
        """
        Compute the fraction of the background window that has activity annotations.

        This measures how much of the signal has labeled activities vs unlabeled gaps.
        Gaps occur when the original Capture-24 annotations don't map to the label scheme.

        Returns:
            Float between 0.0 and 1.0 representing the fraction of the window
            that is covered by activity annotations.

        Example:
            If activity_timeline = [(0.0, 0.3, "walking"), (0.7, 1.0, "sitting")]
            The coverage would be 0.3 + 0.3 = 0.6 (60%)
        """
        if not self.activity_timeline:
            return 0.0

        total_coverage = sum(
            end_frac - start_frac
            for start_frac, end_frac, _ in self.activity_timeline
        )
        # Clamp to [0, 1] to handle any floating point issues
        return max(0.0, min(1.0, total_coverage))

    def get_activity_at_position(self, position_frac: float) -> Optional[str]:
        """
        Get the activity at a given fractional position in the window.

        Args:
            position_frac: Position as fraction of window (0.0 to 1.0)

        Returns:
            Activity label at that position, or None if not found
        """
        for start_frac, end_frac, activity in self.activity_timeline:
            if start_frac <= position_frac < end_frac:
                return activity
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pid": self.pid,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "activities_present": list(self.activities_present),
            "activity_timeline": self.activity_timeline,
            "recording_time_context": list(self.recording_time_context),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackgroundSample":
        """Create from dictionary."""
        return cls(
            pid=d["pid"],
            start_ms=d["start_ms"],
            end_ms=d["end_ms"],
            duration_ms=d["duration_ms"],
            x=np.array(d["x"]),
            y=np.array(d["y"]),
            z=np.array(d["z"]),
            activities_present=set(d["activities_present"]),
            activity_timeline=[tuple(t) for t in d["activity_timeline"]],
            recording_time_context=tuple(d["recording_time_context"]),
        )
