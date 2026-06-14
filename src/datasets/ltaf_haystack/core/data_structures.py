# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Core data structures for LTAF-Haystack (natural-only).

Every sample is an unmodified slice of a real LTAF recording. Rhythm and
beat timelines come directly from WFDB annotations, clipped and shifted to
be window-relative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Annotation primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LTAFBoutRecord:
    start_sample: int
    end_sample: int
    activity: str
    duration_samples: int
    # True when the bout was truncated by the window boundary rather than
    # ending naturally. Set by the windowed clipper; always False on
    # whole-recording timelines.
    clipped_left: bool = False
    clipped_right: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "activity": self.activity,
            "duration_samples": self.duration_samples,
            "clipped_left": self.clipped_left,
            "clipped_right": self.clipped_right,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFBoutRecord":
        return cls(
            start_sample=int(raw["start_sample"]),
            end_sample=int(raw["end_sample"]),
            activity=str(raw["activity"]),
            duration_samples=int(raw["duration_samples"]),
            clipped_left=bool(raw.get("clipped_left", False)),
            clipped_right=bool(raw.get("clipped_right", False)),
        )


@dataclass(frozen=True)
class LTAFBeatEvent:
    """Single beat annotation (N/A/V/Q) extracted from WFDB atr files.

    ``sample`` is source-relative; ``time_ms`` is source-relative milliseconds.
    When a beat flows into a windowed recording, fields are shifted to
    window-relative coordinates.
    """

    sample: int
    time_ms: int
    symbol: str

    def to_dict(self) -> Dict[str, Any]:
        return {"sample": self.sample, "time_ms": self.time_ms, "symbol": self.symbol}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFBeatEvent":
        return cls(
            sample=int(raw["sample"]),
            time_ms=int(raw["time_ms"]),
            symbol=str(raw["symbol"]),
        )


@dataclass
class LTAFParticipantTimeline:
    participant_id: str
    record_id: str
    source_hz: int
    timeline: List[LTAFBoutRecord]

    @property
    def activities_present(self) -> set[str]:
        return {b.activity for b in self.timeline}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "record_id": self.record_id,
            "source_hz": self.source_hz,
            "timeline": [asdict(b) for b in self.timeline],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFParticipantTimeline":
        return cls(
            participant_id=str(raw["participant_id"]),
            record_id=str(raw["record_id"]),
            source_hz=int(raw["source_hz"]),
            timeline=[LTAFBoutRecord.from_dict(b) for b in raw.get("timeline", [])],
        )


@dataclass(frozen=True)
class LTAFBoutRef:
    participant_id: str
    record_id: str
    start_sample: int
    end_sample: int
    duration_samples: int
    activity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "record_id": self.record_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "duration_samples": self.duration_samples,
            "activity": self.activity,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFBoutRef":
        return cls(
            participant_id=str(raw["participant_id"]),
            record_id=str(raw["record_id"]),
            start_sample=int(raw["start_sample"]),
            end_sample=int(raw["end_sample"]),
            duration_samples=int(raw["duration_samples"]),
            activity=str(raw["activity"]),
        )


@dataclass(frozen=True)
class LTAFActivityStats:
    activity: str
    count: int
    mean_duration_samples: float
    min_duration_samples: int
    max_duration_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity": self.activity,
            "count": self.count,
            "mean_duration_samples": self.mean_duration_samples,
            "min_duration_samples": self.min_duration_samples,
            "max_duration_samples": self.max_duration_samples,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFActivityStats":
        return cls(
            activity=str(raw["activity"]),
            count=int(raw["count"]),
            mean_duration_samples=float(raw["mean_duration_samples"]),
            min_duration_samples=int(raw["min_duration_samples"]),
            max_duration_samples=int(raw["max_duration_samples"]),
        )


@dataclass
class LTAFBoutIndex:
    by_activity: Dict[str, List[LTAFBoutRef]]
    activity_stats: Dict[str, LTAFActivityStats]

    @property
    def activities(self) -> List[str]:
        return sorted(self.by_activity.keys())

    @property
    def total_bouts(self) -> int:
        return sum(len(v) for v in self.by_activity.values())

    def get_bouts_for_activity(
        self,
        activity: str,
        min_duration_samples: int | None = None,
        max_duration_samples: int | None = None,
        exclude_participants: set[str] | None = None,
        exclude_record_ids: set[str] | None = None,
    ) -> List[LTAFBoutRef]:
        bouts = list(self.by_activity.get(activity, []))
        if min_duration_samples is not None:
            bouts = [b for b in bouts if b.duration_samples >= min_duration_samples]
        if max_duration_samples is not None:
            bouts = [b for b in bouts if b.duration_samples <= max_duration_samples]
        if exclude_participants is not None:
            bouts = [b for b in bouts if b.participant_id not in exclude_participants]
        if exclude_record_ids is not None:
            bouts = [b for b in bouts if b.record_id not in exclude_record_ids]
        return bouts


# --------------------------------------------------------------------------- #
# Windowed sample
# --------------------------------------------------------------------------- #


@dataclass
class LTAFRecordingSample:
    """Windowed natural recording consumed by tasks.

    All sample/ms coordinates are *window-relative*. ``signals`` is optional
    so the sampler can defer materialization until a task needs them.
    """

    record_id: str
    window_start_ms: int
    window_end_ms: int
    source_hz: int
    rhythm_timeline: List[LTAFBoutRecord]
    beats_timeline: List[LTAFBeatEvent]
    activity_index: Dict[str, List[LTAFBoutRecord]] = field(default_factory=dict)
    signals: Optional[np.ndarray] = None

    @property
    def duration_ms(self) -> int:
        return int(self.window_end_ms - self.window_start_ms)

    @property
    def duration_samples(self) -> int:
        return int(round(self.duration_ms * self.source_hz / 1000.0))


# --------------------------------------------------------------------------- #
# Generated (task-output) sample
# --------------------------------------------------------------------------- #


@dataclass
class LTAFGeneratedSample:
    """Task output. Written to parquet by the generation orchestrator."""

    task_type: str
    question: str
    answer: str
    answer_type: str
    signals: np.ndarray
    context_length_samples: int
    record_id: str
    window_start_ms: int
    window_end_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    invalid_reason: Optional[str] = None
    source_hz: int = 128

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "question": self.question,
            "answer": self.answer,
            "answer_type": self.answer_type,
            "signals": self.signals,
            "context_length_samples": self.context_length_samples,
            "record_id": self.record_id,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "invalid_reason": self.invalid_reason,
            "source_hz": self.source_hz,
        }


# --------------------------------------------------------------------------- #
# Legacy config stub (callers should prefer generation/config.py)
# --------------------------------------------------------------------------- #


@dataclass
class LTAFDifficultyConfig:
    """Per-task runtime knobs. Kept as a minimal carrier for task-specific
    options; policy lives in ``generation/config.py``."""

    context_length_samples: int
    min_annotation_coverage: float = 0.8
    task_specific: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_length_samples": self.context_length_samples,
            "min_annotation_coverage": self.min_annotation_coverage,
            "task_specific": self.task_specific,
        }


__all__ = [
    "LTAFActivityStats",
    "LTAFBeatEvent",
    "LTAFBoutIndex",
    "LTAFBoutRecord",
    "LTAFBoutRef",
    "LTAFDifficultyConfig",
    "LTAFGeneratedSample",
    "LTAFParticipantTimeline",
    "LTAFRecordingSample",
]
