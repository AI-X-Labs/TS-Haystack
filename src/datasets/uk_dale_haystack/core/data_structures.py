"""
Dataclasses shared by every UK-DALE-Haystack task.

1-D mains signals (no x/y/z axes, no SignalStatistics, no blend regions —
additive insertion has no boundary to blend).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Bout records
# ---------------------------------------------------------------------------

@dataclass
class BoutRecord:
    """Absolute-time bout, the row schema of bout_index.parquet."""
    house_id: int
    meter_id: int
    appliance: str
    regime: str
    instance: int
    start_ns: int
    end_ns: int
    duration_s: float
    peak_w: float
    mean_w: float
    total_kwh: float
    iso_week: str
    split: str


@dataclass
class BoutRef:
    """Window-relative bout reference (used for QA timeline metadata)."""
    appliance: str
    start_sample: int
    end_sample: int          # exclusive
    duration_s: float
    peak_w: float = 0.0
    mean_w: float = 0.0


# ---------------------------------------------------------------------------
# Sampler outputs
# ---------------------------------------------------------------------------

@dataclass
class BackgroundSample:
    """A target-OFF mains window resampled onto a regular dt_s grid."""
    house_id: int
    start_ns: int
    end_ns: int
    dt_s: float
    mains_w: np.ndarray                          # (ctx_samples,) float32
    other_bouts: list[BoutRef] = field(default_factory=list)
    recording_time_context: tuple[str, str] = ("", "")  # ("HH:MM", "HH:MM") local

    @property
    def n_samples(self) -> int:
        return int(self.mains_w.shape[0])


@dataclass
class NeedleSample:
    """Real per-appliance submeter excerpt of one ON-event, on a regular grid."""
    source_house_id: int
    source_meter_id: int
    appliance: str
    start_ns: int
    end_ns: int
    duration_s: float
    submeter_w: np.ndarray                       # (n_samples,) float32
    dt_s: float
    peak_w: float
    mean_w: float
    is_anomalous: bool = False
    anomaly_class: str | None = None             # "truncated_cycle" | "abnormal_peak" | None
    anomaly_params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.submeter_w.shape[0])


# ---------------------------------------------------------------------------
# Insertion result
# ---------------------------------------------------------------------------

@dataclass
class InsertedNeedle:
    """One needle as it appears inside a GeneratedSample."""
    appliance: str
    insert_position_samples: int
    insert_duration_samples: int
    duration_s: float
    peak_w: float
    source_house_id: int
    source_meter_id: int
    source_start_ns: int          # bout's original start in source meter time
    source_end_ns: int            # bout's original end (exclusive of right pad)
    timestamp_start: str          # "HH:MM:SS" window-relative
    timestamp_end: str
    is_anomalous: bool = False
    anomaly_class: str | None = None
    anomaly_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedSample:
    """One QA sample. Mirrors the Capture-24 schema with 1-D adjustments."""
    task_type: str
    question: str
    answer: str
    answer_type: str               # boolean | time_range | integer | category

    mains_w: np.ndarray            # (ctx_samples,) float32, additive mains
    context_length_samples: int
    context_length_s: float
    dt_s: float

    background_house_id: int
    background_start_ns: int
    background_end_ns: int

    needles: list[InsertedNeedle] = field(default_factory=list)
    other_bouts_metadata: list[BoutRef] = field(default_factory=list)

    difficulty_config: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    validation_notes: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
