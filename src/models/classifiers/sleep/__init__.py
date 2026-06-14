# SPDX-License-Identifier: CC-BY-NC-4.0
"""Sleep PSG classifier (ARTS Sleep tool): stages + arousals."""

from src.models.classifiers.sleep.model import (
    SleepClassifier,
    SLEEP_STAGE_CLASS_NAMES,
    AROUSAL_CLASS_NAMES,
    default_class_names,
    default_window_samples,
)

__all__ = [
    "SleepClassifier",
    "SLEEP_STAGE_CLASS_NAMES",
    "AROUSAL_CLASS_NAMES",
    "default_class_names",
    "default_window_samples",
]
