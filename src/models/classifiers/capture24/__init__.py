# SPDX-License-Identifier: CC-BY-NC-4.0
"""Capture24 activity classifier (ARTS Capture24 tool)."""

from src.models.classifiers.capture24.model import (
    HARClassifier,
    WILLETTS_SPECIFIC_2018_CLASSES,
)

__all__ = ["HARClassifier", "WILLETTS_SPECIFIC_2018_CLASSES"]
