# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

# Capture24 Chain-of-Thought Dataset Module
# This module provides tools for generating and loading CoT-augmented accelerometer data

from .cot_generator import (
    CAPTURE24_COT_DATA_DIR,
    CAPTURE24_DISSIMILAR_MAPPING,
    Capture24CoTGenerator,
    GenerationConfig,
)

__all__ = [
    "CAPTURE24_COT_DATA_DIR",
    "CAPTURE24_DISSIMILAR_MAPPING",
    "Capture24CoTGenerator",
    "GenerationConfig",
]
