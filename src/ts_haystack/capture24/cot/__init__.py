# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

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
