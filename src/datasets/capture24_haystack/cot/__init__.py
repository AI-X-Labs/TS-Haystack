# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
TS-Haystack Chain-of-Thought (CoT) generation module.

This module provides LLM-based CoT rationale generation for TS-Haystack samples.
Unlike template-based approaches, it uses rich metadata (needle positions, activities,
timestamps) to generate grounded, natural reasoning.

Components:
    - OpenAICoTClient: OpenAI API client with retry logic
    - TSHaystackCoTGenerator: Main CoT generation class
    - create_accelerometer_plot: Plot generation for LLM input
    - prompt builders: Task-specific prompt construction
"""

from .llm_client import OpenAICoTClient
from .plot_generator import create_accelerometer_plot
from .prompt_builder import create_cot_prompt, format_needle_metadata, get_task_context
from .cot_generator import TSHaystackCoTGenerator

__all__ = [
    "OpenAICoTClient",
    "TSHaystackCoTGenerator",
    "create_accelerometer_plot",
    "create_cot_prompt",
    "format_needle_metadata",
    "get_task_context",
]
