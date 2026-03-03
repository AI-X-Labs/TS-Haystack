# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
TS-Haystack: A semi-synthetic benchmark for testing retrieval and reasoning
over long time series (1K-1M+ datapoints) using Capture-24 accelerometer data.

Modules:
- core: Core infrastructure (data structures, samplers, style transfer)
- tasks: Task generators (existence, localization, counting, ordering, etc.)
- cot: Chain-of-thought rationale generation
- utils: Utility functions (timestamp conversion, position sampling)
- generation: Dataset generation pipeline and configuration
- capture24: Capture-24 data loading and preprocessing
"""

# =============================================================================
# Core Infrastructure
# =============================================================================
from ts_haystack.core import (
    # Data structures
    ActivityStats,
    BackgroundSample,
    BoutIndex,
    BoutRecord,
    BoutRef,
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
    ParticipantTimeline,
    SignalStatistics,
    TaskConfig,
    # Seed management
    ReproducibilityConfig,
    SeedManager,
    # Phase 1: Timeline & Index builders
    TimelineBuilder,
    BoutIndexer,
    TransitionMatrix,

    BackgroundSampler,
    NeedleSampler,
    StyleTransfer,
    PromptTemplateBank,
    TemplateVariant,
)

# =============================================================================
# Task Generators
# =============================================================================
from ts_haystack.tasks import (
    BaseTaskGenerator,
    ExistenceTaskGenerator,
    LocalizationTaskGenerator,
    CountingTaskGenerator,
    OrderingTaskGenerator,
    TASK_REGISTRY,
    get_task_generator,
    list_available_tasks,
)

# =============================================================================
# CoT Rationale Generation
# =============================================================================
from ts_haystack.cot import (
    OpenAICoTClient,
    TSHaystackCoTGenerator,
    create_accelerometer_plot,
    create_cot_prompt,
)

__all__ = [
    # Data structures
    "BoutRecord",
    "ParticipantTimeline",
    "BoutRef",
    "ActivityStats",
    "BoutIndex",
    "DifficultyConfig",
    "TaskConfig",
    "InsertedNeedle",
    "GeneratedSample",
    "SignalStatistics",
    "NeedleSample",
    "BackgroundSample",
    # Seed management
    "SeedManager",
    "ReproducibilityConfig",
    # Phase 1: Core builders
    "TimelineBuilder",
    "BoutIndexer",
    "TransitionMatrix",
    # Phase 2: Samplers & Style Transfer
    "BackgroundSampler",
    "NeedleSampler",
    "StyleTransfer",
    "PromptTemplateBank",
    "TemplateVariant",
    # Phase 3: Task Generators
    "BaseTaskGenerator",
    "ExistenceTaskGenerator",
    "LocalizationTaskGenerator",
    "CountingTaskGenerator",
    "OrderingTaskGenerator",
    "TASK_REGISTRY",
    "get_task_generator",
    "list_available_tasks",
    # CoT Generation
    "OpenAICoTClient",
    "TSHaystackCoTGenerator",
    "create_accelerometer_plot",
    "create_cot_prompt",
]
