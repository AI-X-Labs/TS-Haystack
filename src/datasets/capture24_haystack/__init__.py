# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
TS-Haystack: A semi-synthetic benchmark for testing retrieval and reasoning
over long time series (1K-1M+ datapoints) using Capture-24 accelerometer data.

Modules:
- core: Core infrastructure (data structures, samplers, style transfer)
- tasks: Task generators (existence, localization, counting, ordering, etc.)
- dataset: QADataset implementations for OpenTSLM training
- cot: Chain-of-thought rationale generation
- utils: Utility functions (timestamp conversion, position sampling)
"""

# =============================================================================
# Core Infrastructure
# =============================================================================
from src.datasets.capture24_haystack.core import (
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
from src.datasets.capture24_haystack.tasks import (
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
# Dataset & QADataset Integration
# =============================================================================
from src.datasets.capture24_haystack.qa_loader import (
    ALL_CONTEXT_LENGTHS,
    load_ts_haystack_splits,
    get_available_tasks,
    get_available_context_lengths,
)
from src.datasets.capture24_haystack.qa_dataset import TSHaystackQADataset
from src.datasets.capture24_haystack.cot_qa_dataset import TSHaystackCoTQADataset
from src.datasets.capture24_haystack.oracle_qa_dataset import TSHaystackOracleQADataset

# =============================================================================
# CoT Rationale Generation
# =============================================================================
from src.datasets.capture24_haystack.cot import (
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
    # Phase 4: Dataset & QADataset
    "TSHaystackQADataset",
    "TSHaystackCoTQADataset",
    "TSHaystackOracleQADataset",
    "ALL_CONTEXT_LENGTHS",
    "load_ts_haystack_splits",
    "get_available_tasks",
    "get_available_context_lengths",
    # CoT Generation
    "OpenAICoTClient",
    "TSHaystackCoTGenerator",
    "create_accelerometer_plot",
    "create_cot_prompt",
]