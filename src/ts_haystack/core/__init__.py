# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Core infrastructure for TS-Haystack benchmark.

- Data structures for timelines, bouts, and indices
- Seed management for reproducibility
- Timeline building from Capture-24 data
- Cross-participant bout indexing
- Activity transition matrix

- Background sampling using bout index
- Needle sampling with duration filtering
- Style transfer (covariance projection + boundary blending)
- Prompt template bank for NL diversity
"""

from ts_haystack.core.data_structures import (
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
)
from ts_haystack.core.seed_manager import (
    ReproducibilityConfig,
    SeedManager,
)
from ts_haystack.core.timeline_builder import (
    TimelineBuilder,
    get_timeline_path,
    get_timelines_dir,
)
from ts_haystack.core.bout_indexer import (
    BoutIndexer,
    get_bout_index_path,
)
from ts_haystack.core.transition_matrix import (
    TransitionMatrix,
    get_transition_matrix_path,
)
from ts_haystack.core.background_sampler import (
    BackgroundSampler,
)
from ts_haystack.core.needle_sampler import (
    NeedleSampler,
)
from ts_haystack.core.style_transfer import (
    StyleTransfer,
)
from ts_haystack.core.prompt_templates import (
    PromptTemplateBank,
    TemplateVariant,
)
from ts_haystack.core.activity_regimes import (
    WILLETTS_ACTIVITY_REGIMES,
    ACTIVITY_TO_REGIME,
    ALL_ACTIVITIES,
    get_regime,
    get_regime_activities,
    get_same_regime_activities,
    get_distractor_candidates,
    get_other_regime_activities,
    get_regime_for_activities,
    filter_activities_by_regime,
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
    # Phase 2: Sampling structures
    "SignalStatistics",
    "NeedleSample",
    "BackgroundSample",
    # Seed management
    "SeedManager",
    "ReproducibilityConfig",
    # Timeline builder
    "TimelineBuilder",
    "get_timeline_path",
    "get_timelines_dir",
    # Bout indexer
    "BoutIndexer",
    "get_bout_index_path",
    # Transition matrix
    "TransitionMatrix",
    "get_transition_matrix_path",
    # Phase 2: Samplers
    "BackgroundSampler",
    "NeedleSampler",
    "StyleTransfer",
    "PromptTemplateBank",
    "TemplateVariant",
    # Activity regimes
    "WILLETTS_ACTIVITY_REGIMES",
    "ACTIVITY_TO_REGIME",
    "ALL_ACTIVITIES",
    "get_regime",
    "get_regime_activities",
    "get_same_regime_activities",
    "get_distractor_candidates",
    "get_other_regime_activities",
    "get_regime_for_activities",
    "filter_activities_by_regime",
]
