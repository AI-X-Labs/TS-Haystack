# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Task generators for TS-Haystack benchmark.

This module provides task generators for 8 retrieval and reasoning tasks
over long time series data:

1. Existence: "Is there {activity} in this recording?"
2. Localization: "When did the {activity} bout occur?"
3. Counting: "How many {activity} bouts occurred?"
4. Ordering: "Did {activity_a} occur before {activity_b}?"
5. State Query: "What was the activity level when {event} occurred?"
6. Antecedent: "What activity occurred before {target}?"
7. Comparison: "What was the longest/shortest period with {activity}?"
8. Multi-Hop: "When did the Kth {target} occur after {anchor}?"

Each generator follows the same pattern:
- Dependency injection of Phase 2 components
- Single RNG per sample for reproducibility
- Outputs GeneratedSample objects
- Supports batch generation with retry handling
"""

from ts_haystack.tasks.base_task import (
    BaseTaskGenerator,
)
from ts_haystack.tasks.task_existence import (
    ExistenceTaskGenerator,
)
from ts_haystack.tasks.task_localization import (
    LocalizationTaskGenerator,
)
from ts_haystack.tasks.task_counting import (
    CountingTaskGenerator,
)
from ts_haystack.tasks.task_ordering import (
    OrderingTaskGenerator,
)
from ts_haystack.tasks.task_state_query import (
    StateQueryTaskGenerator,
)
from ts_haystack.tasks.task_antecedent import (
    AntecedentTaskGenerator,
)
from ts_haystack.tasks.task_comparison import (
    ComparisonTaskGenerator,
)
from ts_haystack.tasks.task_multi_hop import (
    MultiHopTaskGenerator,
)
from ts_haystack.tasks.task_anomaly_detection import (
    AnomalyDetectionTaskGenerator,
)
from ts_haystack.tasks.task_anomaly_localization import (
    AnomalyLocalizationTaskGenerator,
)

# Registry of all available task generators
TASK_REGISTRY = {
    "existence": ExistenceTaskGenerator,
    "localization": LocalizationTaskGenerator,
    "counting": CountingTaskGenerator,
    "ordering": OrderingTaskGenerator,
    "state_query": StateQueryTaskGenerator,
    "antecedent": AntecedentTaskGenerator,
    "comparison": ComparisonTaskGenerator,
    "multi_hop": MultiHopTaskGenerator,
    "anomaly_detection": AnomalyDetectionTaskGenerator,
    "anomaly_localization": AnomalyLocalizationTaskGenerator,
}


def get_task_generator(task_name: str) -> type:
    """
    Get task generator class by name.

    Args:
        task_name: Task identifier (e.g., "existence", "localization")

    Returns:
        Task generator class

    Raises:
        ValueError: If task_name is not recognized
    """
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {available}"
        )
    return TASK_REGISTRY[task_name]


def list_available_tasks() -> list:
    """Return list of available task names."""
    return sorted(TASK_REGISTRY.keys())


__all__ = [
    # Base class
    "BaseTaskGenerator",
    # Task generators
    "ExistenceTaskGenerator",
    "LocalizationTaskGenerator",
    "CountingTaskGenerator",
    "OrderingTaskGenerator",
    "StateQueryTaskGenerator",
    "AntecedentTaskGenerator",
    "ComparisonTaskGenerator",
    "MultiHopTaskGenerator",
    "AnomalyDetectionTaskGenerator",
    "AnomalyLocalizationTaskGenerator",
    # Registry functions
    "TASK_REGISTRY",
    "get_task_generator",
    "list_available_tasks",
]
