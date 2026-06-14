# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""LTAF-Haystack task registry (10 tasks).

1. existence              — "Is there {activity} in this ECG window?" → boolean
2. localization           — "When did the Nth {activity} occur?" → time range
3. counting               — "How many {activity} bouts?" → integer
4. ordering               — "Did the Nth {A} occur before the Mth {B}?" → boolean
5. state_query            — "What rhythm at the Nth {V/A/Q} beat?" → category
6. antecedent             — "Which rhythm preceded the Nth {activity}?" → category
7. comparison             — "Which appeared more: {A} or {B}?" → category
8. multi_hop              — "Kth {target} {direction} the Nth {anchor}?" → time range
9. anomaly_detection      — "Does this window contain V/A beats?" → boolean
10. anomaly_localization  — "Where is the Nth V/A beat?" → timestamp
"""

from typing import List, Type

from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_existence import LTAFExistenceTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_localization import LTAFLocalizationTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_counting import LTAFCountingTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_ordering import LTAFOrderingTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_state_query import LTAFStateQueryTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_antecedent import LTAFAntecedentTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_comparison import LTAFComparisonTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_multi_hop import LTAFMultiHopTaskGenerator
from src.datasets.ltaf_haystack.tasks.task_anomaly_detection import (
    LTAFAnomalyDetectionTaskGenerator,
)
from src.datasets.ltaf_haystack.tasks.task_anomaly_localization import (
    LTAFAnomalyLocalizationTaskGenerator,
)


TASK_REGISTRY: dict[str, Type[LTAFBaseTaskGenerator]] = {
    "existence": LTAFExistenceTaskGenerator,
    "localization": LTAFLocalizationTaskGenerator,
    "counting": LTAFCountingTaskGenerator,
    "ordering": LTAFOrderingTaskGenerator,
    "state_query": LTAFStateQueryTaskGenerator,
    "antecedent": LTAFAntecedentTaskGenerator,
    "comparison": LTAFComparisonTaskGenerator,
    "multi_hop": LTAFMultiHopTaskGenerator,
    "anomaly_detection": LTAFAnomalyDetectionTaskGenerator,
    "anomaly_localization": LTAFAnomalyLocalizationTaskGenerator,
}

TASKS_BY_LABEL_CLASS: dict[str, List[str]] = {
    "rhythms": list(TASK_REGISTRY.keys()),
}


def get_task_generator(task_name: str) -> Type[LTAFBaseTaskGenerator]:
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return TASK_REGISTRY[task_name]


def get_tasks_for_label_class(label_class: str) -> List[str]:
    return list(TASKS_BY_LABEL_CLASS.get(label_class, list(TASK_REGISTRY.keys())))


def list_available_tasks() -> List[str]:
    return sorted(TASK_REGISTRY.keys())


__all__ = [
    "LTAFBaseTaskGenerator",
    "LTAFExistenceTaskGenerator",
    "LTAFLocalizationTaskGenerator",
    "LTAFCountingTaskGenerator",
    "LTAFOrderingTaskGenerator",
    "LTAFStateQueryTaskGenerator",
    "LTAFAntecedentTaskGenerator",
    "LTAFComparisonTaskGenerator",
    "LTAFMultiHopTaskGenerator",
    "LTAFAnomalyDetectionTaskGenerator",
    "LTAFAnomalyLocalizationTaskGenerator",
    "TASK_REGISTRY",
    "TASKS_BY_LABEL_CLASS",
    "get_task_generator",
    "get_tasks_for_label_class",
    "list_available_tasks",
]
