# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Task generators for the natural annotation-based Sleep PSG benchmark.

Sleep stages dataset: 7 tasks
Arousals dataset: 6 tasks (no state_query)

1. Localization: "When did the Nth {activity} occur?" → time_range
2. Counting: "How many {activity} bouts?" → integer
3. Ordering: "Did the Nth {A} occur before the Mth {B}?" → boolean
4. State Query: "What sleep stage during the Nth {arousal}?" → category (sleep_stages only)
5. Antecedent: "What came before the Nth {activity}?" → category
6. Comparison: "What was the longest {activity}?" → time_range
7. Multi-Hop: "When did the Kth {target} occur {direction} the Nth {anchor}?" → time_range
"""

from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_existence import PSGExistenceTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_localization import PSGLocalizationTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_counting import PSGCountingTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_ordering import PSGOrderingTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_state_query import PSGStateQueryTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_antecedent import PSGAntecedentTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_comparison import PSGComparisonTaskGenerator
from src.datasets.sleep_psg_haystack.tasks.task_multi_hop import PSGMultiHopTaskGenerator

PSG_TASK_REGISTRY = {
    "existence": PSGExistenceTaskGenerator,
    "localization": PSGLocalizationTaskGenerator,
    "counting": PSGCountingTaskGenerator,
    "ordering": PSGOrderingTaskGenerator,
    "state_query": PSGStateQueryTaskGenerator,
    "antecedent": PSGAntecedentTaskGenerator,
    "comparison": PSGComparisonTaskGenerator,
    "multi_hop": PSGMultiHopTaskGenerator,
}

TASKS_BY_LABEL_CLASS = {
    "sleep_stages": [
        "existence", "localization", "counting", "ordering", "state_query",
        "antecedent", "comparison", "multi_hop",
    ],
    "arousals": [
        "existence", "localization", "counting", "ordering",
        "antecedent", "comparison", "multi_hop",
    ],
}


def get_psg_task_generator(task_name: str) -> type:
    if task_name not in PSG_TASK_REGISTRY:
        available = ", ".join(sorted(PSG_TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return PSG_TASK_REGISTRY[task_name]


def get_tasks_for_label_class(label_class: str) -> list:
    """Return the list of applicable task names for a label class."""
    return TASKS_BY_LABEL_CLASS.get(label_class, list(PSG_TASK_REGISTRY.keys()))
