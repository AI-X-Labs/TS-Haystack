"""Task registry for UK-DALE-Haystack generators."""
from src.datasets.uk_dale_haystack.tasks.task_anomaly_detection import (
    AnomalyDetectionTaskGenerator,
)
from src.datasets.uk_dale_haystack.tasks.task_anomaly_localization import (
    AnomalyLocalizationTaskGenerator,
)
from src.datasets.uk_dale_haystack.tasks.task_antecedent import AntecedentTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_comparison import ComparisonTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_counting import CountingTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_existence import ExistenceTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_localization import LocalizationTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_multi_hop import MultiHopTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_ordering import OrderingTaskGenerator
from src.datasets.uk_dale_haystack.tasks.task_state_query import StateQueryTaskGenerator

TASK_REGISTRY = {
    "existence":              ExistenceTaskGenerator,
    "localization":           LocalizationTaskGenerator,
    "counting":               CountingTaskGenerator,
    "ordering":               OrderingTaskGenerator,
    "antecedent":             AntecedentTaskGenerator,
    "comparison":             ComparisonTaskGenerator,
    "multi_hop":              MultiHopTaskGenerator,
    "state_query":            StateQueryTaskGenerator,
    "anomaly_detection":      AnomalyDetectionTaskGenerator,
    "anomaly_localization":   AnomalyLocalizationTaskGenerator,
}

__all__ = ["TASK_REGISTRY"]
