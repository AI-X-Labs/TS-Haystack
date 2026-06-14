# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Dataset registry mapping dataset names to QADataset subclasses."""

from src.datasets.qa_base import QADataset
from src.datasets.capture24_haystack.qa_dataset import TSHaystackQADataset
from src.datasets.capture24_haystack.cot_qa_dataset import TSHaystackCoTQADataset
from src.datasets.capture24.qa_dataset import Capture24AccQADataset
from src.datasets.ltaf_haystack.qa_dataset import LTAFHaystackQADataset
from src.datasets.ltaf_haystack.cot_qa_dataset import LTAFHaystackCoTQADataset
from src.datasets.sleep_psg_haystack.qa_dataset import SleepPSGHaystackQADataset
from src.datasets.uk_dale_haystack.qa_dataset import UKDaleHaystackQADataset

DATASET_REGISTRY: dict[str, type[QADataset]] = {
    "capture24_haystack_classification": TSHaystackQADataset,
    "capture24_haystack_cot": TSHaystackCoTQADataset,
    "capture24_classification": Capture24AccQADataset,
    "ltaf_haystack": LTAFHaystackQADataset,
    "ltaf_haystack_cot": LTAFHaystackCoTQADataset,
    "sleep_psg_haystack": SleepPSGHaystackQADataset,
    "uk_dale_haystack": UKDaleHaystackQADataset,
}


def get_dataset_class(name: str) -> type[QADataset]:
    """Look up a dataset class by name.

    Args:
        name: Registry key (e.g. 'capture24_haystack_cot').

    Returns:
        The dataset class.

    Raises:
        KeyError: If the dataset name is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


def list_datasets() -> list[str]:
    """Return all registered dataset names."""
    return list(DATASET_REGISTRY.keys())
