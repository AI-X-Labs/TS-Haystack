# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Dataset implementations for TS-Haystack."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.datasets.base import BaseDataset
    from src.datasets.qa_base import QADataset
    from src.datasets.registry import DATASET_REGISTRY, get_dataset_class, list_datasets


def __getattr__(name: str) -> Any:
    if name == "BaseDataset":
        from src.datasets.base import BaseDataset

        return BaseDataset
    if name == "QADataset":
        from src.datasets.qa_base import QADataset

        return QADataset
    if name in {"DATASET_REGISTRY", "get_dataset_class", "list_datasets"}:
        import src.datasets.registry as registry

        return getattr(registry, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "BaseDataset",
    "QADataset",
    "DATASET_REGISTRY",
    "get_dataset_class",
    "list_datasets",
]
