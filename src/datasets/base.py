# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Base dataset class defining the interface for all TS-Haystack datasets.
WARNING: Currently not in use, since all the applications of TSLM have been using QA datasets - keeping in case future datasets require it.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets in TS-Haystack.

    All dataset implementations must inherit from this class and implement
    the required abstract methods to ensure compatibility with the training
    infrastructure.

    Sample Output Format (Standardized):
        All datasets should return samples with these keys:

        Required for all:
            - time_series: Tensor of shape (n_channels, seq_len)
            - answer: str - Ground truth answer

        Required for QA tasks:
            - pre_prompt: str - Context before time series
            - post_prompt: str - Instructions after time series
            - question: str - The question being asked

        Optional:
            - rationale: str | None - CoT rationale (for CoT-trained models)
            - task_type: str - e.g., "existence", "classification"
            - answer_type: str - e.g., "boolean", "category", "integer"
            - sample_id: str - Unique identifier
            - context_length_samples: int - Window size in samples
            - needles: list | None - Inserted needle metadata
            - difficulty_config: dict | None - Generation parameters
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample dictionary with standardized keys.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing at minimum 'time_series' and 'answer' keys.
        """
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable | None:
        """Return a collate function for the DataLoader.

        Returns:
            A callable that takes a list of samples and returns a batched
            dictionary, or None to use the default PyTorch collate.
        """
        pass

    @classmethod
    @abstractmethod
    def download(cls, data_dir: str) -> None:
        """Download the dataset to the specified directory.

        Args:
            data_dir: Path to the directory where data should be stored.
        """
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the type of task this dataset represents.

        Returns:
            One of: 'classification', 'qa', 'retrieval', 'generation'
        """
        pass

    @property
    @abstractmethod
    def classes(self) -> list[str] | None:
        """Return the list of class labels for classification tasks.

        Returns:
            List of class label strings, or None if not a classification task.
        """
        pass

    @property
    def sample_rate(self) -> float | None:
        """Return the sampling rate of the time series in Hz.

        Returns:
            Sampling rate in Hz, or None if not applicable.
        """
        return None

    @property
    def num_channels(self) -> int | None:
        """Return the number of channels in the time series.

        Returns:
            Number of channels, or None if variable.
        """
        return None

    def get_split(self, split: str) -> "BaseDataset":
        """Return a dataset for the specified split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            Dataset instance for the requested split.

        Raises:
            NotImplementedError: If splits are not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support splits")

    @property
    def num_classes(self) -> int | None:
        """Return the number of classes (convenience property).

        Returns:
            Length of classes list, or None if not a classification task.
        """
        return len(self.classes) if self.classes is not None else None
