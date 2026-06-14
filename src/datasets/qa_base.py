# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List, Literal, Tuple

import numpy as np
from src.prompt.prompt_with_answer import PromptWithAnswer
from src.prompt.text_prompt import TextPrompt
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class QADataset(Dataset, ABC):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
    ):
        """
        Initializes the dataset by loading and formatting the specified data split.
        Args:
            split (Literal["train", "test", "validation"]): The dataset split to load. Must be one of "train", "test", or "validation".
            EOS_TOKEN (str): End-of-sequence token to be used in formatting.
            format_sample_str (bool, optional): If True, applies a string formatting function to each sample. Defaults to False.
            time_series_format_function (Callable[[np.ndarray], str] | None, optional): Optional function to format time series data as strings. Used only if `format_sample_str` is True.
            lazy_loading (bool, optional): If True (default), samples are formatted on-demand to save memory.
                                           If False, all samples are pre-formatted into memory (legacy behavior).
        Raises:
            RuntimeError: If the provided split is not one of "train", "test", or "validation".
        Notes:
            - With lazy_loading=True (default), only raw data is kept in memory-mapped HuggingFace Dataset format.
              Samples are formatted when accessed via __getitem__. This significantly reduces memory usage.
            - With lazy_loading=False, datasets are loaded and formatted once per class and cached as class attributes.
        """

        self.EOS_TOKEN = EOS_TOKEN
        self._format_sample_str_flag = format_sample_str
        self._time_series_format_function = time_series_format_function
        self._lazy_loading = lazy_loading

        if lazy_loading:
            # Lazy loading: keep raw datasets in a per-configuration cache.
            cache = self.__class__.__dict__.get("_raw_split_cache")
            if cache is None:
                self.__class__._raw_split_cache = {}
                cache = self.__class__._raw_split_cache

            raw_key = self._raw_cache_key()
            if raw_key not in cache:
                train, val, test = self._load_splits()
                cache[raw_key] = (train, val, test)
                print(f"Loaded raw datasets - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

            train, val, test = cache[raw_key]

            match split:
                case "train":
                    self._raw_dataset = train
                case "validation":
                    self._raw_dataset = val
                case "test":
                    self._raw_dataset = test
                case _:
                    raise RuntimeError(
                        "Split must be a literal of 'train', 'test', or 'validation'"
                    )
            self.dataset = None  # Not used in lazy mode
        else:
            # Legacy behavior: pre-format all samples into memory.
            # Cache by both data configuration and formatting options.
            cache = self.__class__.__dict__.get("_formatted_split_cache")
            if cache is None:
                self.__class__._formatted_split_cache = {}
                cache = self.__class__._formatted_split_cache

            formatted_key = self._formatted_cache_key(
                format_sample_str=format_sample_str,
                time_series_format_function=time_series_format_function,
            )

            if formatted_key not in cache:
                train, val, test = self._load_splits()

                format_function = (
                    partial(self._format_sample_str, time_series_format_function)
                    if format_sample_str
                    else self._format_sample
                )

                from tqdm import tqdm

                print("Formatting training samples...")
                train_dataset = list(
                    tqdm(map(format_function, train), total=len(train), desc="Training samples")
                )

                print("Formatting validation samples...")
                validation_dataset = list(
                    tqdm(map(format_function, val), total=len(val), desc="Validation samples")
                )

                print("Formatting test samples...")
                test_dataset = list(
                    tqdm(map(format_function, test), total=len(test), desc="Test samples")
                )

                cache[formatted_key] = (train_dataset, validation_dataset, test_dataset)

            train_dataset, validation_dataset, test_dataset = cache[formatted_key]

            match split:
                case "train":
                    self.dataset = train_dataset
                case "validation":
                    self.dataset = validation_dataset
                case "test":
                    self.dataset = test_dataset
                case _:
                    raise RuntimeError(
                        "Split must be a literal of 'train', 'test', or 'validation'"
                    )
            self._raw_dataset = None  # Not used in eager mode

    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @abstractmethod
    def _get_answer(self, row) -> str:
        pass

    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        pass

    def _dataset_cache_key(self) -> tuple[Any, ...]:
        """Key representing arguments that affect _load_splits().

        Subclasses should override this when constructor options change the
        underlying split contents (e.g. selected tasks or context lengths).
        """
        return ()

    def _raw_cache_key(self) -> tuple[Any, ...]:
        return (self.__class__.__qualname__, self._dataset_cache_key())

    def _formatted_cache_key(
        self,
        format_sample_str: bool,
        time_series_format_function: Callable[[np.ndarray], str] | None,
    ) -> tuple[Any, ...]:
        return (
            self.__class__.__qualname__,
            self._dataset_cache_key(),
            bool(format_sample_str),
            self._callable_cache_key(time_series_format_function),
            self.EOS_TOKEN,
        )

    @staticmethod
    def _callable_cache_key(
        fn: Callable[[np.ndarray], str] | None,
    ) -> tuple[Any, ...] | None:
        if fn is None:
            return None

        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", None))
        if module and qualname:
            return (module, qualname)

        return (type(fn).__module__, type(fn).__qualname__, id(fn))

    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN

        return PromptWithAnswer(
            TextPrompt(self._get_pre_prompt(row).strip()),
            self._get_text_time_series_prompt_list(row),
            TextPrompt(self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()

    def _format_sample_str(
        self, time_series_format_function: Callable[[np.ndarray], str] | None, row
    ):
        def fallback_timeseries_formatter(time_series: np.ndarray) -> str:
            # Fallback formatter for time series data
        
            return np.array2string(
                time_series,
                separator=" ",
                formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
                threshold=sys.maxsize,
                max_line_width=sys.maxsize,
            ).removeprefix("[").removesuffix("]")

               

        if not time_series_format_function:
            time_series_format_function = fallback_timeseries_formatter

        # Create the prompt chunks: pre-prompt, time series prompts, and post-prompt
        prompt_chunks = [self._get_pre_prompt(row).strip()]

        for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
            prompt_chunks.append(text_time_series_prompt.get_text())
            time_series = time_series_format_function(
                text_time_series_prompt.get_time_series()
            )
            prompt_chunks.append(time_series)

        prompt_chunks.append(self._get_post_prompt(row).strip())

        return {"prompt": "\n".join(prompt_chunks), "answer": self._get_answer(row)}

    # ------------------------------------------------------------------
    # Evaluation helpers (override per-dataset for custom logic)
    # ------------------------------------------------------------------

    @property
    def primary_metric(self) -> str:
        """The metric used for aggregate evaluation reporting.

        "accuracy" (default) — QA/classification tasks, aggregated as sum(correct)/total.
        Any other string (e.g. "bleu") — averaged across samples from evaluate_answer() dicts.
        """
        return "accuracy"

    @property
    def category_key(self) -> str | None:
        """Sample dict key used for per-category accuracy grouping.

        Returns None if no per-category breakdown is meaningful.
        Override in subclasses to use a different key (e.g. "question_type").
        """
        return None

    def get_ground_truth(self, sample: dict) -> str:
        """Extract the clean ground truth answer for display/logging.

        Override for dataset-specific ground truth extraction.
        Default: sample["direct_answer"] or extract "Answer: ..." from sample["answer"].
        """
        gt = sample.get("direct_answer") or sample.get("answer", "")
        gt = gt.replace(self.EOS_TOKEN, "").strip()
        if "Answer:" in gt:
            gt = gt.split("Answer:")[-1].strip().rstrip(".")
        return gt

    def extract_answer(self, prediction: str, sample: dict) -> str:
        """Extract the final answer from a model prediction.

        Override for dataset-specific extraction (e.g., CoT "Answer:" parsing).
        Default: return the prediction stripped.
        """
        return prediction.strip()

    def evaluate_answer(self, prediction: str, sample: dict) -> dict:
        """Evaluate a predicted answer against the ground truth.

        Override for dataset-specific evaluation (e.g., IoU for time ranges).
        Default: exact string match against sample["answer"] (with EOS stripped).

        Returns:
            Dict with at minimum {"correct": bool}. May include extra keys
            like "iou", "normalized_gt", "normalized_pred" for dataset-specific metrics.
        """
        gt = sample.get("direct_answer") or sample.get("answer", "")
        gt = gt.replace(self.EOS_TOKEN, "").strip().lower()
        pred = prediction.strip().lower()
        return {"correct": gt == pred}

    def __len__(self):
        if self._lazy_loading:
            return len(self._raw_dataset)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._lazy_loading:
            # Format sample on-demand
            row = self._raw_dataset[idx]
            if self._format_sample_str_flag:
                return self._format_sample_str(self._time_series_format_function, row)
            return self._format_sample(row)
        return self.dataset[idx]
