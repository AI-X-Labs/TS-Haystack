# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""CoT QADataset scaffolding for LTAF-Haystack."""

from typing import Callable, Literal
from pathlib import Path

import numpy as np

from src.datasets.ltaf_haystack.qa_dataset import LTAFHaystackQADataset


class LTAFHaystackCoTQADataset(LTAFHaystackQADataset):
    """Variant that expects rationale text with final Answer: marker."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        tasks: list[str] | None = None,
        context_lengths_seconds: list[float | str] | None = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
        lazy_loading: bool = True,
        base_dir: str | Path | None = None,
    ):
        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            tasks=tasks,
            context_lengths_seconds=context_lengths_seconds,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
            lazy_loading=lazy_loading,
            use_cot=True,
            base_dir=base_dir,
        )

    def _get_answer(self, row) -> str:
        if row.get("rationale"):
            return str(row["rationale"])
        return super()._get_answer(row)

    def _get_post_prompt(self, row) -> str:
        base = super()._get_post_prompt(row)
        return (
            f"{base}\\n"
            "Write your reasoning as a short paragraph and end with 'Answer: <final answer>'."
        )
