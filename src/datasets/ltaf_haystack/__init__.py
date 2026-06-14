# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
LTAF-Haystack: windowed-natural long-context ECG QA benchmark.

Every sample is an unmodified 2-lead slice of a real LTAF recording;
questions are generated from whatever rhythm/beat annotations fall inside
the window.

This package contains:
- core: ECG timeline/bout/window-index infrastructure
- tasks: 10 natural QA tasks over rhythm + beat annotations
- generation: config and orchestration helpers
- qa_loader/qa_dataset: runtime integration with QADataset pipeline
"""

from src.datasets.ltaf_haystack.qa_loader import (
    ALL_CONTEXT_LENGTHS,
    load_ltaf_haystack_splits,
    get_available_context_lengths,
    get_available_tasks,
)
from src.datasets.ltaf_haystack.qa_dataset import LTAFHaystackQADataset
from src.datasets.ltaf_haystack.cot_qa_dataset import LTAFHaystackCoTQADataset

__all__ = [
    "LTAFHaystackQADataset",
    "LTAFHaystackCoTQADataset",
    "ALL_CONTEXT_LENGTHS",
    "load_ltaf_haystack_splits",
    "get_available_context_lengths",
    "get_available_tasks",
]
