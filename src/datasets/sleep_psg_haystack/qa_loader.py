# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Loader for the Sleep PSG TS-Haystack benchmark.

Discovers parquet splits under
``data/sleep_psg/ts_haystack/{label_class}/tasks/{ctx_dir}/{task}/{split}/data.parquet``
and exposes them as :class:`LazyParquetDataset` (reused from the Capture-24
TS-Haystack loader). Signals are *not* loaded here — they are slurped on demand
inside :class:`SleepPSGHaystackQADataset` via ``loader.load_window``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import pyarrow.parquet as pq

from src.datasets.capture24_haystack.qa_loader import LazyParquetDataset
from src.datasets.capture24_haystack.utils import format_context_dir


# Tasks present (or potentially present) under data/sleep_psg/ts_haystack/{label_class}/tasks/.
ALL_TASKS = [
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
]

# Per-label-class context length grid (matches what the generator produced).
ALL_CONTEXT_LENGTHS_PER_LABEL = {
    "sleep_stages": [900, 3600, 7200, "full"],
    "arousals": [100, 900, 3600, "full"],
}

REQUIRED_COLUMNS = [
    "subject_id",
    "recording_duration_ms",
    "label_class",
    "task_type",
    "context_length_s",
    "window_start_ms",
    "window_end_ms",
    "question",
    "answer",
    "answer_type",
    "metadata",
]

BASE_DIR = Path("data/sleep_psg/ts_haystack")
ALL_CONTEXT_LENGTHS = "all"


def _format_ctx_dir(ctx: Union[int, float, str]) -> str:
    """Convert a context length entry into the on-disk directory name."""
    if isinstance(ctx, str):
        return ctx  # "full"
    return format_context_dir(ctx)


def load_sleep_psg_haystack_splits(
    label_class: str,
    tasks: List[str],
    context_lengths_seconds: List[Union[int, float, str]],
    base_dir: Path = BASE_DIR,
) -> Tuple[LazyParquetDataset, LazyParquetDataset, LazyParquetDataset]:
    """
    Discover parquet files for ``label_class`` and return (train, val, test)
    :class:`LazyParquetDataset` instances. Missing (task, ctx) combinations
    are skipped silently.
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(context_lengths_seconds, (str, int, float)):
        context_lengths_seconds = [context_lengths_seconds]

    if "all" in tasks:
        tasks = list(ALL_TASKS)

    if "all" in [str(c) for c in context_lengths_seconds]:
        context_lengths_seconds = list(ALL_CONTEXT_LENGTHS_PER_LABEL[label_class])

    label_dir = Path(base_dir) / label_class / "tasks"
    if not label_dir.exists():
        raise FileNotFoundError(
            f"Sleep PSG haystack tree not found at {label_dir}. "
            f"Generate it before training."
        )

    split_files: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ctx in context_lengths_seconds:
        ctx_dir = _format_ctx_dir(ctx)
        for task in tasks:
            for split in ("train", "val", "test"):
                p = label_dir / ctx_dir / task / split / "data.parquet"
                if not p.exists():
                    continue
                split_files[split].append(str(p))
                num_rows = pq.read_metadata(p).num_rows
                print(
                    f"Found {p.relative_to(base_dir)}: {num_rows} samples"
                )

    datasets = []
    for split in ("train", "val", "test"):
        files = split_files[split]
        if files:
            ds = LazyParquetDataset(files, REQUIRED_COLUMNS)
            print(f"Total {split}: {len(ds)} samples (lazy loading)")
        else:
            print(f"Warning: No data found for {split} split, creating empty dataset")
            ds = LazyParquetDataset([], REQUIRED_COLUMNS)
        datasets.append(ds)

    return tuple(datasets)
