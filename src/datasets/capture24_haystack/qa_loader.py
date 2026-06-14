# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-FileCopyrightText: 2026 Anonymous Authors
#
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Loader for TS-Haystack benchmark data in HuggingFace Dataset format.

This module bridges the generated parquet files to the QADataset interface
by converting them to HuggingFace Dataset objects with the expected schema.

Usage:
    from src.datasets.capture24_haystack import load_ts_haystack_splits

    # Load single task at single context length
    train, val, test = load_ts_haystack_splits(
        tasks=["existence"],
        context_lengths_seconds=[100],
    )

    # Load multiple tasks at multiple context lengths
    train, val, test = load_ts_haystack_splits(
        tasks=["existence", "localization", "counting"],
        context_lengths_seconds=[100, 1000],
    )

    # Load with CoT rationales
    train, val, test = load_ts_haystack_splits(
        tasks=["existence"],
        context_lengths_seconds=[100],
        use_cot=True,
    )
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset as TorchDataset

from src.datasets.constants import RAW_DATA
from src.datasets.capture24_haystack.utils import format_context_dir


# Signals sidecar filename — memmap-friendly, per-file companion holding the
# (N, 3, L) float32 accelerometer signals. When present alongside a
# ``data.parquet``, :class:`LazyParquetDataset` reads signals from this memmap
# instead of from the inline ``x_axis/y_axis/z_axis`` parquet columns. Mirrors
# the LTAF / Sleep-PSG loading pattern (tiny parquet + memmap'd signals).
SIGNALS_SIDECAR_NAME = "signals.npy"
_SIGNAL_COLUMNS = ("x_axis", "y_axis", "z_axis")


# Default data directories
TS_HAYSTACK_TASKS_DIR = os.path.join(RAW_DATA, "capture24", "ts_haystack", "tasks")
TS_HAYSTACK_COT_DIR = os.path.join(RAW_DATA, "capture24", "ts_haystack", "cot")
TS_HAYSTACK_COT_SMALL_DIR = os.path.join(RAW_DATA, "capture24", "ts_haystack", "cot_small")

# Sentinel value for "all context lengths"
ALL_CONTEXT_LENGTHS = "all"

# All available tasks
ALL_TASKS = [
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
    "anomaly_detection",
    "anomaly_localization",
]

# Columns required for QADataset
REQUIRED_COLUMNS = [
    "x_axis",
    "y_axis",
    "z_axis",
    "task_type",
    "context_length_samples",
    "recording_time_start",
    "recording_time_end",
    "question",
    "answer",
    "answer_type",
    "needles",
    "difficulty_config",
]

# Additional column for CoT datasets
COT_COLUMN = "rationale"


def get_available_tasks() -> List[str]:
    """Return list of available task names."""
    return list(ALL_TASKS)


def get_available_context_lengths(use_cot: bool = False, base_dir: Optional[Path] = None, cot_small: bool = False) -> List[float]:
    """
    Discover available context lengths from filesystem.

    Args:
        use_cot: If True, look in CoT directory, else tasks directory
        base_dir: Optional override for base directory
        cot_small: If True and use_cot=True, use cot_small directory

    Returns:
        List of context lengths in seconds, sorted ascending
    """
    data_dir = _get_data_dir(use_cot, base_dir, cot_small)

    if not data_dir.exists():
        return []

    context_lengths = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.endswith("s"):
            # Parse directory name like "100s" or "2_56s" -> 100.0 or 2.56
            try:
                # Replace underscore with decimal point for names like "2_56s"
                name = item.name[:-1]  # Remove trailing 's'
                name = name.replace("_", ".")
                context_lengths.append(float(name))
            except ValueError:
                continue

    return sorted(context_lengths)


def _get_data_dir(use_cot: bool, base_dir: Optional[Path] = None, cot_small: bool = False) -> Path:
    """Get the appropriate data directory based on whether CoT is used."""
    if base_dir is not None:
        return Path(base_dir)
    if use_cot:
        return Path(TS_HAYSTACK_COT_SMALL_DIR if cot_small else TS_HAYSTACK_COT_DIR)
    return Path(TS_HAYSTACK_TASKS_DIR)


def _get_columns_to_load(use_cot: bool, extra_columns: Optional[List[str]] = None) -> List[str]:
    """Get the list of columns to load from parquet files."""
    columns = list(REQUIRED_COLUMNS)
    if use_cot:
        columns.append(COT_COLUMN)
    if extra_columns:
        columns.extend(extra_columns)
    return columns


class LazyParquetDataset(TorchDataset):
    """
    A truly lazy dataset that reads individual rows from parquet files on demand.

    Uses an LRU cache to store loaded DataFrames. With sufficient RAM, all files
    can be cached for maximum performance. Files are loaded on first access and
    kept in memory for subsequent accesses.

    Memory usage scales with number of unique files accessed.
    """

    # Default cache size in bytes (10GB per worker - conservative to avoid OOM)
    DEFAULT_CACHE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10GB

    def __init__(self, parquet_files: List[str], columns: List[str],
                 cache_size_bytes: Optional[int] = None):
        """
        Args:
            parquet_files: List of parquet file paths
            columns: Columns to load from each file
            cache_size_bytes: Max memory for file cache.
                              None (default) = unlimited cache (recommended with lots of RAM)
                              0 = disable caching
                              >0 = limit cache to this many bytes
        """
        self.parquet_files = parquet_files
        self.columns = columns
        # None means unlimited cache
        self._cache_size_bytes = cache_size_bytes if cache_size_bytes is not None else self.DEFAULT_CACHE_SIZE_BYTES

        # Per-file: has a sibling signals.npy memmap? Path → cached later.
        self._sidecar_paths: List[Optional[str]] = []
        # Whether to read signal columns from parquet (only if no sidecar).
        self._parquet_columns_per_file: List[List[str]] = []

        # Build index: map global idx -> (file_idx, local_row_idx)
        self._file_row_counts = []
        self._file_sizes = []  # Approximate memory size per file
        self._cumulative_counts = [0]

        for f in parquet_files:
            metadata = pq.read_metadata(f)
            num_rows = metadata.num_rows
            self._file_row_counts.append(num_rows)
            self._cumulative_counts.append(self._cumulative_counts[-1] + num_rows)
            # Estimate memory size from file size (parquet is compressed, so multiply)
            file_size = Path(f).stat().st_size
            self._file_sizes.append(file_size * 3)  # Rough estimate: 3x compression

            # Detect sidecar memmap. When present, skip loading the inline
            # signal columns from parquet — signals are hydrated via memmap.
            sidecar = Path(f).with_name(SIGNALS_SIDECAR_NAME)
            if sidecar.exists():
                self._sidecar_paths.append(str(sidecar))
                self._parquet_columns_per_file.append(
                    [c for c in columns if c not in _SIGNAL_COLUMNS]
                )
            else:
                self._sidecar_paths.append(None)
                self._parquet_columns_per_file.append(list(columns))

        self._total_rows = self._cumulative_counts[-1]

        # LRU cache for loaded DataFrames: {file_idx: (dataframe, access_order)}
        self._df_cache: Dict[int, pd.DataFrame] = {}
        self._cache_order: List[int] = []  # Most recent at end
        self._current_cache_size = 0

        # Per-worker memmap cache (lazy). ``np.load(mmap_mode='r')`` is
        # process-local; workers build their own after pickling.
        self._mmap_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._total_rows

    def _get_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Convert global index to (file_index, local_row_index)."""
        if global_idx < 0 or global_idx >= self._total_rows:
            raise IndexError(f"Index {global_idx} out of range [0, {self._total_rows})")

        # Linear search (fast enough for typical number of files)
        for file_idx in range(len(self._file_row_counts)):
            if global_idx < self._cumulative_counts[file_idx + 1]:
                local_idx = global_idx - self._cumulative_counts[file_idx]
                return file_idx, local_idx

        raise IndexError(f"Index {global_idx} out of range")

    def _evict_if_needed(self, needed_size: int) -> None:
        """Evict oldest cached files until we have room for needed_size."""
        # Skip eviction if cache is unlimited
        if self._cache_size_bytes is None:
            return

        while (self._current_cache_size + needed_size > self._cache_size_bytes
               and self._cache_order):
            oldest_idx = self._cache_order.pop(0)
            if oldest_idx in self._df_cache:
                evicted_size = self._file_sizes[oldest_idx]
                del self._df_cache[oldest_idx]
                self._current_cache_size -= evicted_size

    def _get_dataframe(self, file_idx: int) -> Optional[pd.DataFrame]:
        """Get DataFrame for file, using cache. Returns None if file too large or caching disabled."""
        if file_idx in self._df_cache:
            # Move to end of access order (most recent)
            if file_idx in self._cache_order:
                self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._df_cache[file_idx]

        # Check if caching is disabled (cache_size_bytes == 0)
        if self._cache_size_bytes == 0:
            return None

        file_size = self._file_sizes[file_idx]

        # Check if file is too large to ever fit in cache
        if self._cache_size_bytes is not None and file_size > self._cache_size_bytes:
            return None

        # Evict old entries if needed
        self._evict_if_needed(file_size)

        # Read only needed columns (skip inline signal columns when a sidecar
        # memmap is present — signals are hydrated in __getitem__).
        df = pd.read_parquet(
            self.parquet_files[file_idx],
            columns=self._parquet_columns_per_file[file_idx],
            engine='pyarrow',
        )

        # Cache the DataFrame
        self._df_cache[file_idx] = df
        self._cache_order.append(file_idx)
        self._current_cache_size += file_size

        return df

    def _read_single_row(self, file_idx: int, local_idx: int) -> Dict:
        """Read a single row directly from parquet without caching the whole file."""
        # Use pyarrow to read just the row we need via row group iteration
        pf = pq.ParquetFile(self.parquet_files[file_idx])
        columns = self._parquet_columns_per_file[file_idx]

        # Find which row group contains our row
        rows_seen = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_num_rows = pf.metadata.row_group(rg_idx).num_rows
            if rows_seen + rg_num_rows > local_idx:
                # This row group contains our row
                row_in_rg = local_idx - rows_seen
                table = pf.read_row_group(rg_idx, columns=columns)
                row = table.slice(row_in_rg, 1).to_pydict()
                return {k: v[0] for k, v in row.items()}
            rows_seen += rg_num_rows

        raise IndexError(f"Row {local_idx} not found in file {self.parquet_files[file_idx]}")

    def _get_mmap(self, file_idx: int) -> Optional[np.ndarray]:
        """Return per-file memmap of signals (shape (N, 3, L)), or None."""
        sidecar = self._sidecar_paths[file_idx]
        if sidecar is None:
            return None
        mm = self._mmap_cache.get(file_idx)
        if mm is None:
            mm = np.load(sidecar, mmap_mode="r")
            self._mmap_cache[file_idx] = mm
        return mm

    def _attach_signals_from_mmap(
        self, row: Dict, file_idx: int, local_idx: int
    ) -> Dict:
        """If a sidecar memmap exists, slice row's signals into the row dict."""
        mm = self._get_mmap(file_idx)
        if mm is None:
            return row
        # mm shape: (N, 3, L) float32. Slice is a view; copy to detach from
        # the memmap so the returned row is safe after mmap eviction.
        sig = np.asarray(mm[local_idx], dtype=np.float32)
        row["x_axis"] = np.ascontiguousarray(sig[0])
        row["y_axis"] = np.ascontiguousarray(sig[1])
        row["z_axis"] = np.ascontiguousarray(sig[2])
        return row

    def __getitem__(self, idx: int) -> Dict:
        """Load a single row from the appropriate parquet file."""
        file_idx, local_idx = self._get_file_and_local_idx(idx)

        # Try to get from cache
        df = self._get_dataframe(file_idx)
        if df is not None:
            # Cache hit or file loaded into cache
            row = df.iloc[local_idx].to_dict()
        else:
            # Caching disabled (cache_size_bytes=0) - read single row directly
            row = self._read_single_row(file_idx, local_idx)

        return self._attach_signals_from_mmap(row, file_idx, local_idx)

    def column_names(self) -> List[str]:
        """Return column names (for compatibility with HF Dataset interface)."""
        return self.columns

    def clear_cache(self) -> None:
        """Clear the DataFrame cache to free memory."""
        self._df_cache.clear()
        self._cache_order.clear()
        self._current_cache_size = 0

    def __getstate__(self):
        """Prepare for pickling (multiprocessing). Clear cache to reduce memory."""
        state = self.__dict__.copy()
        # Clear cache - each worker will build its own
        state['_df_cache'] = {}
        state['_cache_order'] = []
        state['_current_cache_size'] = 0
        # Memmaps are not picklable; each worker will rebuild from file paths.
        state['_mmap_cache'] = {}
        return state

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)


def load_ts_haystack_splits(
    tasks: List[str],
    context_lengths_seconds: List[Union[str, float, int]],
    data_dir: Optional[Path] = None,
    use_cot: bool = False,
    cot_small: bool = False,
    extra_columns: Optional[List[str]] = None,
) -> Tuple[LazyParquetDataset, LazyParquetDataset, LazyParquetDataset]:
    """
    Load TS-Haystack datasets for specified tasks and context lengths.

    Uses truly lazy loading - data is only read from disk when individual
    samples are accessed via __getitem__. Memory usage is O(1) regardless
    of total dataset size.

    Directory structure expected:
        data_dir/{context_seconds}s/{task}/{split}/data.parquet

    Args:
        tasks: List of task names (e.g., ["existence", "localization"])
               Use ["all"] to load all tasks
        context_lengths_seconds: List of context lengths in seconds
                                 (e.g., [100] for 10000 samples at 100Hz)
                                 Use ["all"] to load all available context lengths
        data_dir: Base data directory. If None, uses default based on use_cot
        use_cot: If True, load from cot/ directory (with rationale column)
        cot_small: If True and use_cot=True, use cot_small/ directory instead

    Returns:
        Tuple of (train, val, test) LazyParquetDataset objects

    Example:
        >>> train, val, test = load_ts_haystack_splits(
        ...     tasks=["existence", "counting"],
        ...     context_lengths_seconds=[100],
        ... )
        >>> print(f"Train: {len(train)} samples")
    """
    # Normalize scalar strings to lists
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(context_lengths_seconds, (str, int, float)):
        context_lengths_seconds = [context_lengths_seconds]

    # Resolve tasks
    if tasks == ["all"] or "all" in tasks:
        tasks = ALL_TASKS

    # Resolve context lengths - support "all" to auto-discover
    if context_lengths_seconds == ["all"] or "all" in context_lengths_seconds:
        context_lengths_seconds = get_available_context_lengths(use_cot, data_dir, cot_small)
        if not context_lengths_seconds:
            raise ValueError(
                f"No context length directories found in "
                f"{_get_data_dir(use_cot, data_dir, cot_small)}. "
                f"Make sure datasets have been generated first."
            )
        print(f"Auto-discovered context lengths: {context_lengths_seconds}")

    # Validate tasks
    for task in tasks:
        if task not in ALL_TASKS:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {ALL_TASKS}"
            )

    # Get data directory
    base_dir = _get_data_dir(use_cot, data_dir, cot_small)

    # Collect parquet file paths for each split (NOT loading data yet)
    split_files: Dict[str, List[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    total_samples = {"train": 0, "val": 0, "test": 0}

    # Discover parquet files (only reads metadata, not data)
    for ctx_seconds in context_lengths_seconds:
        ctx_dir = format_context_dir(ctx_seconds)

        for task in tasks:
            for split in ["train", "val", "test"]:
                parquet_path = base_dir / ctx_dir / task / split / "data.parquet"

                if parquet_path.exists():
                    split_files[split].append(str(parquet_path))
                    # Quick row count without loading data
                    num_rows = pq.read_metadata(parquet_path).num_rows
                    total_samples[split] += num_rows
                    print(f"Found {parquet_path.relative_to(base_dir)}: {num_rows} samples")

    # Create lazy datasets
    columns_to_load = _get_columns_to_load(use_cot, extra_columns)
    datasets_list = []

    for split in ["train", "val", "test"]:
        files = split_files[split]

        if files:
            # Create lazy dataset - NO data is loaded yet
            lazy_ds = LazyParquetDataset(files, columns_to_load)
            datasets_list.append(lazy_ds)
            print(f"Total {split}: {len(lazy_ds)} samples (lazy loading)")
        else:
            # Create empty lazy dataset
            print(f"Warning: No data found for {split} split, creating empty dataset")
            datasets_list.append(LazyParquetDataset([], columns_to_load))

    return tuple(datasets_list)


def get_task_distribution(dataset) -> Dict[str, int]:
    """
    Get task type distribution for a dataset.

    Args:
        dataset: LazyParquetDataset or HuggingFace Dataset object

    Returns:
        Dictionary mapping task types to counts
    """
    if len(dataset) == 0:
        return {}

    # For lazy datasets, we need to iterate (but only load task_type column)
    if isinstance(dataset, LazyParquetDataset):
        task_counts: Dict[str, int] = {}
        for i in range(len(dataset)):
            task = dataset[i].get("task_type", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
        return task_counts

    task_types = dataset["task_type"]
    return dict(pd.Series(task_types).value_counts())


def get_answer_type_distribution(dataset) -> Dict[str, int]:
    """
    Get answer type distribution for a dataset.

    Args:
        dataset: LazyParquetDataset or HuggingFace Dataset object

    Returns:
        Dictionary mapping answer types to counts
    """
    if len(dataset) == 0:
        return {}

    # For lazy datasets, we need to iterate
    if isinstance(dataset, LazyParquetDataset):
        answer_counts: Dict[str, int] = {}
        for i in range(len(dataset)):
            answer_type = dataset[i].get("answer_type", "unknown")
            answer_counts[answer_type] = answer_counts.get(answer_type, 0) + 1
        return answer_counts

    answer_types = dataset["answer_type"]
    return dict(pd.Series(answer_types).value_counts())


def print_dataset_info(dataset, name: str) -> None:
    """
    Print information about a dataset split.

    Args:
        dataset: The dataset split
        name: Name of the split (e.g., "Train")
    """
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) == 0:
        return

    # Task distribution
    task_dist = get_task_distribution(dataset)
    if task_dist:
        print(f"  Task distribution:")
        for task, count in sorted(task_dist.items()):
            print(f"    {task}: {count} ({count/len(dataset)*100:.1f}%)")

    # Answer type distribution
    answer_dist = get_answer_type_distribution(dataset)
    if answer_dist:
        print(f"  Answer type distribution:")
        for answer_type, count in sorted(answer_dist.items()):
            print(f"    {answer_type}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack QA Loader Demo")
    print("=" * 60)

    # Demo configuration
    tasks = ["existence", "localization"]
    context_lengths = [100]

    print(f"\nConfiguration:")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths (seconds): {context_lengths}")

    # Load the dataset splits
    print("\nLoading dataset splits...")
    try:
        train_ds, val_ds, test_ds = load_ts_haystack_splits(
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )

        # Print dataset information
        print_dataset_info(train_ds, "Train")
        print_dataset_info(val_ds, "Validation")
        print_dataset_info(test_ds, "Test")

        # Show sample data
        if len(train_ds) > 0:
            print("\n" + "=" * 50)
            print("Sample data from training set:")
            sample = train_ds[0]
            for key, value in sample.items():
                if key in ["x_axis", "y_axis", "z_axis"]:
                    if isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: {value[:5]}... (length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                elif key in ["needles", "difficulty_config"]:
                    if isinstance(value, str) and len(value) > 80:
                        print(f"  {key}: {value[:80]}...")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure TS-Haystack datasets have been generated first.")

    print("\n" + "=" * 60)
    print("Loader demo complete!")
    print("=" * 60)
