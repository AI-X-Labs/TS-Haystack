"""Parquet loader for UK-DALE-Haystack runtime datasets.

Mirrors ``ltaf_haystack/qa_loader.py``. UK-DALE shards live under
``data/uk_dale/uk_dale_haystack/tasks/{ctx}s/{task}/{split}/data.parquet``
with integer-second context directories (e.g. ``900s``, ``3600s``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset as HFDataset

from src.datasets.constants import RAW_DATA


ALL_CONTEXT_LENGTHS = "all"
UK_DALE_HAYSTACK_TASKS_DIR = Path(RAW_DATA) / "uk_dale" / "uk_dale_haystack" / "tasks"


def _get_data_dir(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return UK_DALE_HAYSTACK_TASKS_DIR


def _context_dir_name(seconds: float) -> str:
    # UK-DALE uses integer-second directories (900s, 3600s, ...).
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return str(seconds).replace(".", "_") + "s"


def _resolve_context_dir(root: Path, context: float | str) -> Path:
    if isinstance(context, str):
        c = context.strip()
        if not c:
            return root
        if c.endswith("s"):
            return root / c
        try:
            return root / _context_dir_name(float(c))
        except ValueError:
            return root / c
    return root / _context_dir_name(float(context))


def _split_dir_candidates(split: str) -> list[str]:
    if split == "val":
        return ["validation", "val"]
    return [split]


def _empty_split() -> HFDataset:
    return HFDataset.from_list([])


def _load_split(paths: Iterable[Path]) -> HFDataset:
    import pyarrow.parquet as _pq

    existing: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            meta = _pq.read_metadata(p)
        except Exception:  # noqa: BLE001
            continue
        if int(meta.num_rows) <= 0:
            continue
        existing.append(str(p))
    if not existing:
        return _empty_split()
    ds = HFDataset.from_parquet(existing, keep_in_memory=False, on_bad_files="error")
    # Drop generator-rejected rows. Invalid samples carry placeholder
    # background_house_id=-1 / start_ns=-1 from base_task._invalid(); calling
    # reconstruct_sample_signal on them raises KeyError at runtime.
    if "is_valid" in ds.column_names:
        ds = ds.filter(lambda row: bool(row.get("is_valid", True)))
    return ds


def get_available_tasks(base_dir: Path | None = None) -> list[str]:
    root = _get_data_dir(base_dir=base_dir)
    if not root.exists():
        return []

    tasks: set[str] = set()
    for context_dir in root.iterdir():
        if not context_dir.is_dir() or not context_dir.name.endswith("s"):
            continue
        for task_dir in context_dir.iterdir():
            if task_dir.is_dir():
                tasks.add(task_dir.name)
    return sorted(tasks)


def get_available_context_lengths(base_dir: Path | None = None) -> list[float]:
    root = _get_data_dir(base_dir=base_dir)
    if not root.exists():
        return []

    out: list[float] = []
    for context_dir in root.iterdir():
        if not context_dir.is_dir() or not context_dir.name.endswith("s"):
            continue
        raw = context_dir.name[:-1].replace("_", ".")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return sorted(out)


def load_uk_dale_haystack_splits(
    tasks: list[str] | None = None,
    context_lengths_seconds: list[float | str] | None = None,
    base_dir: Path | None = None,
) -> tuple[HFDataset, HFDataset, HFDataset]:
    root = _get_data_dir(base_dir=base_dir)

    if tasks is None or (len(tasks) == 1 and tasks[0] == "all") or ("all" in tasks):
        tasks = get_available_tasks(root)
    if context_lengths_seconds is None or (
        len(context_lengths_seconds) == 1 and context_lengths_seconds[0] == "all"
    ):
        context_lengths_seconds = get_available_context_lengths(root)

    if not root.exists() or not tasks or not context_lengths_seconds:
        return _empty_split(), _empty_split(), _empty_split()

    selected_tasks = sorted({str(t) for t in tasks if str(t).strip()})

    split_paths = {"train": [], "val": [], "test": []}
    for context in context_lengths_seconds:
        context_dir = _resolve_context_dir(root, context)
        for task_name in selected_tasks:
            for split in split_paths:
                for split_dir in _split_dir_candidates(split):
                    split_paths[split].append(context_dir / task_name / split_dir / "data.parquet")

    return (
        _load_split(split_paths["train"]),
        _load_split(split_paths["val"]),
        _load_split(split_paths["test"]),
    )
