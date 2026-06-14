#!/usr/bin/env python3
"""
Phase 9.2: per-(task, context) verification plots.

Renders sample_NN.png + sample_NN.json under
data/uk_dale/uk_dale_haystack/verification/{ctx}s/{task}/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.datasets.uk_dale_haystack.plot_generator import plot_sample_row

DEFAULT_TASKS_DIR = Path("data/uk_dale/uk_dale_haystack/tasks")
DEFAULT_OUT_DIR = Path("data/uk_dale/uk_dale_haystack/verification")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Subset of task names. Default: all under tasks-dir.")
    ap.add_argument("--context-lengths", nargs="+", type=int, default=None,
                    help="Subset of context lengths in seconds.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-samples", type=int, default=3)
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    out_dir = Path(args.out_dir)

    for ctx_dir in sorted(tasks_dir.glob("*s")):
        try:
            ctx_s = int(ctx_dir.name[:-1])
        except ValueError:
            continue
        if args.context_lengths and ctx_s not in args.context_lengths:
            continue
        for task_dir in sorted(ctx_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if args.tasks and task_dir.name not in args.tasks:
                continue
            shard = task_dir / args.split / "data.parquet"
            if not shard.exists():
                continue
            df = pl.read_parquet(shard).filter(pl.col("is_valid"))
            if df.height == 0:
                print(f"[{task_dir.name} ctx={ctx_s}] no valid rows")
                continue
            n = min(args.n_samples, df.height)
            target_dir = out_dir / f"{ctx_s}s" / task_dir.name
            for i in range(n):
                row = df.row(i, named=True)
                png = target_dir / f"sample_{i:02d}.png"
                json_path = target_dir / f"sample_{i:02d}.json"
                plot_sample_row(row, png)
                json_path.write_text(json.dumps(
                    {k: row[k] for k in (
                        "task_type", "context_length_s", "split", "question",
                        "answer", "answer_type", "background_house_id",
                        "background_start_ns", "background_end_ns", "n_needles",
                        "difficulty_config_json", "needles_json",
                    )}, indent=2, default=str,
                ))
                print(f"  [{task_dir.name} ctx={ctx_s}] -> {png}")


if __name__ == "__main__":
    main()
