#!/usr/bin/env python3
"""Print a task x context accuracy grid from a GPT benchmark run.

Reads all ``<ctx>/<task>/test/trajectories.jsonl`` files under a run
directory and prints:

  * Accuracy per (task, ctx) with row/col marginals and overall.
  * Sample counts per (task, ctx) with row/col totals.

Handles heterogeneous shards (e.g. existence gated to short contexts):
empty cells render as ``--``.

Usage:
    python scripts/eval/print_gpt_benchmark_grid.py \\
        results/gpt_sleep_prepass_benchmark/<run>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

CTX_LABELS = {
    100.0: "100s",
    900.0: "15min",
    3600.0: "1h",
    7200.0: "2h",
    -1.0: "full",
}


def ctx_label(ctx: float) -> str:
    return CTX_LABELS.get(ctx, f"{ctx:g}s")


def iter_records(run_dir: Path) -> Iterable[dict]:
    for path in sorted(run_dir.glob("*/*/test/trajectories.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def print_grid(
    title: str,
    row_keys: list[str],
    col_keys: list[str],
    cell: dict[tuple[str, str], str],
    row_total: dict[str, str],
    col_total: dict[str, str],
    overall: str,
    corner: str = "overall",
) -> None:
    col_w = max(10, max(len(c) for c in col_keys) + 2)
    row_w = max(len(k) for k in row_keys + [corner, "task"]) + 4

    header = f"{'task':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in col_keys) + f"{corner:>{col_w}}"
    bar = "-" * len(header)
    print()
    print(title)
    print(bar)
    print(header)
    print(bar)
    for r in row_keys:
        line = f"{r:<{row_w}}"
        for c in col_keys:
            line += f"{cell.get((r, c), '--'):>{col_w}}"
        line += f"{row_total.get(r, '--'):>{col_w}}"
        print(line)
    print(bar)
    foot = f"{corner:<{row_w}}"
    for c in col_keys:
        foot += f"{col_total.get(c, '--'):>{col_w}}"
    foot += f"{overall:>{col_w}}"
    print(foot)
    print(bar)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="Path to the GPT benchmark run directory")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # correct, total
    ctx_seen: list[float] = []
    task_seen: list[str] = []

    for rec in iter_records(run_dir):
        task = rec.get("task_type") or "unknown"
        ctx = float(rec.get("context_length_s", 0))
        key = (task, ctx_label(ctx))
        counts[key][1] += 1
        if rec.get("correct"):
            counts[key][0] += 1
        if ctx not in ctx_seen:
            ctx_seen.append(ctx)
        if task not in task_seen:
            task_seen.append(task)

    if not counts:
        raise SystemExit(f"No trajectories.jsonl records found under {run_dir}")

    ctx_seen.sort(key=lambda x: (x < 0, x))  # positive ctxs ascending, then -1 (full) last
    col_keys = [ctx_label(c) for c in ctx_seen]
    row_keys = sorted(task_seen)

    acc_cell: dict[tuple[str, str], str] = {}
    cnt_cell: dict[tuple[str, str], str] = {}
    for (t, c), (corr, tot) in counts.items():
        if tot:
            acc_cell[(t, c)] = f"{corr / tot:.3f}"
            cnt_cell[(t, c)] = str(tot)

    row_acc, row_cnt = {}, {}
    for t in row_keys:
        corr = sum(v[0] for (tt, _), v in counts.items() if tt == t)
        tot = sum(v[1] for (tt, _), v in counts.items() if tt == t)
        if tot:
            row_acc[t] = f"{corr / tot:.3f}"
            row_cnt[t] = str(tot)

    col_acc, col_cnt = {}, {}
    for c in col_keys:
        corr = sum(v[0] for (_, cc), v in counts.items() if cc == c)
        tot = sum(v[1] for (_, cc), v in counts.items() if cc == c)
        if tot:
            col_acc[c] = f"{corr / tot:.3f}"
            col_cnt[c] = str(tot)

    tot_corr = sum(v[0] for v in counts.values())
    tot_n = sum(v[1] for v in counts.values())
    overall_acc = f"{tot_corr / tot_n:.3f}" if tot_n else "--"

    print("=" * 72)
    print(f"GPT benchmark grid — {run_dir.name}")
    print("=" * 72)
    print_grid(
        "Accuracy — rows: task, cols: context length",
        row_keys, col_keys, acc_cell, row_acc, col_acc, overall_acc,
        corner="overall",
    )
    print_grid(
        "Sample counts per cell",
        row_keys, col_keys, cnt_cell, row_cnt, col_cnt, str(tot_n),
        corner="total",
    )


if __name__ == "__main__":
    main()
