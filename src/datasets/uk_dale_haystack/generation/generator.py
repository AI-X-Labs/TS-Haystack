"""
Generation orchestrator for UK-DALE-Haystack.

Iterates (task, context_length, split) and writes lightweight parquet shards
under {output_dir}/{ctx}s/{task}/{split}/data.parquet.

Per-sample mains_w is NOT embedded (signals are reconstructed at evaluation
time by the loader from background_start_ns + needle metadata). The shard
carries (question, answer, task, ctx, split) plus reconstruction metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.core.needle_sampler import NeedleSampler
from src.datasets.uk_dale_haystack.core.prompt_templates import PromptTemplateBank
from src.datasets.uk_dale_haystack.generation.config import GenerationConfig
from src.datasets.uk_dale_haystack.tasks import TASK_REGISTRY


def _seed_for(master: int, task: str, ctx_s: int, split: str, idx: int) -> int:
    h = hashlib.md5(f"{master}-{task}-{ctx_s}-{split}-{idx}".encode()).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFF


def _sample_to_row(sample, ctx_s: int, split: str) -> dict[str, Any]:
    return {
        "task_type": sample.task_type,
        "context_length_s": int(ctx_s),
        "split": split,
        "question": sample.question,
        "answer": sample.answer,
        "answer_type": sample.answer_type,
        "background_house_id": int(sample.background_house_id),
        "background_start_ns": int(sample.background_start_ns),
        "background_end_ns": int(sample.background_end_ns),
        "dt_s": float(sample.dt_s),
        "n_needles": len(sample.needles),
        "needles_json": json.dumps([asdict(n) for n in sample.needles]),
        "other_bouts_json": json.dumps([asdict(b) for b in sample.other_bouts_metadata]),
        "difficulty_config_json": json.dumps(sample.difficulty_config),
        "is_valid": bool(sample.is_valid),
        "validation_notes": sample.validation_notes or "",
    }


def generate_shard(
    task_name: str,
    ctx_s: int,
    split: str,
    n_samples: int,
    background_sampler: BackgroundSampler,
    needle_sampler: NeedleSampler,
    template_bank: PromptTemplateBank,
    *,
    master_seed: int,
    allow_cross_house: bool,
    dt_s: float,
    max_retries_per_sample: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cls = TASK_REGISTRY[task_name]
    gen = cls(background_sampler, needle_sampler, template_bank,
              dt_s=dt_s, allow_cross_house=allow_cross_house)
    rows: list[dict[str, Any]] = []
    counters = {"valid": 0, "invalid": 0, "retried": 0}
    for i in range(n_samples):
        for attempt in range(max_retries_per_sample):
            seed = _seed_for(master_seed, task_name, ctx_s, split, i * 100 + attempt)
            rng = np.random.default_rng(seed)
            sample = gen.generate_sample(ctx_s, rng)
            if sample.is_valid:
                rows.append(_sample_to_row(sample, ctx_s, split))
                counters["valid"] += 1
                if attempt > 0:
                    counters["retried"] += 1
                break
        else:
            rows.append(_sample_to_row(sample, ctx_s, split))
            counters["invalid"] += 1
    return rows, counters


def run(
    cfg: GenerationConfig,
    *,
    bout_index_path: Path,
    only_tasks: list[str] | None = None,
    only_ctx: list[int] | None = None,
    only_splits: list[str] | None = None,
    max_samples_override: int | None = None,
    overwrite: bool = False,
) -> None:
    bouts = pl.read_parquet(bout_index_path)
    splits = list(cfg.samples.keys())
    if only_splits:
        splits = [s for s in splits if s in only_splits]
    samplers_per_split: dict[str, tuple[BackgroundSampler, NeedleSampler]] = {}
    template_bank = PromptTemplateBank()
    for split in splits:
        samplers_per_split[split] = (
            BackgroundSampler(bouts, split, dt_s=cfg.global_.dt_s),
            NeedleSampler(bouts, split, dt_s=cfg.global_.dt_s),
        )

    out_root = Path(cfg.global_.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    tasks = only_tasks or cfg.enabled_tasks()
    ctxs = only_ctx or cfg.context_lengths_seconds

    for ctx_s in ctxs:
        ctx_dir = out_root / f"{ctx_s}s"
        for task_name in tasks:
            if task_name not in TASK_REGISTRY:
                print(f"[skip] unknown task: {task_name}")
                continue
            for split in splits:
                shard_dir = ctx_dir / task_name / split
                shard_path = shard_dir / "data.parquet"
                if shard_path.exists() and not overwrite:
                    print(f"[skip-existing] {shard_path}")
                    continue
                n_samples = (max_samples_override
                             if max_samples_override is not None
                             else int(cfg.samples[split]))
                t0 = time.time()
                bg_sampler, nd_sampler = samplers_per_split[split]
                rows, counters = generate_shard(
                    task_name, ctx_s, split, n_samples,
                    bg_sampler, nd_sampler, template_bank,
                    master_seed=cfg.global_.seed,
                    allow_cross_house=cfg.allow_cross_house,
                    dt_s=cfg.global_.dt_s,
                )
                shard_dir.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(rows).write_parquet(shard_path)
                dt = time.time() - t0
                print(
                    f"  ctx={ctx_s:>5}s  task={task_name:<22} split={split:<10} "
                    f"valid={counters['valid']:>4}/{n_samples}  "
                    f"invalid={counters['invalid']:>4}  retried={counters['retried']:>4}  "
                    f"({dt:.1f}s)  -> {shard_path}"
                )
                summary_rows.append({
                    "ctx_s": int(ctx_s), "task": task_name, "split": split,
                    **counters, "n_samples_requested": n_samples,
                    "duration_s": float(dt),
                })

    summary_path = out_root / "generation_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"\n  -> {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Path to YAML config (default: defaults.yaml).")
    ap.add_argument("--bout-index",
                    default="data/uk_dale/uk_dale_haystack/bout_index.parquet")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--context-lengths", nargs="+", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--max-samples-per-split", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = GenerationConfig.from_yaml(args.config) if args.config else GenerationConfig.from_yaml()
    run(
        cfg,
        bout_index_path=Path(args.bout_index),
        only_tasks=args.tasks,
        only_ctx=args.context_lengths,
        only_splits=args.splits,
        max_samples_override=args.max_samples_per_split,
        overwrite=args.overwrite,
    )
