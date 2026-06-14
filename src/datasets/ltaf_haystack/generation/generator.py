#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Natural-only generation orchestrator for LTAF-Haystack.

For each (task, context_length_s, split):

  1. Check the task's :meth:`supports_context_length` gate.
  2. Build / load the :class:`LTAFWindowIndex` for that ctx on the split.
  3. Instantiate :class:`RecordingSampler` over the split's records.
  4. Generate ``n_samples`` via the task's :meth:`generate_dataset`.
  5. Save one parquet shard per (task, ctx, split) via :meth:`save_dataset`.

Signals are real slices; nothing is synthesised or spliced.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.datasets.ltaf_haystack.core.ltaf_prompt_templates import (
    LTAFPromptTemplateBank,
)
from src.datasets.ltaf_haystack.core.participant_splitter import (
    load_split_manifest,
)
from src.datasets.ltaf_haystack.core.recording_sampler import RecordingSampler
from src.datasets.ltaf_haystack.core.seed_manager import LTAFSeedManager
from src.datasets.ltaf_haystack.core.window_index import LTAFWindowIndex
from src.datasets.ltaf_haystack.generation.config import (
    DEFAULT_CONFIG_PATH,
    LTAFGenerationConfig,
)
from src.datasets.ltaf_haystack.tasks import (
    get_task_generator,
    get_tasks_for_label_class,
    list_available_tasks,
)


def _format_context_dir(seconds: float) -> str:
    """Match ``qa_loader._context_dir_name`` for cross-tool compatibility."""
    return str(float(seconds)).replace(".", "_") + "s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LTAF-Haystack task datasets (natural-only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to generation YAML config.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Task name override.")
    parser.add_argument(
        "--context-lengths",
        type=float,
        nargs="*",
        default=None,
        help="Context length override in seconds.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Cap each split at this size.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root directory; defaults to config.output_dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = LTAFGenerationConfig.from_yaml(args.config)
    if args.output_root is not None:
        cfg.output_dir = args.output_root

    enabled = cfg.get_enabled_tasks() or get_tasks_for_label_class(cfg.label_class)
    if args.tasks:
        available = set(list_available_tasks())
        requested = [t for t in args.tasks if t]
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(f"Unknown tasks: {unknown}. Available: {sorted(available)}")
        enabled = [t for t in requested if t in available]

    context_lengths_s = [float(x) for x in (args.context_lengths or cfg.context_lengths_seconds)]
    samples_per_split = dict(cfg.samples_per_split)
    if args.max_samples_per_split is not None:
        cap = max(0, int(args.max_samples_per_split))
        samples_per_split = {k: min(v, cap) for k, v in samples_per_split.items()}

    # -----------------------------------------------------------------
    # Load core artifacts
    # -----------------------------------------------------------------
    manifest = load_split_manifest()
    template_bank = LTAFPromptTemplateBank()
    seed_manager = LTAFSeedManager(master_seed=cfg.seed)

    print("LTAF-Haystack natural-only generator")
    print(f"  config       = {args.config}")
    print(f"  label_class  = {cfg.label_class}")
    print(f"  tasks        = {enabled}")
    print(f"  contexts (s) = {context_lengths_s}")
    print(f"  splits       = {samples_per_split}")
    print(f"  output       = {cfg.output_dir}")

    if args.dry_run:
        print("[DRY RUN] No generation performed.")
        return

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict] = []

    # Build one window index per ctx over the union of all splits' records,
    # then let each split's RecordingSampler filter to its own records.
    # This mirrors sleep_psg_haystack and avoids a split-specific cache
    # shadowing the other splits' records on the second / third pass.
    all_records = sorted({
        rid for split_ids in manifest.values() if isinstance(split_ids, list)
        for rid in split_ids
    })

    # -----------------------------------------------------------------
    # Generation loop
    # -----------------------------------------------------------------
    for ctx_s in context_lengths_s:
        ctx_dir = _format_context_dir(ctx_s)

        try:
            window_index = LTAFWindowIndex.get_or_build(
                label_class=cfg.label_class,
                context_length_s=float(ctx_s),
                record_ids=all_records,
            )
        except Exception as exc:
            print(f"  [skip] window_index build failed for ctx={ctx_s}s: {exc}")
            continue

        for split, n_samples in samples_per_split.items():
            split_records = manifest.get(split, [])
            if not split_records:
                continue

            try:
                recording_sampler = RecordingSampler(
                    record_ids=list(split_records),
                    label_class=cfg.label_class,
                    window_index=window_index,
                )
            except ValueError as exc:
                print(f"  [skip] recording_sampler init failed for ctx={ctx_s}s, split={split}: {exc}")
                continue

            if not recording_sampler._indexed_pairs:
                print(f"  [skip] no indexed windows for ctx={ctx_s}s, split={split}")
                continue

            for task_name in enabled:
                TaskClass = get_task_generator(task_name)
                if not TaskClass.supports_context_length(cfg.label_class, ctx_s):
                    continue

                output_path = (
                    cfg.output_dir / ctx_dir / task_name / split / "data.parquet"
                )
                if output_path.exists() and not args.overwrite:
                    print(f"  [skip] {output_path} exists (use --overwrite to rebuild)")
                    continue

                generator = TaskClass(
                    recording_sampler=recording_sampler,
                    template_bank=template_bank,
                    seed_manager=seed_manager,
                    label_class=cfg.label_class,
                )

                samples = generator.generate_dataset(
                    n_samples=n_samples,
                    split=split,
                    n_jobs=max(1, cfg.n_jobs),
                    verbose=False,
                )
                saved_path = generator.save_dataset(
                    samples=samples,
                    split=split,
                    output_dir=cfg.output_dir,
                    context_dir=ctx_dir,
                )
                print(
                    f"  [{task_name:24}][{ctx_s:>6.0f}s][{split:10}] "
                    f"requested={n_samples}, generated={len(samples)} → {saved_path}"
                )
                all_stats.append(
                    {
                        "task": task_name,
                        "context_length_seconds": float(ctx_s),
                        "split": split,
                        "requested": n_samples,
                        "generated": len(samples),
                        "path": str(saved_path),
                    }
                )

    # -----------------------------------------------------------------
    # Global metadata
    # -----------------------------------------------------------------
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "seed": cfg.seed,
        "source_hz": cfg.source_hz,
        "label_class": cfg.label_class,
        "tasks": enabled,
        "context_lengths_seconds": context_lengths_s,
        "samples_per_split": samples_per_split,
        "output_root": str(cfg.output_dir),
        "per_shard_stats": all_stats,
    }
    meta_path = cfg.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Generation complete. Metadata written to {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
