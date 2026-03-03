#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Centralized dataset generator for TS-Haystack benchmark.

Generates task samples across all enabled tasks and context lengths
using YAML-based configuration.

Usage:
    # Generate using config file
    python scripts/generate_dataset.py --config configs/default_generation_config.yaml

    # Print default config
    python scripts/generate_dataset.py --print-default-config > my_config.yaml

    # Dry run to validate config
    python scripts/generate_dataset.py --config configs/default_generation_config.yaml --dry-run

    # Override specific tasks/lengths
    python scripts/generate_dataset.py \
        --config configs/default_generation_config.yaml \
        --tasks existence localization \
        --context-lengths 100 1000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ts_haystack.generation.config import (
    DEFAULT_CONFIG_PATH,
    GenerationConfig,
    print_default_config,
)
from ts_haystack.core import (
    BackgroundSampler,
    BoutIndexer,
    NeedleSampler,
    PromptTemplateBank,
    SeedManager,
    StyleTransfer,
    TimelineBuilder,
    TransitionMatrix,
)
from ts_haystack.tasks import (
    TASK_REGISTRY,
    get_task_generator,
    list_available_tasks,
)
from ts_haystack.utils import format_context_dir


def setup_components(
    config: GenerationConfig,
) -> Tuple[
    Dict,  # timelines
    Any,  # bout_index
    Any,  # transition_matrix
    SeedManager,
    BackgroundSampler,
    NeedleSampler,
    StyleTransfer,
    PromptTemplateBank,
]:
    """
    Load Phase 1 artifacts and initialize Phase 2 components.

    Args:
        config: Generation configuration

    Returns:
        Tuple of (timelines, bout_index, transition_matrix,
                  seed_manager, background_sampler, needle_sampler,
                  style_transfer, template_bank)
    """
    print("Loading Phase 1 artifacts...")
    timelines = TimelineBuilder.load_all_timelines()
    bout_index = BoutIndexer.load_index()
    transition_matrix = TransitionMatrix.load()
    print(f"  Loaded {len(timelines)} participant timelines")
    print(
        f"  Bout index: {bout_index.total_bouts} bouts across "
        f"{len(bout_index.activities)} activities"
    )

    print("\nInitializing Phase 2 components...")
    seed_manager = SeedManager(master_seed=config.seed)
    background_sampler = BackgroundSampler(timelines, bout_index, config.source_hz)
    needle_sampler = NeedleSampler(bout_index, transition_matrix, config.source_hz)
    style_transfer = StyleTransfer(
        transfer_mode=config.style_transfer.transfer_mode,
        blend_mode=config.style_transfer.blend_mode,
        blend_window_samples=config.style_transfer.blend_window_samples,
    )
    template_bank = PromptTemplateBank()

    return (
        timelines,
        bout_index,
        transition_matrix,
        seed_manager,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
    )


def generate_for_task(
    task_name: str,
    config: GenerationConfig,
    context_length_samples: int,
    background_sampler: BackgroundSampler,
    needle_sampler: NeedleSampler,
    style_transfer: StyleTransfer,
    template_bank: PromptTemplateBank,
    seed_manager: SeedManager,
) -> Dict[str, Any]:
    """
    Generate dataset for a single task at a single context length.

    Args:
        task_name: Name of the task to generate
        config: Generation configuration
        context_length_samples: Context length in samples
        background_sampler: Background sampler
        needle_sampler: Needle sampler
        style_transfer: Style transfer component
        template_bank: Prompt template bank
        seed_manager: Seed manager

    Returns:
        Dictionary with generation statistics
    """
    context_s = context_length_samples / config.source_hz
    print(f"\n{'='*60}")
    print(f"Task: {task_name} | Context: {context_s}s ({context_length_samples} samples)")
    print(f"{'='*60}")

    # Get task generator class
    TaskClass = get_task_generator(task_name)

    # Create generator instance
    generator = TaskClass(
        background_sampler=background_sampler,
        needle_sampler=needle_sampler,
        style_transfer=style_transfer,
        template_bank=template_bank,
        seed_manager=seed_manager,
        source_hz=config.source_hz,
    )

    # Get difficulty config
    difficulty = config.get_difficulty_config(task_name, context_length_samples)

    stats = {
        "task": task_name,
        "context_length_samples": context_length_samples,
        "context_length_seconds": context_s,
        "splits": {},
    }

    # Generate for each split
    for split, n_samples in config.samples_per_split.items():
        if n_samples <= 0:
            continue

        print(f"\n  [{split}] Generating {n_samples} samples...")

        # Check if output exists
        output_path = (
            config.output_dir / format_context_dir(context_s) / task_name / split / "data.parquet"
        )

        if output_path.exists() and not config.overwrite:
            print(f"    Skipping - already exists: {output_path}")
            stats["splits"][split] = {"skipped": True, "path": str(output_path)}
            continue

        # Generate samples
        samples = generator.generate_dataset(
            n_samples=n_samples,
            difficulty=difficulty,
            split=split,
            n_jobs=config.n_jobs,
            verbose=True,
        )

        # Save to parquet
        saved_path = generator.save_dataset(
            samples=samples,
            split=split,
            context_length=context_length_samples,
            output_dir=config.output_dir,
        )

        valid_count = sum(1 for s in samples if s.is_valid)
        print(f"    Saved {valid_count} samples to: {saved_path}")

        stats["splits"][split] = {
            "generated": len(samples),
            "valid": valid_count,
            "path": str(saved_path),
        }

    # Save task-specific metadata
    task_metadata_path = config.output_dir / format_context_dir(context_s) / task_name / "metadata.json"
    task_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    task_metadata = {
        "task_name": task_name,
        "answer_type": generator.answer_type,
        "context_length_samples": context_length_samples,
        "context_length_seconds": context_s,
        "difficulty_config": difficulty.to_dict(),
        "generation_stats": stats,
        "generation_timestamp": datetime.now().isoformat(),
    }
    with open(task_metadata_path, "w") as f:
        json.dump(task_metadata, f, indent=2)

    return stats


def save_global_metadata(
    config: GenerationConfig,
    all_stats: List[Dict[str, Any]],
) -> Path:
    """
    Save global generation metadata.

    Args:
        config: Generation configuration
        all_stats: Statistics from all task generations

    Returns:
        Path to metadata file
    """
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "tasks_generated": list(set(s["task"] for s in all_stats)),
        "context_lengths_samples": config.get_context_lengths_samples(),
        "context_lengths_seconds": config.context_lengths_seconds,
        "generation_stats": all_stats,
    }

    output_path = config.output_dir / "metadata.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate TS-Haystack benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate using config file
    python scripts/generate_dataset.py --config configs/default_generation_config.yaml

    # Print default config to create starting point
    python scripts/generate_dataset.py --print-default-config > my_config.yaml

    # Dry run to validate config
    python scripts/generate_dataset.py --config configs/default_generation_config.yaml --dry-run

    # Override specific tasks/lengths
    python scripts/generate_dataset.py \\
        --config configs/default_generation_config.yaml \\
        --tasks existence localization \\
        --context-lengths 100 1000
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print default configuration YAML and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show plan without generating",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Override: Only generate these tasks (space-separated)",
    )
    parser.add_argument(
        "--context-lengths",
        type=float,
        nargs="+",
        help="Override: Context lengths in seconds (space-separated, supports floats like 2.56)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override: Force overwrite existing files",
    )

    args = parser.parse_args()

    # Handle --print-default-config
    if args.print_default_config:
        print(print_default_config())
        sys.exit(0)

    # Determine config file
    if args.config is None:
        # Use default config
        if DEFAULT_CONFIG_PATH.exists():
            config_path = DEFAULT_CONFIG_PATH
            print(f"Using default config: {config_path}")
        else:
            parser.error(
                "--config is required (or use --print-default-config to create one)"
            )
    else:
        config_path = args.config
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")

    # Load configuration
    print("=" * 60)
    print("TS-Haystack Dataset Generator")
    print("=" * 60)
    print(f"\nLoading config: {config_path}")
    config = GenerationConfig.from_yaml(config_path)

    # Apply CLI overrides
    if args.tasks:
        # Filter tasks to only specified ones
        available_tasks = list_available_tasks()
        for task_name in args.tasks:
            if task_name not in available_tasks:
                parser.error(
                    f"Unknown task: {task_name}. "
                    f"Available: {', '.join(available_tasks)}"
                )
        for task_name in list(config.tasks.keys()):
            config.tasks[task_name].enabled = task_name in args.tasks

    if args.context_lengths:
        config.context_lengths_seconds = args.context_lengths

    if args.overwrite:
        config.overwrite = True

    # Print configuration summary
    enabled_tasks = config.get_enabled_tasks()
    context_lengths_samples = config.get_context_lengths_samples()

    print(f"\nConfiguration Summary:")
    print(f"  Seed: {config.seed}")
    print(f"  Workers: {config.n_jobs}")
    print(f"  Output: {config.output_dir}")
    print(f"  Overwrite: {config.overwrite}")
    print(f"  Context lengths: {config.context_lengths_seconds} seconds")
    print(f"  Samples per split: {config.samples_per_split}")
    print(f"  Enabled tasks ({len(enabled_tasks)}): {', '.join(enabled_tasks)}")
    print(
        f"  Style transfer: {config.style_transfer.transfer_mode}, "
        f"{config.style_transfer.blend_mode}"
    )

    # Calculate total samples
    total_samples = (
        len(enabled_tasks)
        * len(config.context_lengths_seconds)
        * sum(config.samples_per_split.values())
    )
    print(f"\n  Total samples to generate: {total_samples:,}")

    # Dry run - just show plan
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - No generation performed")
        print("=" * 60)
        print("\nGeneration plan:")
        for ctx_s in config.context_lengths_seconds:
            for task_name in enabled_tasks:
                for split, n in config.samples_per_split.items():
                    output_path = (
                        config.output_dir
                        / format_context_dir(ctx_s)
                        / task_name
                        / split
                        / "data.parquet"
                    )
                    exists = "EXISTS" if output_path.exists() else "NEW"
                    will_skip = (
                        " [SKIP]" if exists == "EXISTS" and not config.overwrite else ""
                    )
                    print(f"  {ctx_s}s/{task_name}/{split}: {n} samples [{exists}]{will_skip}")
        sys.exit(0)

    # Load components
    (
        timelines,
        bout_index,
        transition_matrix,
        seed_manager,
        background_sampler,
        needle_sampler,
        style_transfer,
        template_bank,
    ) = setup_components(config)

    # Generate datasets
    all_stats = []

    for context_s in config.context_lengths_seconds:
        context_length_samples = int(context_s * config.source_hz)

        for task_name in enabled_tasks:
            try:
                stats = generate_for_task(
                    task_name=task_name,
                    config=config,
                    context_length_samples=context_length_samples,
                    background_sampler=background_sampler,
                    needle_sampler=needle_sampler,
                    style_transfer=style_transfer,
                    template_bank=template_bank,
                    seed_manager=seed_manager,
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"\n  ERROR generating {task_name}: {e}")
                all_stats.append(
                    {
                        "task": task_name,
                        "context_length_samples": context_length_samples,
                        "context_length_seconds": context_s,
                        "error": str(e),
                    }
                )

    # Save global metadata
    metadata_path = save_global_metadata(config, all_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nMetadata saved: {metadata_path}")
    print(f"Output directory: {config.output_dir}")

    # Summarize by task
    print("\nSummary:")
    for stats in all_stats:
        ctx_s = stats.get("context_length_seconds", "?")
        task = stats["task"]
        if "error" in stats:
            print(f"  {ctx_s}s/{task}: ERROR - {stats['error']}")
        else:
            for split, split_stats in stats.get("splits", {}).items():
                if split_stats.get("skipped"):
                    print(f"  {ctx_s}s/{task}/{split}: SKIPPED (exists)")
                else:
                    print(f"  {ctx_s}s/{task}/{split}: {split_stats.get('valid', 0)} samples")


if __name__ == "__main__":
    main()
