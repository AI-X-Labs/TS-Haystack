#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
TS-Haystack Chain-of-Thought (CoT) Rationale Generator CLI.

This script generates LLM-based chain-of-thought rationales for TS-Haystack
benchmark samples. It reads the generated task parquet files and adds a
'rationale' column containing the reasoning.

Prerequisites:
    1. Task datasets must be generated first using generate_dataset.py
    2. OPENAI_API_KEY environment variable must be set

Usage:
    # Generate CoT for all tasks at 100s context length
    python scripts/generate_cot.py \
        --context-lengths 100 \
        --tasks all \
        --max-workers 4

    # Generate CoT for specific tasks
    python scripts/generate_cot.py \
        --context-lengths 100 \
        --tasks existence localization counting \
        --splits train val

    # Test with a few samples
    python scripts/generate_cot.py \
        --context-lengths 100 \
        --tasks existence \
        --splits test \
        --max-samples 10
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ts_haystack.capture24.loader import DATA_DIR
from ts_haystack.cot.cot_generator import (
    GenerationStats,
    TSHaystackCoTGenerator,
    save_generation_metadata,
)
from ts_haystack.cot.llm_client import (
    OpenAICoTClient,
    OpenAIConfig,
)
from ts_haystack.utils import format_context_dir


# Default paths
DEFAULT_INPUT_DIR = os.path.join(DATA_DIR, "capture24", "ts_haystack", "tasks")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "capture24", "ts_haystack", "cot")

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

# Available splits
ALL_SPLITS = ["train", "val", "test"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CoT rationales for TS-Haystack benchmark samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate CoT for all tasks at 100s
    python scripts/generate_cot.py --context-lengths 100 --tasks all

    # Test with small sample
    python scripts/generate_cot.py \\
        --context-lengths 100 --tasks existence --splits test --max-samples 5
        """,
    )

    # Data paths
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with task parquets (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CoT parquets (default: {DEFAULT_OUTPUT_DIR})",
    )

    # What to process
    parser.add_argument(
        "--context-lengths",
        type=float,
        nargs="+",
        default=[2.56, 10, 100, 900, 3600, 7200],
        help="Context lengths in seconds (default: all, supports floats like 2.56)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["all"],
        help=f"Tasks to process (default: all). Options: {ALL_TASKS}",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )

    # LLM configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model to use (default: gpt-4.1-mini-2025-04-14)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max API retries per sample (default: 5)",
    )

    # Processing options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers for API calls (default: 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per file (for testing, default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing output files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't include plot in LLM input",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Don't annotate needle regions in plots",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Don't validate generated answers",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save partial results every N samples (default: 100)",
    )

    # Debug options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without generating",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save JSON files with sample data, prompts, rationales, and plot images",
    )
    parser.add_argument(
        "--debug-output-dir",
        type=str,
        default=None,
        help="Directory for debug output (default: {output-dir}/debug)",
    )

    return parser.parse_args()


def find_input_files(
    input_dir: Path,
    context_lengths: List[float],
    tasks: List[str],
    splits: List[str],
) -> List[Dict]:
    """
    Find all input parquet files to process.

    Returns:
        List of dicts with keys: input_path, output_path, context_length, task, split
    """
    files = []

    for ctx_seconds in context_lengths:
        ctx_dir = format_context_dir(ctx_seconds)

        for task in tasks:
            for split in splits:
                input_path = input_dir / ctx_dir / task / split / "data.parquet"

                if input_path.exists():
                    files.append({
                        "input_path": input_path,
                        "context_length": ctx_seconds,
                        "task": task,
                        "split": split,
                    })
                else:
                    print(f"Warning: {input_path} not found, skipping")

    return files


def main():
    args = parse_args()

    print("=" * 70)
    print("TS-Haystack CoT Rationale Generator")
    print("=" * 70)

    # Resolve tasks
    if "all" in args.tasks:
        tasks = ALL_TASKS
    else:
        tasks = args.tasks
        for task in tasks:
            if task not in ALL_TASKS:
                print(f"Error: Unknown task '{task}'. Available: {ALL_TASKS}")
                sys.exit(1)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Context lengths: {args.context_lengths}s")
    print(f"  Tasks: {tasks}")
    print(f"  Splits: {args.splits}")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max workers: {args.max_workers}")
    print(f"  Include plot: {not args.no_plot}")
    print(f"  Annotate needles: {not args.no_annotations}")
    print(f"  Validate answers: {not args.no_validation}")
    if args.max_samples:
        print(f"  Max samples per file: {args.max_samples}")
    if args.debug:
        debug_dir = Path(args.debug_output_dir) if args.debug_output_dir else Path(args.output_dir) / "debug"
        print(f"  Debug mode: ENABLED")
        print(f"  Debug output dir: {debug_dir}")

    # Find input files
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = find_input_files(input_dir, args.context_lengths, tasks, args.splits)

    if not files:
        print("\nNo input files found!")
        print("Make sure task datasets have been generated first.")
        sys.exit(1)

    print(f"\nFound {len(files)} files to process:")
    for f in files:
        print(f"  {f['context_length']}s/{f['task']}/{f['split']}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without generating.")
        sys.exit(0)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Initialize LLM client
    config = OpenAIConfig(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    client = OpenAICoTClient(config)

    # Determine debug output directory
    debug_output_dir = None
    if args.debug:
        debug_output_dir = Path(args.debug_output_dir) if args.debug_output_dir else output_dir / "debug"

    # Initialize generator
    generator = TSHaystackCoTGenerator(
        llm_client=client,
        include_plot=not args.no_plot,
        annotate_needles=not args.no_annotations,
        max_workers=args.max_workers,
        validate_answers=not args.no_validation,
        save_interval=args.save_interval,
        debug_mode=args.debug,
        debug_output_dir=debug_output_dir,
    )

    # Process files
    all_stats: Dict[str, GenerationStats] = {}
    start_time = datetime.now()

    for i, file_info in enumerate(files):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(files)}] Processing: {file_info['context_length']}s/{file_info['task']}/{file_info['split']}")
        print(f"{'='*70}")

        input_path = file_info["input_path"]

        # Build output path with same structure
        output_path = (
            output_dir
            / format_context_dir(file_info['context_length'])
            / file_info["task"]
            / file_info["split"]
            / "data.parquet"
        )

        # Process
        stats = generator.process_dataset(
            input_path,
            output_path,
            resume=not args.no_resume,
            max_samples=args.max_samples,
        )

        all_stats[str(output_path)] = stats

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    save_generation_metadata(output_dir, config, all_stats)

    # Print summary
    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed}")
    print(f"Files processed: {len(files)}")

    total_success = sum(s.successful for s in all_stats.values())
    total_failed = sum(s.failed for s in all_stats.values())
    total_api_errors = sum(s.api_errors for s in all_stats.values())
    total_validation_errors = sum(s.validation_errors for s in all_stats.values())

    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Total API errors: {total_api_errors}")
    print(f"Total validation errors: {total_validation_errors}")

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
