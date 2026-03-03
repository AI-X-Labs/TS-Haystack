#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Inspect TS-Haystack generated parquet files.

Displays sample contents in readable JSON format for manual review.

Usage:
    # Show summary of a parquet file
    python scripts/inspect_dataset.py \
        data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet

    # Show first 3 samples in full JSON
    python scripts/inspect_dataset.py \
        data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet \
        --samples 3

    # Show specific sample by index
    python scripts/inspect_dataset.py \
        data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet \
        --index 5

    # Show only metadata (no time series data)
    python scripts/inspect_dataset.py \
        data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet \
        --samples 3 --no-timeseries

    # Output as JSON file
    python scripts/inspect_dataset.py \
        data/capture24/ts_haystack/tasks/100s/existence/train/data.parquet \
        --samples 3 --output samples.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


def load_and_parse_sample(row: Dict[str, Any], include_timeseries: bool = True) -> Dict[str, Any]:
    """
    Parse a row from the parquet file into a readable dictionary.

    Handles JSON-encoded fields (needles, difficulty_config) and optionally
    truncates or excludes time series data.
    """
    sample = {}

    for key, value in row.items():
        # Parse JSON-encoded fields
        if key in ("needles", "difficulty_config") and isinstance(value, str):
            try:
                sample[key] = json.loads(value)
            except json.JSONDecodeError:
                sample[key] = value
        # Handle time series data
        elif key in ("x_axis", "y_axis", "z_axis"):
            if include_timeseries:
                if isinstance(value, list):
                    sample[key] = {
                        "length": len(value),
                        "first_5": value[:5],
                        "last_5": value[-5:],
                        "min": round(min(value), 4) if value else None,
                        "max": round(max(value), 4) if value else None,
                        "mean": round(sum(value) / len(value), 4) if value else None,
                    }
                else:
                    sample[key] = value
            else:
                if isinstance(value, list):
                    sample[f"{key}_length"] = len(value)
        else:
            sample[key] = value

    return sample


def get_summary(df: pl.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset."""
    summary = {
        "total_samples": len(df),
        "columns": df.columns,
    }

    # Task type distribution
    if "task_type" in df.columns:
        task_counts = df.group_by("task_type").len().to_dicts()
        summary["task_distribution"] = {d["task_type"]: d["len"] for d in task_counts}

    # Answer type distribution
    if "answer_type" in df.columns:
        answer_counts = df.group_by("answer_type").len().to_dicts()
        summary["answer_type_distribution"] = {d["answer_type"]: d["len"] for d in answer_counts}

    # Context length
    if "context_length_samples" in df.columns:
        summary["context_length_samples"] = df["context_length_samples"][0]

    # Validation stats
    if "is_valid" in df.columns:
        valid_count = df.filter(pl.col("is_valid") == True).height
        summary["valid_samples"] = valid_count
        summary["invalid_samples"] = len(df) - valid_count

    # Sample time series length
    if "x_axis" in df.columns:
        first_x = df["x_axis"][0]
        if isinstance(first_x, list):
            summary["timeseries_length"] = len(first_x)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Inspect TS-Haystack parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to parquet file to inspect",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=0,
        help="Number of samples to display (default: 0, summary only)",
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=None,
        help="Display specific sample by index",
    )
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Exclude time series data from output",
    )
    parser.add_argument(
        "--full-timeseries",
        action="store_true",
        help="Include full time series arrays (warning: large output)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output to JSON file instead of stdout",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)",
    )

    args = parser.parse_args()

    # Validate path
    if not args.parquet_path.exists():
        print(f"Error: File not found: {args.parquet_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading: {args.parquet_path}", file=sys.stderr)
    df = pl.read_parquet(args.parquet_path)

    # Build output
    output = {
        "file": str(args.parquet_path),
        "summary": get_summary(df),
    }

    # Add samples if requested
    samples_to_show = []

    if args.index is not None:
        if 0 <= args.index < len(df):
            samples_to_show = [args.index]
        else:
            print(f"Error: Index {args.index} out of range (0-{len(df)-1})", file=sys.stderr)
            sys.exit(1)
    elif args.samples > 0:
        samples_to_show = list(range(min(args.samples, len(df))))

    if samples_to_show:
        output["samples"] = []
        for idx in samples_to_show:
            row = df.row(idx, named=True)

            if args.full_timeseries:
                # Include full arrays
                sample = {}
                for key, value in row.items():
                    if key in ("needles", "difficulty_config") and isinstance(value, str):
                        try:
                            sample[key] = json.loads(value)
                        except json.JSONDecodeError:
                            sample[key] = value
                    else:
                        sample[key] = value
            else:
                sample = load_and_parse_sample(
                    row,
                    include_timeseries=not args.no_timeseries
                )

            sample["_index"] = idx
            output["samples"].append(sample)

    # Format output
    indent = None if args.compact else 2
    json_output = json.dumps(output, indent=indent, default=str)

    # Write output
    if args.output:
        args.output.write_text(json_output)
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
