#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Aggregate evaluation results from TS-Haystack validation/test output logs.

This script takes a JSON file from validation results (e.g., val_epoch_3.json)
and outputs comprehensive evaluation metrics aggregated by:
- Overall accuracy
- Per-task accuracy
- Per-context-length accuracy
- Per-task-per-context-length accuracy (cross-tabulation)
- Per-answer-type accuracy
- IoU statistics for time_range/timestamp tasks

Usage:
    # Basic usage - print summary to console
    python scripts/aggregate_eval_results.py \
        results/val_epoch_3.json

    # Save detailed report to file
    python scripts/aggregate_eval_results.py \
        results/val_epoch_3.json --output eval_report.json

    # Re-evaluate answers with custom IoU threshold
    python scripts/aggregate_eval_results.py \
        results/val_epoch_3.json --re-evaluate --iou-threshold 0.3

    # Show error samples for specific task
    python scripts/aggregate_eval_results.py \
        results/val_epoch_3.json --show-errors --task localization --max-errors 10
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ts_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)


# All available tasks in TS-Haystack
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

# Sampling rate for context length conversion
SAMPLING_RATE_HZ = 100

# Random baseline accuracy for each task
RANDOM_BASELINES = {
    "existence": 0.50,
    "localization": 0.00,
    "counting": 0.20,
    "ordering": 0.50,
    "state_query": 0.10,
    "antecedent": 0.10,
    "comparison": 0.00,
    "multi_hop": 0.00,
    "anomaly_detection": 0.50,
    "anomaly_localization": 0.00,
}

# Random baseline by answer type (fallback)
RANDOM_BASELINES_BY_TYPE = {
    "boolean": 0.50,
    "integer": 0.20,
    "category": 0.10,
    "time_range": 0.00,
    "timestamp": 0.00,
}


@dataclass
class MetricAccumulator:
    """Accumulates metrics for a group of samples."""

    correct: int = 0
    total: int = 0
    iou_values: List[float] = field(default_factory=list)

    def add(self, is_correct: bool, iou: Optional[float] = None):
        self.total += 1
        if is_correct:
            self.correct += 1
        if iou is not None:
            self.iou_values.append(iou)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def mean_iou(self) -> Optional[float]:
        return float(np.mean(self.iou_values)) if self.iou_values else None

    @property
    def std_iou(self) -> Optional[float]:
        return float(np.std(self.iou_values)) if len(self.iou_values) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
        }
        if self.iou_values:
            result["mean_iou"] = self.mean_iou
            result["std_iou"] = self.std_iou
            result["n_iou_samples"] = len(self.iou_values)
        return result


def samples_to_seconds(samples: int) -> float:
    """Convert samples to seconds at 100Hz."""
    return samples / SAMPLING_RATE_HZ


def format_context_length(samples: int) -> str:
    """Format context length as human-readable string."""
    seconds = samples_to_seconds(samples)
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def aggregate_results(
    outputs: List[Dict[str, Any]],
    re_evaluate: bool = False,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Aggregate evaluation results from output logs.

    Args:
        outputs: List of output dictionaries from validation JSON
        re_evaluate: If True, re-run evaluation on predictions
        iou_threshold: IoU threshold for time range correctness

    Returns:
        Dictionary with aggregated metrics
    """
    # Initialize accumulators
    overall = MetricAccumulator()
    by_task: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_context_length: Dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_answer_type: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_task_and_context: Dict[str, Dict[int, MetricAccumulator]] = defaultdict(
        lambda: defaultdict(MetricAccumulator)
    )

    # Process each output
    for output in outputs:
        task_type = output.get("task_type", "unknown")
        answer_type = output.get("answer_type", "unknown")
        context_length = output.get("context_length_samples", 0)

        if re_evaluate:
            ground_truth = output.get("ground_truth", "")
            prediction = output.get("predicted_answer", "")

            if not prediction and output.get("prediction_preview"):
                prediction = extract_final_answer(
                    output["prediction_preview"], answer_type
                )

            eval_result = evaluate_answer(
                ground_truth=ground_truth,
                prediction=prediction,
                answer_type=answer_type,
                iou_threshold=iou_threshold,
            )
            is_correct = eval_result["correct"]
            iou = eval_result.get("iou")
        else:
            is_correct = output.get("correct", False)
            iou = output.get("iou")

        # Accumulate metrics
        overall.add(is_correct, iou)
        by_task[task_type].add(is_correct, iou)
        by_context_length[context_length].add(is_correct, iou)
        by_answer_type[answer_type].add(is_correct, iou)
        by_task_and_context[task_type][context_length].add(is_correct, iou)

    # Build result dictionary
    result = {
        "overall": overall.to_dict(),
        "by_task": {task: acc.to_dict() for task, acc in sorted(by_task.items())},
        "by_context_length": {
            context_length: {
                **acc.to_dict(),
                "context_seconds": samples_to_seconds(context_length),
                "context_formatted": format_context_length(context_length),
            }
            for context_length, acc in sorted(by_context_length.items())
        },
        "by_answer_type": {
            atype: acc.to_dict() for atype, acc in sorted(by_answer_type.items())
        },
        "by_task_and_context": {
            task: {
                context_length: acc.to_dict()
                for context_length, acc in sorted(context_accs.items())
            }
            for task, context_accs in sorted(by_task_and_context.items())
        },
    }

    return result


def get_error_samples(
    outputs: List[Dict[str, Any]],
    task_filter: Optional[str] = None,
    context_filter: Optional[int] = None,
    max_errors: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get samples where the model made errors.

    Args:
        outputs: List of output dictionaries
        task_filter: Filter to specific task type
        context_filter: Filter to specific context length
        max_errors: Maximum number of errors to return

    Returns:
        List of error samples with relevant fields
    """
    errors = []

    for output in outputs:
        if not output.get("correct", True):
            task_type = output.get("task_type", "unknown")
            context_length = output.get("context_length_samples", 0)

            if task_filter and task_type != task_filter:
                continue
            if context_filter and context_length != context_filter:
                continue

            errors.append({
                "sample_idx": output.get("sample_idx"),
                "task_type": task_type,
                "answer_type": output.get("answer_type"),
                "context_length_samples": context_length,
                "context_formatted": format_context_length(context_length),
                "question": output.get("question", "")[:200],
                "ground_truth": output.get("ground_truth", ""),
                "predicted_answer": output.get("predicted_answer", ""),
                "iou": output.get("iou"),
            })

            if len(errors) >= max_errors:
                break

    return errors


def print_summary(
    aggregated: Dict[str, Any],
    epoch: Optional[int] = None,
    split: Optional[str] = None,
    loss: Optional[float] = None,
):
    """Print a formatted summary of aggregated results."""
    print("=" * 70)
    print("TS-Haystack Evaluation Summary")
    print("=" * 70)

    if epoch is not None:
        print(f"Epoch: {epoch}")
    if split is not None:
        print(f"Split: {split}")
    if loss is not None:
        print(f"Loss: {loss:.4f}")

    overall = aggregated["overall"]

    weighted_baseline = 0.0
    total_samples = 0
    for task, metrics in aggregated["by_task"].items():
        task_baseline = RANDOM_BASELINES.get(task, 0.0)
        task_count = metrics["total"]
        weighted_baseline += task_baseline * task_count
        total_samples += task_count
    if total_samples > 0:
        weighted_baseline /= total_samples

    diff = overall['accuracy'] - weighted_baseline
    print(f"\nOverall Accuracy: {overall['accuracy']*100:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"Weighted Random Baseline: {weighted_baseline*100:.2f}% (vs Random: {diff*100:+.2f}%)")

    # Per-task accuracy
    print("\n" + "-" * 80)
    print("Per-Task Accuracy")
    print("-" * 80)
    print(f"{'Task':<22} {'Accuracy':>10} {'Baseline':>10} {'vs Random':>10} {'Correct':>8} {'Total':>8}")
    print("-" * 80)

    for task, metrics in aggregated["by_task"].items():
        acc = metrics['accuracy']
        acc_str = f"{acc*100:.1f}%"
        baseline = RANDOM_BASELINES.get(task, 0.0)
        baseline_str = f"{baseline*100:.0f}%"
        diff = acc - baseline
        diff_str = f"{diff*100:+.1f}%" if diff != 0 else "0.0%"
        print(f"{task:<22} {acc_str:>10} {baseline_str:>10} {diff_str:>10} {metrics['correct']:>8} {metrics['total']:>8}")

    # Per-context-length accuracy
    print("\n" + "-" * 80)
    print("Per-Context-Length Accuracy")
    print("-" * 80)
    print(f"{'Context Length':<18} {'Accuracy':>10} {'Correct':>10} {'Total':>10} {'Mean IoU':>12}")
    print("-" * 80)

    for context_length, metrics in aggregated["by_context_length"].items():
        ctx_str = metrics.get("context_formatted", f"{context_length} samples")
        acc_str = f"{metrics['accuracy']*100:.1f}%"
        iou_str = f"{metrics['mean_iou']:.3f}" if metrics.get("mean_iou") else "N/A"
        print(f"{ctx_str:<18} {acc_str:>10} {metrics['correct']:>10} {metrics['total']:>10} {iou_str:>12}")

    # Per-answer-type accuracy
    print("\n" + "-" * 80)
    print("Per-Answer-Type Accuracy")
    print("-" * 80)
    print(f"{'Answer Type':<15} {'Accuracy':>10} {'Baseline':>10} {'vs Random':>10} {'Correct':>8} {'Total':>8} {'Mean IoU':>10}")
    print("-" * 80)

    for atype, metrics in aggregated["by_answer_type"].items():
        acc = metrics['accuracy']
        acc_str = f"{acc*100:.1f}%"
        baseline = RANDOM_BASELINES_BY_TYPE.get(atype, 0.0)
        baseline_str = f"{baseline*100:.0f}%"
        diff = acc - baseline
        diff_str = f"{diff*100:+.1f}%" if diff != 0 else "0.0%"
        iou_str = f"{metrics['mean_iou']:.3f}" if metrics.get("mean_iou") else "N/A"
        print(f"{atype:<15} {acc_str:>10} {baseline_str:>10} {diff_str:>10} {metrics['correct']:>8} {metrics['total']:>8} {iou_str:>10}")

    # Cross-tabulation
    print("\n" + "-" * 80)
    print("Task x Context Length Accuracy Matrix")
    print("-" * 80)

    all_context_lengths = sorted(set(
        cl for task_data in aggregated["by_task_and_context"].values()
        for cl in task_data.keys()
    ))

    if all_context_lengths:
        header = f"{'Task':<20}"
        for cl in all_context_lengths:
            header += f" {format_context_length(cl):>10}"
        print(header)
        print("-" * 80)

        for task in sorted(aggregated["by_task_and_context"].keys()):
            row = f"{task:<20}"
            for cl in all_context_lengths:
                metrics = aggregated["by_task_and_context"][task].get(cl, {})
                if metrics:
                    acc_str = f"{metrics['accuracy']*100:.1f}%"
                else:
                    acc_str = "-"
                row += f" {acc_str:>10}"
            print(row)

    print("\n" + "=" * 70)


def print_errors(errors: List[Dict[str, Any]]):
    """Print error samples in a readable format."""
    print("\n" + "-" * 80)
    print(f"Error Samples ({len(errors)} shown)")
    print("-" * 80)

    for i, error in enumerate(errors):
        print(f"\n[{i+1}] Sample {error['sample_idx']} - {error['task_type']} ({error['context_formatted']})")
        print(f"    Question: {error['question'][:100]}...")
        print(f"    Ground Truth: {error['ground_truth']}")
        print(f"    Predicted: {error['predicted_answer']}")
        if error.get("iou") is not None:
            print(f"    IoU: {error['iou']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate TS-Haystack evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input_json",
        type=str,
        help="Path to validation/test output JSON file (e.g., val_epoch_3.json)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save detailed JSON report (optional)",
    )
    parser.add_argument(
        "--re-evaluate",
        action="store_true",
        help="Re-evaluate answers instead of using pre-computed correctness",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for time range correctness (default: 0.5)",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show sample errors",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=ALL_TASKS + ["all"],
        help="Filter errors to specific task (use with --show-errors)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Filter errors to specific context length in samples (use with --show-errors)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Maximum number of error samples to show (default: 10)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output JSON, no console summary",
    )

    args = parser.parse_args()

    # Load input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    outputs = data.get("outputs", [])
    if not outputs:
        print("Error: No outputs found in input JSON", file=sys.stderr)
        sys.exit(1)

    # Aggregate results
    aggregated = aggregate_results(
        outputs,
        re_evaluate=args.re_evaluate,
        iou_threshold=args.iou_threshold,
    )

    # Add metadata
    aggregated["metadata"] = {
        "input_file": str(input_path),
        "epoch": data.get("epoch"),
        "split": data.get("split"),
        "loss": data.get("loss"),
        "re_evaluated": args.re_evaluate,
        "iou_threshold": args.iou_threshold,
        "total_samples": len(outputs),
    }

    # Print summary
    if not args.quiet:
        print_summary(
            aggregated,
            epoch=data.get("epoch"),
            split=data.get("split"),
            loss=data.get("loss"),
        )

    # Show errors if requested
    if args.show_errors:
        task_filter = args.task if args.task != "all" else None
        errors = get_error_samples(
            outputs,
            task_filter=task_filter,
            context_filter=args.context_length,
            max_errors=args.max_errors,
        )
        if not args.quiet:
            print_errors(errors)
        aggregated["error_samples"] = errors

    # Save detailed report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        if not args.quiet:
            print(f"\nDetailed report saved to: {output_path}")

    return aggregated


if __name__ == "__main__":
    main()
