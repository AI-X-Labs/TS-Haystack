# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluation utilities for Capture24 Classification task.
Provides classification-specific metrics: accuracy, precision, recall, F1,
balanced accuracy, confusion matrix, and detailed reporting.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# WillettsSpecific2018 labels (10 classes)
WILLETTS_SPECIFIC_2018_LABELS = [
    "bicycling",
    "household-chores",
    "manual-work",
    "mixed-activity",
    "sitting",
    "sleep",
    "sports",
    "standing",
    "vehicle",
    "walking",
]


def normalize_label(text: str) -> str:
    """
    Normalize a predicted label string for comparison.

    Args:
        text: Raw predicted text

    Returns:
        Normalized label string
    """
    if not text:
        return ""

    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Remove common prefixes that might appear in generated text
    prefixes_to_remove = [
        "the activity is",
        "the person is",
        "activity:",
        "answer:",
        "the activity being performed is",
        "based on the data,",
        "based on the accelerometer data,",
        "the person is performing",
        "this appears to be",
        "the data suggests",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove punctuation at the end
    text = text.rstrip(".,!?;:")

    # Handle common variations
    label_variations = {
        "sleeping": "sleep",
        "asleep": "sleep",
        "laying down": "sleep",
        "lying down": "sleep",
        "resting": "sleep",
        "sit": "sitting",
        "seated": "sitting",
        "sat": "sitting",
        "stand": "standing",
        "stood": "standing",
        "walk": "walking",
        "walked": "walking",
        "walks": "walking",
        "bike": "bicycling",
        "biking": "bicycling",
        "cycling": "bicycling",
        "cycle": "bicycling",
        "riding a bicycle": "bicycling",
        "riding bicycle": "bicycling",
        "driving": "vehicle",
        "car": "vehicle",
        "bus": "vehicle",
        "train": "vehicle",
        "transport": "vehicle",
        "traveling": "vehicle",
        "travel": "vehicle",
        "in a vehicle": "vehicle",
        "sport": "sports",
        "exercise": "sports",
        "exercising": "sports",
        "gym": "sports",
        "workout": "sports",
        "working out": "sports",
        "chores": "household-chores",
        "housework": "household-chores",
        "cleaning": "household-chores",
        "cooking": "household-chores",
        "household chores": "household-chores",
        "household tasks": "household-chores",
        "doing chores": "household-chores",
        "manual labor": "manual-work",
        "manual labour": "manual-work",
        "labor": "manual-work",
        "labour": "manual-work",
        "physical work": "manual-work",
        "mixed": "mixed-activity",
        "multiple activities": "mixed-activity",
        "various": "mixed-activity",
        "combination": "mixed-activity",
        "mixed activities": "mixed-activity",
    }

    if text in label_variations:
        return label_variations[text]

    # Try to extract label from longer text
    for label in WILLETTS_SPECIFIC_2018_LABELS:
        if label in text:
            return label

    # Check variations in longer text
    for variation, normalized in label_variations.items():
        if variation in text:
            return normalized

    return text


def extract_predicted_label(prediction: str, labels: Optional[List[str]] = None) -> str:
    """
    Extract the predicted label from model output.

    Args:
        prediction: Raw model prediction string
        labels: List of valid labels (default: WillettsSpecific2018)

    Returns:
        Extracted and normalized label
    """
    if labels is None:
        labels = WILLETTS_SPECIFIC_2018_LABELS

    if not prediction:
        return ""

    # Split by newlines and take the first meaningful line
    lines = prediction.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a label directly
        normalized = normalize_label(line)
        if normalized in labels:
            return normalized

        # Try to extract from patterns like "Activity: walking"
        patterns = [
            r"activity[:\s]+(\w+(?:-\w+)?)",
            r"^(\w+(?:-\w+)?)$",  # Single word
            r"the (?:activity|person) (?:is|was) (\w+(?:-\w+)?)",
            r"answer[:\s]+(\w+(?:-\w+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                extracted = normalize_label(match.group(1))
                if extracted in labels:
                    return extracted

    # Fallback: normalize the entire prediction
    return normalize_label(prediction)


def compute_balanced_accuracy(
    confusion_matrix: np.ndarray,
    labels: List[str],
) -> float:
    """
    Compute balanced accuracy from confusion matrix.

    Balanced accuracy is the average of recall for each class,
    which accounts for class imbalance.

    Args:
        confusion_matrix: N x N confusion matrix
        labels: List of class labels

    Returns:
        Balanced accuracy (0.0 to 1.0)
    """
    num_classes = len(labels)
    recalls = []

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        total_actual = confusion_matrix[i, :].sum()

        if total_actual > 0:
            recall = true_positives / total_actual
            recalls.append(recall)

    if not recalls:
        return 0.0

    return np.mean(recalls)


def evaluate_classification(
    ground_truths: List[str],
    predictions: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate classification predictions against ground truth.

    Args:
        ground_truths: List of ground truth labels
        predictions: List of predicted labels (raw model output)
        labels: List of valid label names (default: WillettsSpecific2018)

    Returns:
        Dictionary with evaluation results including:
        - overall: accuracy, macro_f1, balanced_accuracy, etc.
        - per_class: precision, recall, f1, support per class
        - confusion_matrix: NxN matrix
        - prediction_distribution: count of each predicted label
    """
    if labels is None:
        labels = WILLETTS_SPECIFIC_2018_LABELS

    label_to_idx = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    correct = 0
    total = len(ground_truths)

    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_predicted = defaultdict(int)

    detailed_results = []

    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        gt_normalized = normalize_label(gt)
        pred_normalized = extract_predicted_label(pred, labels)

        is_correct = (gt_normalized == pred_normalized)

        if is_correct:
            correct += 1

        # Update per-class stats
        per_class_total[gt_normalized] += 1
        per_class_predicted[pred_normalized] += 1
        if is_correct:
            per_class_correct[gt_normalized] += 1

        # Update confusion matrix
        if gt_normalized in label_to_idx and pred_normalized in label_to_idx:
            confusion_matrix[label_to_idx[gt_normalized], label_to_idx[pred_normalized]] += 1

        detailed_results.append({
            "index": i,
            "ground_truth": gt_normalized,
            "prediction": pred_normalized,
            "raw_prediction": pred[:100] if len(pred) > 100 else pred,
            "correct": is_correct,
        })

    # Calculate per-class metrics
    per_class_metrics = {}
    for label in labels:
        tp = per_class_correct.get(label, 0)
        total_actual = per_class_total.get(label, 0)
        total_pred = per_class_predicted.get(label, 0)

        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_actual if total_actual > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / total_actual if total_actual > 0 else 0.0

        per_class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": total_actual,
            "predicted": total_pred,
            "correct": tp,
        }

    # Calculate macro and weighted averages
    macro_precision = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])

    total_support = sum(m["support"] for m in per_class_metrics.values())
    weighted_precision = sum(m["precision"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
    weighted_recall = sum(m["recall"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_support if total_support > 0 else 0.0

    # Compute balanced accuracy
    balanced_accuracy = compute_balanced_accuracy(confusion_matrix, labels)

    # Prediction distribution
    prediction_distribution = dict(per_class_predicted)

    return {
        "overall": {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_accuracy,
        },
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "labels": labels,
        "prediction_distribution": prediction_distribution,
        "detailed_results": detailed_results,
    }


def print_classification_summary(
    results: Dict[str, Any],
    split: str = "test",
    context_length: Optional[float] = None,
):
    """
    Print a formatted summary of classification results.

    Args:
        results: Evaluation results dictionary
        split: Name of the split (for display)
        context_length: Context length in seconds (optional, for display)
    """
    ctx_str = f" [{context_length}s]" if context_length else ""
    print(f"\n{'='*70}")
    print(f"Classification Results ({split}){ctx_str}")
    print(f"{'='*70}")

    overall = results["overall"]
    print(f"\nOverall Metrics:")
    print(f"  Test Loss:          N/A")  # Loss computed separately
    print(f"  Accuracy:           {overall['accuracy']:.4f} ({overall['correct']}/{overall['total']})")
    print(f"  Macro-F1:           {overall['macro_f1']:.4f}  <- PRIMARY METRIC")
    print(f"  Balanced Accuracy:  {overall['balanced_accuracy']:.4f}  <- SECONDARY METRIC")
    print(f"")
    print(f"  Macro Precision:    {overall['macro_precision']:.4f}")
    print(f"  Macro Recall:       {overall['macro_recall']:.4f}")
    print(f"  Weighted F1:        {overall['weighted_f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"  {'Label':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for label in results["labels"]:
        m = results["per_class"].get(label, {})
        print(
            f"  {label:<20} {m.get('correct', 0):>8} {m.get('support', 0):>8} "
            f"{m.get('accuracy', 0)*100:>9.2f}% {m.get('f1', 0)*100:>9.2f}%"
        )

    # Prediction distribution
    print(f"\nPrediction Distribution:")
    print(f"  {'Predicted':<20} {'Count':>8} {'Percent':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")

    pred_dist = results.get("prediction_distribution", {})
    total_preds = sum(pred_dist.values())
    for label, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / total_preds if total_preds > 0 else 0
            print(f"  {label:<20} {count:>8} {pct:>9.2f}%")

    print(f"\n{'='*70}")


def print_samples_per_activity(
    results: Dict[str, Any],
    context_length: float,
):
    """
    Print the number of samples per activity for a context length.

    Args:
        results: Evaluation results dictionary
        context_length: Context length in seconds
    """
    print(f"\nSamples per Activity (Context Length: {context_length}s):")
    for label in results["labels"]:
        m = results["per_class"].get(label, {})
        support = m.get("support", 0)
        print(f"  {label:<20}: {support:>6} samples")


def format_confusion_matrix(
    confusion_matrix: List[List[int]],
    labels: List[str],
) -> str:
    """
    Format confusion matrix as a string for display.

    Args:
        confusion_matrix: 2D confusion matrix
        labels: List of label names

    Returns:
        Formatted string representation
    """
    # Create short labels for display
    short_labels = [l[:8] for l in labels]
    max_label_len = max(len(l) for l in short_labels)

    # Header
    header = " " * (max_label_len + 2)
    for sl in short_labels:
        header += f"{sl:>8}"

    lines = [header]

    # Rows
    for i, (label, row) in enumerate(zip(short_labels, confusion_matrix)):
        line = f"{label:<{max_label_len+2}}"
        for val in row:
            line += f"{val:>8}"
        lines.append(line)

    return "\n".join(lines)


def aggregate_results_by_context_length(
    all_results: Dict[float, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate results across all context lengths.

    Args:
        all_results: Dictionary mapping context length to results

    Returns:
        Aggregated summary dictionary
    """
    summary = {
        "context_lengths": [],
        "accuracies": [],
        "macro_f1s": [],
        "balanced_accuracies": [],
        "sample_counts": [],
    }

    for ctx_len in sorted(all_results.keys()):
        results = all_results[ctx_len]
        overall = results["overall"]

        summary["context_lengths"].append(ctx_len)
        summary["accuracies"].append(overall["accuracy"])
        summary["macro_f1s"].append(overall["macro_f1"])
        summary["balanced_accuracies"].append(overall["balanced_accuracy"])
        summary["sample_counts"].append(overall["total"])

    # Compute averages
    summary["mean_accuracy"] = np.mean(summary["accuracies"])
    summary["mean_macro_f1"] = np.mean(summary["macro_f1s"])
    summary["mean_balanced_accuracy"] = np.mean(summary["balanced_accuracies"])

    return summary


def print_curriculum_summary(all_results: Dict[float, Dict[str, Any]]):
    """
    Print a summary of results across all curriculum stages.

    Args:
        all_results: Dictionary mapping context length to results
    """
    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING SUMMARY")
    print("=" * 80)

    print(f"\n{'Context':>10} {'Accuracy':>10} {'Macro-F1':>10} {'Balanced':>10} {'Samples':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for ctx_len in sorted(all_results.keys()):
        results = all_results[ctx_len]
        overall = results["overall"]
        print(
            f"{ctx_len:>9}s "
            f"{overall['accuracy']*100:>9.2f}% "
            f"{overall['macro_f1']*100:>9.2f}% "
            f"{overall['balanced_accuracy']*100:>9.2f}% "
            f"{overall['total']:>10}"
        )

    summary = aggregate_results_by_context_length(all_results)
    print(f"\n{'Average':>10} "
          f"{summary['mean_accuracy']*100:>9.2f}% "
          f"{summary['mean_macro_f1']*100:>9.2f}% "
          f"{summary['mean_balanced_accuracy']*100:>9.2f}%")

    print("=" * 80)
