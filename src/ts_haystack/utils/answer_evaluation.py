# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Answer evaluation utilities for TS-Haystack benchmark.

Provides task-type-aware answer comparison logic with:
- Boolean answer normalization (handles "Yes, it does appear." -> "yes")
- Time range parsing (preserves milliseconds, doesn't split on ".")
- IoU calculation for time range comparisons
- Integer extraction
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Union

from ts_haystack.utils.timestamp_utils import (
    parse_time_string,
)


# Regex pattern for timestamps with AM/PM and optional milliseconds
# Matches: "3:25:04.240 AM", "8:35:17 PM", "12:00:00.000 AM", etc.
TIME_PATTERN = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\s*[AP]M)"


def extract_final_answer(rationale: str, answer_type: str) -> str:
    """
    Extract the final answer from a chain-of-thought rationale.

    Args:
        rationale: Full rationale text (may contain multiple reasoning steps)
        answer_type: Type of answer (boolean, integer, timestamp, category, time_range)

    Returns:
        Extracted answer string
    """
    rationale = rationale.strip()

    # Find the last occurrence of "Answer:" (case-insensitive)
    matches = list(re.finditer(r"answer:\s*", rationale, re.IGNORECASE))

    if matches:
        start = matches[-1].end()
        answer = rationale[start:].strip()

        if answer_type == "boolean":
            # For boolean, just extract yes/no from the start
            answer_lower = answer.lower()
            if answer_lower.startswith("yes") or "yes" in answer_lower[:10]:
                return "Yes"
            elif answer_lower.startswith("no") or "no" in answer_lower[:10]:
                return "No"
            return answer

        elif answer_type == "integer":
            # For integer, extract first number
            match = re.search(r"\d+", answer)
            if match:
                return match.group()
            return answer

        elif answer_type in ("time_range", "timestamp"):
            # For time-based answers, DON'T split on "." as it destroys timestamps
            # Only split on newline to get the first line
            answer = answer.split("\n")[0].strip()
            # Remove trailing punctuation but NOT the timestamp periods
            answer = re.sub(r"[,;:!?]+$", "", answer)
            return answer

        else:
            # For category and other types, take first line and remove trailing punctuation
            answer = answer.split("\n")[0].strip()
            answer = re.sub(r"[.,;:!?]+$", "", answer)
            return answer
    else:
        # Fallback: return last word
        words = rationale.split()
        if words:
            return words[-1].rstrip(".,;:!?")

    return ""


def parse_time_range(answer_text: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse start/end times from answers containing time ranges.

    Handles formats like:
    - "The walking bout is from 3:25:04.240 AM to 3:25:08.570 AM."
    - "From 8:35:17.015 AM to 8:35:20.065 AM."
    - "3:25:04.240 AM to 3:25:08.570 AM"

    Args:
        answer_text: Text containing a time range

    Returns:
        Tuple of (start_datetime, end_datetime) or None if parsing fails
    """
    matches = re.findall(TIME_PATTERN, answer_text, re.IGNORECASE)
    if len(matches) >= 2:
        try:
            start = parse_time_string(matches[0])
            end = parse_time_string(matches[-1])  # Use last match in case of multiple
            return (start, end)
        except ValueError:
            return None
    return None


def compute_time_range_iou(
    pred_range: Tuple[datetime, datetime],
    gt_range: Tuple[datetime, datetime],
) -> float:
    """
    Calculate Intersection over Union for two time ranges.

    Args:
        pred_range: Predicted (start, end) datetime tuple
        gt_range: Ground truth (start, end) datetime tuple

    Returns:
        IoU score between 0.0 (no overlap) and 1.0 (identical)
    """
    pred_start, pred_end = pred_range
    gt_start, gt_end = gt_range

    # Handle day wraparound if needed
    if pred_end < pred_start:
        pred_end = pred_end + timedelta(days=1)
    if gt_end < gt_start:
        gt_end = gt_end + timedelta(days=1)

    # Calculate intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)

    if inter_start >= inter_end:
        return 0.0  # No overlap

    intersection = (inter_end - inter_start).total_seconds()
    pred_duration = (pred_end - pred_start).total_seconds()
    gt_duration = (gt_end - gt_start).total_seconds()
    union = pred_duration + gt_duration - intersection

    return intersection / union if union > 0 else 0.0


def normalize_boolean(answer: str) -> Optional[str]:
    """
    Normalize boolean answers.

    Checks if answer STARTS WITH yes/no (handles "Yes, it does appear.").
    Also handles "it does" / "it doesn't" patterns.

    Args:
        answer: Answer text to normalize

    Returns:
        "yes", "no", or None if neither pattern found
    """
    answer_lower = answer.strip().lower()

    # Check if starts with yes/no
    if answer_lower.startswith("yes"):
        return "yes"
    if answer_lower.startswith("no"):
        return "no"

    # Also check for "it does" / "it doesn't" patterns within first 30 chars
    prefix = answer_lower[:30]
    if "it does" in prefix and "doesn't" not in prefix and "does not" not in prefix:
        return "yes"
    if "doesn't" in prefix or "does not" in prefix:
        return "no"

    return None


def normalize_integer(answer: str) -> Optional[int]:
    """
    Extract first integer from answer text.

    Args:
        answer: Answer text containing a number

    Returns:
        Extracted integer or None if not found
    """
    answer = str(answer).strip()
    match = re.search(r"\d+", answer)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    return None


def evaluate_answer(
    ground_truth: str,
    prediction: str,
    answer_type: str,
    iou_threshold: float = 0.5,
) -> Dict[str, Union[bool, float, str, None]]:
    """
    Main evaluation function with task-type-aware comparison.

    Args:
        ground_truth: Ground truth answer string
        prediction: Predicted answer string
        answer_type: Type of answer (boolean, integer, timestamp, category, time_range)
        iou_threshold: IoU threshold for time range correctness (default: 0.5)

    Returns:
        Dict with:
        - correct: bool - whether the answer is correct
        - iou: float or None - IoU score for time_range/timestamp types
        - normalized_gt: str - normalized ground truth
        - normalized_pred: str - normalized prediction
    """
    result = {
        "correct": False,
        "iou": None,
        "normalized_gt": ground_truth,
        "normalized_pred": prediction,
    }

    if answer_type == "boolean":
        gt_norm = normalize_boolean(ground_truth)
        pred_norm = normalize_boolean(prediction)
        result["normalized_gt"] = gt_norm if gt_norm else ground_truth.lower()
        result["normalized_pred"] = pred_norm if pred_norm else prediction.lower()

        if gt_norm is not None and pred_norm is not None:
            result["correct"] = gt_norm == pred_norm
        else:
            # Fallback to simple string comparison
            result["correct"] = (
                ground_truth.strip().lower() == prediction.strip().lower()
            )

    elif answer_type == "integer":
        gt_int = normalize_integer(ground_truth)
        pred_int = normalize_integer(prediction)
        result["normalized_gt"] = str(gt_int) if gt_int is not None else ground_truth
        result["normalized_pred"] = str(pred_int) if pred_int is not None else prediction

        if gt_int is not None and pred_int is not None:
            result["correct"] = gt_int == pred_int
        else:
            # Fallback to string comparison
            result["correct"] = ground_truth.strip() == prediction.strip()

    elif answer_type in ("time_range", "timestamp"):
        # Try to parse time ranges and compute IoU
        gt_range = parse_time_range(ground_truth)
        pred_range = parse_time_range(prediction)

        if gt_range is not None and pred_range is not None:
            iou = compute_time_range_iou(pred_range, gt_range)
            result["iou"] = iou
            result["correct"] = iou >= iou_threshold
        else:
            # Fallback to normalized string comparison (case-insensitive, strip whitespace)
            gt_clean = re.sub(r"\s+", " ", ground_truth.strip().lower())
            pred_clean = re.sub(r"\s+", " ", prediction.strip().lower())
            result["correct"] = gt_clean == pred_clean

    else:
        # Category and other types: simple normalized string comparison
        gt_norm = ground_truth.strip().lower()
        pred_norm = prediction.strip().lower()
        # Remove trailing punctuation for comparison
        gt_norm = re.sub(r"[.,;:!?]+$", "", gt_norm)
        pred_norm = re.sub(r"[.,;:!?]+$", "", pred_norm)
        result["normalized_gt"] = gt_norm
        result["normalized_pred"] = pred_norm
        result["correct"] = gt_norm == pred_norm

    return result
