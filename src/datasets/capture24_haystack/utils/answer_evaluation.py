# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

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

from src.datasets.capture24.evaluation import WILLETTS_SPECIFIC_2018_LABELS
from src.datasets.capture24_haystack.utils.timestamp_utils import (
    parse_time_string,
)


# Regex pattern for timestamps with optional AM/PM and optional milliseconds.
# Matches both 12-hour ("3:25:04.240 AM", "8:35:17 PM") and 24-hour
# ("00:07:45", "23:45:12.500") formats. AM/PM is optional so window-relative
# 24h timestamps used by sleep PSG also parse.
TIME_PATTERN = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?(?:\s*[AP]M)?)"


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
    # No "Answer:" sentinel found — dispatch type-aware extraction over the
    # full rationale instead of the (pathological) "last word" fallback.
    # SFT-trained models commonly emit the answer without the sentinel because
    # template-based ground truths don't contain it.
    if answer_type == "boolean":
        # Run the prefix-based normalizer over the full rationale; if that
        # fails, scan the opening of the response for yes/no keywords.
        norm = normalize_boolean(rationale)
        if norm is not None:
            return "Yes" if norm == "yes" else "No"
        head = rationale.lower()[:30]
        if re.search(r"\b(yes|present|is present|does occur|did occur|came before|came after)\b", head):
            return "Yes"
        if re.search(r"\b(no|not present|is absent|doesn't|does not|did not)\b", head):
            return "No"
        return rationale

    if answer_type == "integer":
        match = re.search(r"-?\d+", rationale)
        if match:
            return match.group()
        return rationale

    if answer_type in ("time_range", "timestamp"):
        # Return the full text; parse_time_range will pull the timestamps out.
        return rationale.split("\n")[0].strip()

    # Category or unknown: return full text; the scorer does symmetric
    # containment against the ground truth.
    return rationale.split("\n")[0].strip()


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


def _gt_activity_in_prediction(gt: str, pred: str) -> bool:
    """Check if the ground-truth activity label appears anywhere in the prediction.

    Normalizes hyphens to spaces so e.g. "household chores" matches
    "household-chores". Matches symmetrically: either gt inside pred, or the
    final content token of pred inside gt. This handles verbose ground truths
    like "The subject was in REM." being matched by a short prediction "REM".
    """
    gt_spaced = gt.lower().replace("-", " ")
    pred_spaced = pred.lower().replace("-", " ")
    if gt_spaced in pred_spaced or pred_spaced in gt_spaced:
        return True
    # Token-level fallback: match on the last content word of each side.
    gt_tokens = re.findall(r"[A-Za-z0-9]+", gt_spaced)
    pred_tokens = re.findall(r"[A-Za-z0-9]+", pred_spaced)
    if gt_tokens and pred_tokens and gt_tokens[-1] == pred_tokens[-1]:
        return True
    return False


def evaluate_answer(
    ground_truth: str,
    prediction: str,
    answer_type: str,
    iou_threshold: float = 0.25,
    timestamp_tolerance_s: float = 0.5,
) -> Dict[str, Union[bool, float, str, None]]:
    """
    Main evaluation function with task-type-aware comparison.

    Dispatch is strict on the declared answer_type:
      * "time_range" → IoU against the parsed (start, end) in the prediction;
      * "timestamp"  → absolute-seconds difference against the single parsed
                       time in the prediction.
    Generators must emit the correct answer_type for this to score correctly.

    Args:
        ground_truth: Ground truth answer string
        prediction: Predicted answer string
        answer_type: Type of answer (boolean, integer, timestamp, category, time_range)
        iou_threshold: IoU threshold for time_range correctness in [0, 1]
            (default: 0.25). Only used for answer_type="time_range".
        timestamp_tolerance_s: Absolute time tolerance in seconds for
            answer_type="timestamp" (default: 0.5). A prediction is correct
            iff |pred - gt| <= timestamp_tolerance_s.

    Returns:
        Dict with:
        - correct: bool - whether the answer is correct
        - iou: float or None - IoU in [0, 1] when scored as a time range
        - timestamp_error_s: float or None - absolute seconds off when
          scored as a single timestamp
        - normalized_gt: str - normalized ground truth
        - normalized_pred: str - normalized prediction
    """
    result = {
        "correct": False,
        "iou": None,
        "timestamp_error_s": None,
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

    elif answer_type == "time_range":
        gt_range = parse_time_range(ground_truth)
        pred_range = parse_time_range(prediction)

        if gt_range is not None and pred_range is not None:
            iou = compute_time_range_iou(pred_range, gt_range)
            result["iou"] = iou
            result["correct"] = iou >= iou_threshold
        else:
            # Handle negatives like anomaly localization where GT or pred is
            # "no anomaly" rather than a range.
            gt_no = ground_truth.strip().lower().startswith("no")
            pred_no = prediction.strip().lower().startswith("no")
            if gt_no and pred_no:
                result["correct"] = True
            elif gt_range is None and pred_range is not None:
                result["correct"] = False
            elif gt_range is not None and pred_range is None:
                result["correct"] = False
            else:
                gt_clean = re.sub(r"\s+", " ", ground_truth.strip().lower())
                pred_clean = re.sub(r"\s+", " ", prediction.strip().lower())
                result["correct"] = gt_clean == pred_clean

    elif answer_type == "timestamp":
        # Single-timestamp answers scored by absolute-seconds tolerance.
        gt_matches = re.findall(TIME_PATTERN, ground_truth, re.IGNORECASE)
        pred_matches = re.findall(TIME_PATTERN, prediction, re.IGNORECASE)
        gt_dt = None
        pred_dt = None
        if gt_matches:
            try:
                gt_dt = parse_time_string(gt_matches[0])
            except ValueError:
                pass
        if pred_matches:
            try:
                pred_dt = parse_time_string(pred_matches[0])
            except ValueError:
                pass

        if gt_dt is not None and pred_dt is not None:
            diff_s = abs((pred_dt - gt_dt).total_seconds())
            result["timestamp_error_s"] = diff_s
            result["correct"] = diff_s <= timestamp_tolerance_s
        else:
            gt_no = ground_truth.strip().lower().startswith("no")
            pred_no = prediction.strip().lower().startswith("no")
            if gt_no and pred_no:
                result["correct"] = True
            elif gt_dt is None and pred_dt is not None:
                result["correct"] = False
            elif gt_dt is not None and pred_dt is None:
                result["correct"] = False
            else:
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

        if gt_norm == pred_norm:
            result["correct"] = True
        else:
            # Verbose predictions often contain the correct activity label
            # plus extra context (e.g. "mostly walking, with a brief sports
            # burst" for GT "walking").  Check if the GT label appears
            # anywhere in the prediction.
            result["correct"] = _gt_activity_in_prediction(gt_norm, pred_norm)

    return result
