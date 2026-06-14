# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Bout parsing utilities for the ARTS orchestrator.

Handles extraction and formatting of <bout> tags, classifier results,
timestamp-to-sample-index conversion, and the ground-truth OracleClassifier.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import timedelta

from src.datasets.capture24_haystack.utils.timestamp_utils import (
    parse_time_string,
    format_timestamp,
)


# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------

BOUT_OPEN = "<bout>"
BOUT_CLOSE = "</bout>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
CLASSIFIER_RESULTS_HEADER = "[Classifier results]"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches: <bout>6:04:54.687 AM - 6:05:02.368 AM</bout>
_BOUT_PATTERN = re.compile(
    rf"{re.escape(BOUT_OPEN)}\s*(.+?)\s*-\s*(.+?)\s*{re.escape(BOUT_CLOSE)}",
    re.IGNORECASE,
)

# Matches: <answer>...</answer>
_ANSWER_PATTERN = re.compile(
    rf"{re.escape(ANSWER_OPEN)}\s*(.*?)\s*{re.escape(ANSWER_CLOSE)}",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: "Answer: ..." at end of text (existing CoT convention)
_ANSWER_COLON_PATTERN = re.compile(r"Answer:\s*(.+?)$", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_bout_tags(text: str) -> list[tuple[str, str]]:
    """Extract (start_timestamp, end_timestamp) pairs from <bout> tags.

    Args:
        text: Orchestrator output containing <bout>start - end</bout> spans.

    Returns:
        List of (start_ts, end_ts) string tuples.

    Example:
        >>> parse_bout_tags("<bout>6:04:54.687 AM - 6:05:02.368 AM</bout>")
        [("6:04:54.687 AM", "6:05:02.368 AM")]
    """
    return _BOUT_PATTERN.findall(text)


def find_bout_truncation_point(text: str) -> int | None:
    """Find the truncation point right after the first batch of <bout> tags.

    The model may generate <bout> tags, then hallucinate [Classifier results],
    then generate more <bout> tags. We want to truncate after the FIRST
    contiguous group of bouts — i.e., right before any hallucinated content.

    Strategy: find the first [Classifier results] header. All bout matches
    ending before it belong to the first batch. Truncate after the last of
    those. If no header found, use the last bout match.

    Args:
        text: Generated text to analyze.

    Returns:
        Character index to truncate at (exclusive), or None if no bouts.
    """
    matches = list(_BOUT_PATTERN.finditer(text))
    if not matches:
        return None

    # Find the first hallucinated [Classifier results] header
    cr_pos = text.find(CLASSIFIER_RESULTS_HEADER)

    if cr_pos >= 0:
        # Keep only bout matches that END before the hallucinated header
        clean_matches = [m for m in matches if m.end() <= cr_pos]
        if clean_matches:
            return clean_matches[-1].end()
        # All matches are after the header — shouldn't happen, but use first
        return matches[0].end()

    # No hallucinated header — use the last match
    return matches[-1].end()


def extract_answer_tag(text: str) -> str | None:
    """Extract the answer from <answer>...</answer> tags.

    Falls back to "Answer: ..." pattern if no tags found.

    Args:
        text: Model output text.

    Returns:
        Extracted answer string, or None if not found.
    """
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    match = _ANSWER_COLON_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_classifier_results(results: list[dict]) -> str:
    """Format classifier results as text to inject into the conversation.

    Args:
        results: List of dicts with keys: start, end, label, confidence.

    Returns:
        Formatted string, e.g.:
            [Classifier results]
            6:04:54.687 AM - 6:05:02.368 AM: sports (0.97)
            6:05:13.949 AM - 6:05:27.680 AM: sitting (0.91)
    """
    lines = [
        f"{r['start']} - {r['end']}: {r['label']} ({r['confidence']:.2f})"
        for r in results
    ]
    return CLASSIFIER_RESULTS_HEADER + "\n" + "\n".join(lines)


def format_chunk_results(
    start_ts: str,
    end_ts: str,
    chunk_results: list[tuple[str, float]],
) -> list[dict]:
    """Convert per-chunk classify_bout output to result dicts.

    For bouts spanning multiple 10 s chunks, each chunk gets its own
    result dict with interpolated sub-timestamps.

    Args:
        start_ts: Bout start timestamp string (e.g., "6:04:54.687 AM").
        end_ts: Bout end timestamp string (e.g., "6:05:02.368 AM").
        chunk_results: List of (class_name, confidence) from HARClassifier.classify_bout().

    Returns:
        List of result dicts for format_classifier_results().
    """
    n = len(chunk_results)
    if n == 0:
        return []

    if n == 1:
        label, conf = chunk_results[0]
        return [{"start": start_ts, "end": end_ts, "label": label, "confidence": conf}]

    # Multiple chunks — compute sub-timestamps by interpolating between bout start/end
    bout_start_dt = parse_time_string(start_ts)
    bout_end_dt = parse_time_string(end_ts)
    bout_duration = (bout_end_dt - bout_start_dt).total_seconds()
    chunk_duration = bout_duration / n

    results = []
    for i, (label, conf) in enumerate(chunk_results):
        chunk_start_dt = bout_start_dt + timedelta(seconds=i * chunk_duration)
        chunk_end_dt = bout_start_dt + timedelta(seconds=(i + 1) * chunk_duration)
        results.append({
            "start": format_timestamp(chunk_start_dt),
            "end": format_timestamp(chunk_end_dt),
            "label": label,
            "confidence": conf,
        })
    return results


# ---------------------------------------------------------------------------
# Timestamp <-> sample index conversion
# ---------------------------------------------------------------------------


def bout_timestamps_to_sample_indices(
    start_ts: str,
    end_ts: str,
    recording_start: str,
    recording_end: str,
    context_length: int,
) -> tuple[int, int]:
    """Convert bout timestamp strings to sample indices within the recording.

    Uses linear interpolation between recording start/end times,
    matching the convention used by timestamp_utils.samples_to_timestamp().

    Args:
        start_ts: Bout start timestamp (e.g., "6:04:54.687 AM").
        end_ts: Bout end timestamp (e.g., "6:05:02.368 AM").
        recording_start: Recording window start time (e.g., "6:04:31.635 AM").
        recording_end: Recording window end time (e.g., "6:06:11.635 AM").
        context_length: Total number of samples in the recording (e.g., 10000).

    Returns:
        (start_idx, end_idx) sample indices. end_idx is exclusive.
    """
    rec_start = parse_time_string(recording_start)
    rec_end = parse_time_string(recording_end)
    bout_start = parse_time_string(start_ts)
    bout_end = parse_time_string(end_ts)

    total_duration = (rec_end - rec_start).total_seconds()
    if total_duration <= 0:
        return 0, context_length

    start_frac = (bout_start - rec_start).total_seconds() / total_duration
    end_frac = (bout_end - rec_start).total_seconds() / total_duration

    start_idx = int(round(start_frac * (context_length - 1)))
    end_idx = int(round(end_frac * (context_length - 1)))

    # Clamp and make end exclusive
    start_idx = max(0, min(start_idx, context_length - 1))
    end_idx = max(start_idx + 1, min(end_idx + 1, context_length))

    return start_idx, end_idx


# ---------------------------------------------------------------------------
# Oracle classifier — resolves bout timestamps to ground-truth labels
# ---------------------------------------------------------------------------


class OracleClassifier:
    """Resolves bout timestamps to ground-truth labels from sample metadata.

    Builds a segment map from needles + difficulty_config (same source data
    used in ``oracle_utils.format_oracle_timeline``), then for any queried
    timestamp range, returns the majority activity with its overlap fraction
    as confidence. Segments below ``min_confidence`` are returned as None.

    Args:
        needles: JSON string or list of needle dicts from sample metadata.
        difficulty_config: JSON string or dict from sample metadata.
        recording_time_start: Recording start timestamp.
        recording_time_end: Recording end timestamp.
        context_length_samples: Total samples in the recording.
        min_confidence: Minimum overlap fraction to return a label (default 0.6).
    """

    def __init__(
        self,
        needles: str | list[dict],
        difficulty_config: str | dict,
        recording_time_start: str,
        recording_time_end: str,
        context_length_samples: int,
        min_confidence: float = 0.6,
    ):
        self.min_confidence = min_confidence
        self.recording_start = recording_time_start
        self.recording_end = recording_time_end

        # Parse JSON if needed
        if isinstance(needles, str):
            needles = json.loads(needles) if needles else []
        if isinstance(difficulty_config, str):
            difficulty_config = json.loads(difficulty_config) if difficulty_config else {}
        needles = needles or []
        difficulty_config = difficulty_config or {}

        bg_activities = difficulty_config.get("background_activities", [])
        self.primary_bg = bg_activities[0] if bg_activities else "unknown"
        global_timeline = difficulty_config.get("global_timeline")

        # Build segments: (start_frac, end_frac, activity)
        segments: list[tuple[float, float, str]] = []

        if global_timeline:
            for entry in global_timeline:
                if len(entry) >= 3:
                    segments.append((float(entry[0]), float(entry[1]), entry[2]))

        # Add needles
        for needle in needles:
            if not isinstance(needle, dict):
                continue
            activity = needle.get("activity", "unknown")
            insert_frac = needle.get("insert_position_frac", 0.0)
            duration_samples = needle.get("duration_samples", 0)
            if context_length_samples > 0 and duration_samples > 0:
                duration_frac = duration_samples / context_length_samples
                end_frac = min(insert_frac + duration_frac, 1.0)
            else:
                end_frac = min(insert_frac + 0.01, 1.0)
            segments.append((insert_frac, end_frac, activity))

        # Fill background gaps if no global_timeline
        if not global_timeline:
            needle_segs = sorted(segments, key=lambda x: x[0])
            filled: list[tuple[float, float, str]] = []
            pos = 0.0
            for sf, ef, _ in needle_segs:
                if sf > pos:
                    filled.append((pos, sf, self.primary_bg))
                pos = max(pos, ef)
            if pos < 1.0:
                filled.append((pos, 1.0, self.primary_bg))
            segments = filled + needle_segs

        self.segments = sorted(segments, key=lambda x: x[0])

        # Precompute time boundaries
        self._start_dt = parse_time_string(recording_time_start)
        self._end_dt = parse_time_string(recording_time_end)
        if self._end_dt < self._start_dt:
            self._end_dt += timedelta(days=1)
        self._total_seconds = (self._end_dt - self._start_dt).total_seconds()

    def _ts_to_frac(self, ts: str) -> float:
        """Convert a timestamp string to a fractional position in the recording."""
        dt = parse_time_string(ts)
        if dt < self._start_dt:
            dt += timedelta(days=1)
        offset = (dt - self._start_dt).total_seconds()
        return max(0.0, min(1.0, offset / self._total_seconds)) if self._total_seconds > 0 else 0.0

    def classify_bout(self, start_ts: str, end_ts: str) -> list[tuple[str, float]]:
        """Classify a bout by looking up ground-truth labels.

        Returns ALL activities overlapping the queried range, each with
        confidence = overlap_fraction.  This makes the oracle a true oracle:
        it always returns correct, complete information.  If a bout spans
        two activities, both are returned so the model can reason about them.

        Args:
            start_ts: Bout start timestamp string.
            end_ts: Bout end timestamp string.

        Returns:
            List of (label, confidence) tuples sorted by confidence descending.
            Always non-empty — falls back to primary background activity.
        """
        q_start = self._ts_to_frac(start_ts)
        q_end = self._ts_to_frac(end_ts)
        if q_end <= q_start:
            return [(self.primary_bg, 0.50)]

        query_duration = q_end - q_start

        # Accumulate overlap per activity
        overlaps: dict[str, float] = {}
        for sf, ef, activity in self.segments:
            overlap_start = max(q_start, sf)
            overlap_end = min(q_end, ef)
            if overlap_end > overlap_start:
                overlaps[activity] = overlaps.get(activity, 0.0) + (overlap_end - overlap_start)

        if not overlaps:
            return [(self.primary_bg, 0.50)]

        # Return all overlapping activities, sorted by confidence descending
        results = [
            (activity, overlap / query_duration)
            for activity, overlap in overlaps.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results