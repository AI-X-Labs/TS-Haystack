# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Prompt construction for TS-Haystack CoT generation.

This module builds task-specific prompts with rich metadata for generating
grounded chain-of-thought rationales. Unlike Capture24 CoT which uses binary
classification, TS-Haystack CoT passes detailed ground truth metadata to the LLM
so it can generate reasoning that discovers patterns at the correct locations.

Key Design:
- LLM receives: plot + ground truth needle positions + task context
- LLM generates: natural reasoning that "discovers" the patterns
- Result: Factually grounded, varied rationales
"""

import json
from typing import Dict, List, Optional


def format_needle_metadata(needles: List[Dict]) -> str:
    """
    Format needle metadata for LLM context.

    Args:
        needles: List of needle dicts with keys:
                 - activity: str
                 - timestamp_start: str
                 - timestamp_end: str
                 - insert_position_frac: float
                 - duration_ms: int (optional)
                 - duration_samples: int (optional)

    Returns:
        Formatted string describing the inserted activity bouts
    """
    if not needles:
        return "No specific activity bouts were inserted. Analyze the natural recording."

    lines = ["INSERTED ACTIVITY BOUTS (ground truth for your reasoning):"]

    for i, needle in enumerate(needles, 1):
        activity = needle.get("activity", "unknown")
        start_time = needle.get("timestamp_start", "unknown")
        end_time = needle.get("timestamp_end", "unknown")
        position_pct = needle.get("insert_position_frac", 0) * 100

        # Calculate duration
        duration_ms = needle.get("duration_ms", 0)
        if duration_ms >= 60000:
            duration_str = f"{duration_ms / 60000:.1f} minutes"
        elif duration_ms > 0:
            duration_str = f"{duration_ms / 1000:.1f} seconds"
        else:
            duration_samples = needle.get("duration_samples", 0)
            duration_str = f"{duration_samples} samples"

        lines.append(f"""
Bout {i}:
  - Activity: {activity}
  - Time: {start_time} to {end_time}
  - Position: {position_pct:.1f}% into recording
  - Duration: {duration_str}""")

    return "\n".join(lines)


def format_background_timeline(difficulty_config: Dict) -> str:
    """
    Format background activity timeline for tasks that need global context.

    Used for state_query, antecedent tasks where the background composition matters.

    Args:
        difficulty_config: Dict containing task-specific config, may include:
                          - global_timeline: List of (start_frac, end_frac, activity) tuples
                          - background_activities: Set of activity names

    Returns:
        Formatted string describing the background activity composition
    """
    global_timeline = difficulty_config.get("global_timeline", [])
    background_activities = difficulty_config.get("background_activities", [])

    if global_timeline:
        lines = ["BACKGROUND ACTIVITY TIMELINE:"]
        for start_frac, end_frac, activity in global_timeline:
            duration_pct = (end_frac - start_frac) * 100
            lines.append(f"  - {activity}: {start_frac*100:.0f}% to {end_frac*100:.0f}% ({duration_pct:.1f}% of recording)")
        return "\n".join(lines)

    if background_activities:
        activities_str = ", ".join(sorted(background_activities))
        return f"BACKGROUND ACTIVITIES: {activities_str}"

    return ""


def get_task_context(task_type: str, difficulty_config: Dict) -> str:
    """
    Get task-specific context for the LLM.

    Each task type has different reasoning requirements, so we provide
    specific guidance on what the LLM should focus on.

    Args:
        task_type: Task name (existence, localization, etc.)
        difficulty_config: Dict containing task-specific parameters

    Returns:
        Task-specific context string for the LLM prompt
    """
    contexts = {
        "existence": f"""
TASK TYPE: Existence Detection
- Target activity to detect: {difficulty_config.get("target_activity", "unknown")}
- Is positive sample (activity present): {difficulty_config.get("is_positive", "unknown")}
- Your reasoning should describe signal characteristics that indicate presence/absence of this activity.""",

        "localization": f"""
TASK TYPE: Temporal Localization
- Target activity to locate: {difficulty_config.get("target_activity", "unknown")}
- Your reasoning should identify WHEN this activity occurs based on signal patterns.
- Reference the specific time range in your answer.""",

        "counting": f"""
TASK TYPE: Bout Counting
- Target activity to count: {difficulty_config.get("target_activity", "unknown")}
- Number of bouts inserted: {difficulty_config.get("n_bouts", "unknown")}
- Your reasoning should identify each distinct bout and count them.""",

        "ordering": f"""
TASK TYPE: Temporal Ordering
- Activity A: {difficulty_config.get("activity_A", difficulty_config.get("activity_a", "unknown"))}
- Activity B: {difficulty_config.get("activity_B", difficulty_config.get("activity_b", "unknown"))}
- True temporal order: {difficulty_config.get("true_order", "unknown")}
- Your reasoning should identify both activities and determine which occurred first.""",

        "state_query": f"""
TASK TYPE: State Query (Cross-Scale)
- Local event (needle activity): {difficulty_config.get("needle_activity", "unknown")}
- Global state at event time: {difficulty_config.get("global_activity", difficulty_config.get("global_state_activity", "unknown"))}
- Your reasoning must identify BOTH the local event AND the surrounding activity regime.
- The answer is the GLOBAL state, not the local event.
{format_background_timeline(difficulty_config)}""",

        "antecedent": f"""
TASK TYPE: Temporal Antecedent
- Target activity: {difficulty_config.get("target_activity", "unknown")}
- Antecedent activity (what came before): {difficulty_config.get("antecedent_activity", "unknown")}
- Your reasoning should identify the target activity and what immediately preceded it.""",

        "comparison": f"""
TASK TYPE: Comparison (Extremum Finding)
- Target activity: {difficulty_config.get("target_activity", "unknown")}
- Query type: Find the {difficulty_config.get("extremum", "unknown")} period {difficulty_config.get("polarity", "unknown")} this activity
- All bout periods: {difficulty_config.get("all_periods", "see needles above")}
- Your reasoning should compare all relevant periods and identify the extremum.""",

        "multi_hop": f"""
TASK TYPE: Multi-Hop Localization
- Anchor activity: {difficulty_config.get("anchor_activity", "unknown")}
- Target activity: {difficulty_config.get("target_activity", "unknown")}
- K (which occurrence): {difficulty_config.get("K", difficulty_config.get("k", "unknown"))}
- Direction: {difficulty_config.get("direction", "unknown")} the anchor
- Your reasoning must: (1) locate the anchor activity, (2) find the Kth target activity in the specified direction.""",

        "anomaly_detection": f"""
TASK TYPE: Anomaly Detection (Contextual Reasoning)
- Background regime: {difficulty_config.get("background_regime", "unknown")}
- Is anomaly present: {difficulty_config.get("is_positive", "unknown")}
- Anomaly activity (if positive): {difficulty_config.get("anomaly_activity", "N/A")}
- Your reasoning should:
  1. First characterize the dominant pattern in the recording (sedentary or active)
  2. Identify any activity that contrasts with this background pattern
  3. Explain WHY it is (or isn't) anomalous based on regime mismatch
- Key insight: An activity is anomalous if it belongs to a DIFFERENT regime than the background
  (e.g., running in a sedentary background, or sleeping in an active background).""",

        "anomaly_localization": f"""
TASK TYPE: Anomaly Localization (Detection + Temporal)
- Background regime: {difficulty_config.get("background_regime", "unknown")}
- Is anomaly present: {difficulty_config.get("is_positive", "unknown")}
- Anomaly activity (if positive): {difficulty_config.get("anomaly_activity", "N/A")}
- Anomaly time range (if positive): {difficulty_config.get("anomaly_start", "N/A")} to {difficulty_config.get("anomaly_end", "N/A")}
- Your reasoning should:
  1. Characterize the dominant pattern (sedentary or active regime)
  2. Identify any cross-regime activity
  3. Specify the EXACT time range of the anomaly
  4. Explain why this activity is anomalous in context
- Key insight: Report both WHAT the anomaly is AND WHEN it occurs.""",
    }

    return contexts.get(task_type, f"TASK TYPE: {task_type}\n- Analyze the data and answer the question.")


def create_cot_prompt(sample: Dict) -> str:
    """
    Create the full CoT generation prompt for a TS-Haystack sample.

    This prompt provides the LLM with:
    1. Recording context (time span)
    2. Ground truth needle positions and activities
    3. Task-specific guidance
    4. The question and expected answer

    The LLM should generate reasoning that "discovers" the patterns,
    referencing the actual timestamps and activities.

    Args:
        sample: Dict containing:
               - recording_time_start: str
               - recording_time_end: str
               - context_length_samples: int
               - task_type: str
               - question: str
               - answer: str
               - needles: str (JSON) or List[Dict]
               - difficulty_config: str (JSON) or Dict

    Returns:
        Complete prompt string for the LLM
    """
    # Parse JSON fields if needed
    needles = sample.get("needles", "[]")
    if isinstance(needles, str):
        needles = json.loads(needles)

    difficulty_config = sample.get("difficulty_config", "{}")
    if isinstance(difficulty_config, str):
        difficulty_config = json.loads(difficulty_config)

    # Get time range
    time_start = sample.get("recording_time_start", "unknown")
    time_end = sample.get("recording_time_end", "unknown")
    context_samples = sample.get("context_length_samples", 0)

    # Calculate duration
    if context_samples > 0:
        duration_seconds = context_samples / 100  # Assuming 100Hz
        if duration_seconds >= 60:
            duration_str = f"{duration_seconds / 60:.1f} minutes"
        else:
            duration_str = f"{duration_seconds:.0f} seconds"
    else:
        duration_str = "unknown duration"

    # Get task type
    task_type = sample.get("task_type", "unknown")

    # Build prompt
    prompt = f"""You are analyzing accelerometer data (X, Y, Z axes) from a wrist-worn sensor.

RECORDING CONTEXT:
- Time span: {time_start} to {time_end}
- Duration: {duration_str}
- Samples: {context_samples:,}

{format_needle_metadata(needles)}

{get_task_context(task_type, difficulty_config)}

QUESTION: {sample.get("question", "")}
CORRECT ANSWER: {sample.get("answer", "")}

YOUR TASK:
Write a step-by-step reasoning that:
1. Describes what you observe in the accelerometer patterns
2. Identifies relevant activity bouts using the timestamps provided above
3. Explains how you arrive at the answer

IMPORTANT GUIDELINES:
- Write your rationale as a natural, flowing paragraph (no bullet points or numbered steps)
- Reference specific timestamps and signal characteristics (variance, amplitude, periodicity)
- Do NOT refer to any plot, figure, image, colors, highlighted regions, or visual elements - reason purely about the time-series data patterns
- Do NOT mention that you were given ground truth metadata - reason as if discovering the patterns yourself
- Do NOT include "Answer:" in your rationale - the answer will be appended separately
- Keep your rationale concise: approximately 100-150 words

Rationale:"""

    return prompt


def create_cot_prompt_minimal(sample: Dict) -> str:
    """
    Create a minimal CoT prompt without revealing ground truth.

    This version is useful for testing whether the model can actually
    identify patterns without being given the answers.

    Args:
        sample: Dict containing sample data

    Returns:
        Minimal prompt string
    """
    time_start = sample.get("recording_time_start", "unknown")
    time_end = sample.get("recording_time_end", "unknown")

    return f"""You are analyzing accelerometer data (X, Y, Z axes) from a wrist-worn sensor.

RECORDING CONTEXT:
- Time span: {time_start} to {time_end}

QUESTION: {sample.get("question", "")}

Analyze the accelerometer patterns and answer the question. Write your reasoning as a natural paragraph (100-150 words), focusing on signal characteristics like variance, amplitude, and periodicity. Do not refer to any plots or visual elements.

Rationale:"""


if __name__ == "__main__":
    print("=" * 60)
    print("Prompt Builder Test")
    print("=" * 60)

    # Create test sample
    test_sample = {
        "recording_time_start": "6:00 AM",
        "recording_time_end": "7:40 AM",
        "context_length_samples": 10000,
        "task_type": "counting",
        "question": "How many walking bouts occurred in this recording?",
        "answer": "3",
        "needles": json.dumps([
            {
                "activity": "walking",
                "timestamp_start": "6:12 AM",
                "timestamp_end": "6:18 AM",
                "insert_position_frac": 0.12,
                "duration_ms": 6000,
            },
            {
                "activity": "walking",
                "timestamp_start": "6:35 AM",
                "timestamp_end": "6:42 AM",
                "insert_position_frac": 0.35,
                "duration_ms": 7000,
            },
            {
                "activity": "walking",
                "timestamp_start": "7:15 AM",
                "timestamp_end": "7:22 AM",
                "insert_position_frac": 0.75,
                "duration_ms": 7000,
            },
        ]),
        "difficulty_config": json.dumps({
            "target_activity": "walking",
            "n_bouts": 3,
            "min_gap_samples": 100,
        }),
    }

    print("\n--- Full CoT Prompt ---")
    prompt = create_cot_prompt(test_sample)
    print(prompt)

    print("\n" + "=" * 60)
    print("\n--- Minimal Prompt ---")
    minimal_prompt = create_cot_prompt_minimal(test_sample)
    print(minimal_prompt)

    print("\n" + "=" * 60)
    print("Prompt builder test complete!")
    print("=" * 60)
