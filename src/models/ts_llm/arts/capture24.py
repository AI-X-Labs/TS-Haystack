#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark LLMs with classifier tool on TS-Haystack.

Measures how well a reasoning LLM performs on TS-Haystack when given a
classifier tool (oracle or real HAR classifier) and must discover activities
through agentic tool-calling rather than direct time series perception.

Supports two API backends via a provider adapter pattern:
    - OpenAI (Responses API with tool calling)
    - Amazon Bedrock (Converse API with tool use)

Stores full thinking trajectories and tool calls for analysis.

Classifier modes:
    - oracle: Uses ground-truth labels from sample metadata (no GPU needed).
    - real:   Uses a trained HARClassifier on the raw time series (needs GPU).

Prerequisites:
    - TS-Haystack CoT datasets must be generated (data/capture24/ts_haystack/cot/)
    - OpenAI: OPENAI_API_KEY environment variable must be set
    - Bedrock: AWS credentials configured (env vars, profile, or IAM role)
    - For real mode: a trained classifier checkpoint

Usage:
    # OpenAI (auto-detected from model name)
    python3 -m src.models.ts_llm.arts.capture24 \
      --context-lengths 100 --tasks existence --max-samples 3 --max-workers 1

    # Bedrock with Claude (auto-detected from model name)
    python3 -m src.models.ts_llm.arts.capture24 \
      --model anthropic.claude-opus-4-6-v1 \
      --context-lengths 100 --tasks existence --max-samples 3 --max-workers 1

    # Explicit provider selection
    python3 -m src.models.ts_llm.arts.capture24 \
      --provider bedrock --model anthropic.claude-opus-4-6-v1 \
      --aws-region us-west-2 --thinking-budget 32768 \
      --context-lengths 100 --tasks all --max-workers 4
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (thread-safe)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.datasets.capture24.evaluation import WILLETTS_SPECIFIC_2018_LABELS
from src.datasets.constants import RAW_DATA
from src.datasets.capture24_haystack.utils import format_context_dir
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.models.ts_llm.arts.bout_utils import (
    OracleClassifier,
    bout_timestamps_to_sample_indices,
    format_chunk_results,
)
from src.models.ts_llm.arts.providers import (
    AnthropicProvider,
    ModelResponse,
    OpenAIProvider,
    ToolCallResponse,
)
from src.models.ts_llm.arts.query import (
    get_max_query_seconds,
    get_tool_rounds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path(RAW_DATA) / "capture24" / "ts_haystack" / "cot"
DEFAULT_OUTPUT_DIR = Path("results") / "gpt_benchmark"

ALL_TASKS = [
    "existence", "localization", "counting", "ordering",
    "state_query", "antecedent", "comparison", "multi_hop",
    "anomaly_detection", "anomaly_localization",
]
ALL_SPLITS = ["train", "val", "test"]

ACTIVITY_CLASSES = ", ".join(WILLETTS_SPECIFIC_2018_LABELS)

# Accelerometer sampling rates
SOURCE_HZ = 100   # Raw data Hz
CLASSIFIER_HZ = 30  # HAR classifier expects 30 Hz

# ---------------------------------------------------------------------------
# Prompts — mode-dependent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are given accelerometer data in all three dimensions from a wrist-worn sensor."""

# Shared instruction tail (answer format + activity classes)
_INSTRUCTIONS_TAIL = f"""\
- After receiving classifier results, continue reasoning and query more bouts if needed.
- When ready, provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting questions answer with just the number. \
For time-based answers provide a range: "H:MM:SS.mmm AM/PM to H:MM:SS.mmm AM/PM".

Valid activity classes: {ACTIVITY_CLASSES}"""

# Oracle: confidence = overlap fraction, multiple activities possible
_INSTRUCTIONS_ORACLE = f"""\
Instructions:
- Analyze the accelerometer data carefully.
- Use the `classify_bout` tool to classify time bouts by providing start and end timestamps.
- You will receive [Classifier results] with activity labels for each queried bout.
- The confidence score represents the fraction of the queried segment containing that activity. \
Any activity with confidence > 0 is present in the segment, you can deep dive into those to better locate the segments.
{_INSTRUCTIONS_TAIL}"""

# Real classifier: confidence = softmax probability, one activity per 10s chunk
_INSTRUCTIONS_REAL = f"""\
Instructions:
- Analyze the accelerometer data carefully.
- Use the `classify_bout` tool to classify time bouts by providing start and end timestamps.
- You will receive [Classifier results] with activity labels for each queried bout.
- Bouts longer than 10 seconds are split into non-overlapping 10-second chunks, \
each classified independently with the top predicted activity and its confidence (softmax probability).
- The classifier may make mistakes — cross-check results with the signal patterns you observe.
{_INSTRUCTIONS_TAIL}"""

EXISTENCE_INSTRUCTIONS = """
Decision rule for existence (yes/no) questions:
- If the classifier returns the target activity in ANY segment with confidence >= 0.35, answer "Yes".
- A single positive detection at moderate confidence is sufficient evidence for existence."""

ANOMALY_INSTRUCTIONS = """
Activity regimes (for anomaly reasoning):
  Sedentary: sleep, sitting, standing, vehicle
  Active: walking, mixed-activity, bicycling, manual-work, sports, household-chores
An anomaly is a cross-regime activity (e.g. sedentary in active background or vice versa). \
Activities within the same regime are NOT anomalous."""


def _get_strategy_hint(recording_duration_sec: float) -> str:
    """Return search strategy guidance scaled to the recording length."""
    max_q = get_max_query_seconds(recording_duration_sec)
    if recording_duration_sec <= 15:
        return (
            f"- This is a short recording ({recording_duration_sec:.3f}s). "
            f"You can query up to {max_q:.3f}s per call — a few queries should cover the full recording."
        )
    elif recording_duration_sec <= 300:
        return (
            f"- Recording is {recording_duration_sec:.0f}s. Max query length is {max_q:.0f}s. "
            f"Start with a few broad queries, then narrow down to regions of interest."
        )
    else:
        mins = recording_duration_sec / 60
        return (
            f"- This is a long recording ({mins:.0f} minutes). Max query length is {max_q:.0f}s. "
            f"Use a divide-and-conquer strategy: start with broad sweeps across the recording, "
            f"then narrow down to specific regions. Be systematic — you cannot cover the entire recording."
        )


def get_instructions(classifier_mode: str, recording_duration_sec: float = 100.0) -> str:
    """Return the appropriate instruction block for the classifier mode."""
    base = _INSTRUCTIONS_ORACLE if classifier_mode == "oracle" else _INSTRUCTIONS_REAL
    strategy = _get_strategy_hint(recording_duration_sec)
    return f"{base}\n{strategy}"


# ---------------------------------------------------------------------------
# Tool definition — mode-dependent description
# ---------------------------------------------------------------------------

_TOOL_DESC_ORACLE = (
    "Classify the human activity in a time segment of the recording. "
    "Returns detected activities with their coverage fraction (0.0-1.0) = "
    "the proportion of the queried segment containing that activity. "
    "A coverage of 0.3 means that activity occupies 30% of the queried window — "
    "To get higher coverage for a detected activity, query a tighter time window "
    "around it. Times must be within the recording window."
)

_TOOL_DESC_REAL = (
    "Classify the human activity in a time segment of the recording. "
    "Bouts longer than 10s are split into non-overlapping 10-second chunks, "
    "each classified independently. Returns the top predicted activity per chunk "
    "with its confidence (softmax probability, 0.0-1.0). "
    "Higher confidence means the classifier is more certain. "
    "Times must be within the recording window."
)


def get_classify_bout_tool(classifier_mode: str) -> dict:
    """Return the tool definition with mode-appropriate description."""
    desc = _TOOL_DESC_ORACLE if classifier_mode == "oracle" else _TOOL_DESC_REAL
    return {
        "type": "function",
        "name": "classify_bout",
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "Segment start time in 'H:MM:SS.mmm AM/PM' format (e.g., '9:35:20.000 AM')",
                },
                "end_time": {
                    "type": "string",
                    "description": "Segment end time in 'H:MM:SS.mmm AM/PM' format (e.g., '9:35:30.000 AM')",
                },
            },
            "required": ["start_time", "end_time"],
            "additionalProperties": False,
        },
        "strict": True,
    }


# ---------------------------------------------------------------------------
# Real classifier — global instance with thread-safe access
# ---------------------------------------------------------------------------

_classifier_lock = Lock()
_har_classifier = None  # HARClassifier | None


def load_real_classifier(checkpoint_path: str, device: str = "cuda"):
    """Load a trained HARClassifier for real-mode classification."""
    global _har_classifier
    from src.models.classifiers.capture24.model import HARClassifier
    print(f"Loading HAR classifier from {checkpoint_path} on {device}...")
    _har_classifier = HARClassifier.load(checkpoint_path, device=device)
    print(f"  Encoder type: {_har_classifier.encoder_type}, "
          f"feature dim: {_har_classifier.feature_dim}, "
          f"classes: {_har_classifier.num_classes}")


def _classify_bout_real(
    time_series: np.ndarray,
    start_time: str,
    end_time: str,
    recording_start: str,
    recording_end: str,
    context_length: int,
) -> str:
    """Classify a bout using the real HAR classifier."""
    import torch

    try:
        s_idx, e_idx = bout_timestamps_to_sample_indices(
            start_time, end_time, recording_start, recording_end, context_length,
        )
    except Exception as e:
        return json.dumps({"error": f"Timestamp conversion failed: {e}",
                           "segment": f"{start_time} - {end_time}"})

    bout_data = time_series[:, s_idx:e_idx]
    if bout_data.shape[1] == 0:
        return json.dumps({"error": "Empty segment",
                           "segment": f"{start_time} - {end_time}"})

    stride = SOURCE_HZ // CLASSIFIER_HZ
    bout_30hz = bout_data[:, ::stride]

    bout_tensor = torch.tensor(bout_30hz, dtype=torch.float32)
    with _classifier_lock:
        bout_tensor = bout_tensor.to(next(_har_classifier.parameters()).device)
        chunk_results = _har_classifier.classify_bout(bout_tensor)

    results = format_chunk_results(start_time, end_time, chunk_results)
    if len(results) > 1:
        classifications = [
            {"activity": r["label"], "confidence": round(r["confidence"], 2),
             "chunk": f"{r['start']} - {r['end']}"}
            for r in results
        ]
    else:
        classifications = [
            {"activity": r["label"], "confidence": round(r["confidence"], 2)}
            for r in results
        ]

    return json.dumps({
        "segment": f"{start_time} - {end_time}",
        "classifications": classifications,
    })


# ---------------------------------------------------------------------------
# Time series image rendering
# ---------------------------------------------------------------------------


def render_ts_image(sample: dict) -> str | None:
    """Render 3-axis accelerometer data as a base64-encoded PNG.

    Returns base64 string or None if data is missing.
    """
    x = sample.get("x_axis")
    y = sample.get("y_axis")
    z = sample.get("z_axis")
    if x is None or y is None or z is None:
        return None

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)

    n_samples = len(x)
    if n_samples == 0:
        return None

    from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string
    start_dt = parse_time_string(sample["recording_time_start"])
    end_dt = parse_time_string(sample["recording_time_end"])
    from datetime import timedelta
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    total_sec = (end_dt - start_dt).total_seconds()

    max_points = 2000
    if n_samples > max_points:
        step = n_samples // max_points
        x, y, z = x[::step], y[::step], z[::step]
        n_plot = len(x)
    else:
        n_plot = n_samples

    t = [start_dt + timedelta(seconds=i * total_sec / (n_plot - 1)) for i in range(n_plot)]

    fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
    ax.plot(t, x, color="#1f77b4", linewidth=0.5, alpha=0.8, label="X")
    ax.plot(t, y, color="#ff7f0e", linewidth=0.5, alpha=0.8, label="Y")
    ax.plot(t, z, color="#2ca02c", linewidth=0.5, alpha=0.8, label="Z")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M:%S %p"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Acceleration (g)", fontsize=8)
    ax.set_title(
        f"Accelerometer: {sample['recording_time_start']} — {sample['recording_time_end']}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")




# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------




def execute_classify_bout(
    start_time: str,
    end_time: str,
    recording_duration_sec: float,
    *,
    classifier_mode: str,
    oracle: OracleClassifier | None = None,
    time_series: np.ndarray | None = None,
    recording_start: str = "",
    recording_end: str = "",
    context_length: int = 0,
) -> str:
    """Execute classify_bout tool call and return JSON result string."""
    try:
        from datetime import timedelta

        from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string

        start_dt = parse_time_string(start_time)
        end_dt = parse_time_string(end_time)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        query_sec = (end_dt - start_dt).total_seconds()
        max_sec = get_max_query_seconds(recording_duration_sec)

        if query_sec > max_sec:
            return json.dumps({
                "error": (
                    f"Segment too long: {query_sec:.3f}s exceeds the maximum of "
                    f"{max_sec:.3f}s. Query segments of {max_sec:.3f}s or shorter."
                ),
                "segment": f"{start_time} - {end_time}",
            })

        if classifier_mode == "oracle":
            assert oracle is not None, "Oracle classifier required in oracle mode"
            results = oracle.classify_bout(start_time, end_time)
            classifications = [
                {"activity": label, "confidence": round(conf, 2)}
                for label, conf in results
            ]
            return json.dumps({
                "segment": f"{start_time} - {end_time}",
                "classifications": classifications,
            })
        else:
            assert time_series is not None, "Time series required in real mode"
            assert _har_classifier is not None, "HAR classifier not loaded"
            return _classify_bout_real(
                time_series, start_time, end_time,
                recording_start, recording_end, context_length,
            )
    except Exception as e:
        return json.dumps({"error": str(e), "segment": f"{start_time} - {end_time}"})


# ---------------------------------------------------------------------------
# User message construction
# ---------------------------------------------------------------------------


def build_user_content(
    sample: dict,
    classifier_mode: str,
    recording_duration_sec: float,
    include_image: bool = True,
) -> tuple[str, str, str | None]:
    """Build user message components.

    Returns (pre_prompt, post_prompt, img_b64) tuple.
    img_b64 is None if image is disabled or unavailable.
    """
    task_type = sample.get("task_type", "")

    pre_prompt = (
        f"You are given accelerometer data in all three dimensions from a "
        f"wrist-worn sensor. The recording spans from "
        f"{sample['recording_time_start']} to {sample['recording_time_end']}.\n\n"
        f"Question: {sample['question']}"
    )

    post_prompt = get_instructions(classifier_mode, recording_duration_sec)
    if task_type == "existence":
        post_prompt += EXISTENCE_INSTRUCTIONS
    if task_type in ("anomaly_detection", "anomaly_localization"):
        post_prompt += ANOMALY_INSTRUCTIONS

    img_b64 = None
    if include_image:
        img_b64 = render_ts_image(sample)

    return pre_prompt, post_prompt, img_b64


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def run_tool_call_sample(
    sample: dict,
    provider: OpenAIProvider | AnthropicProvider,
    max_tool_rounds: int,
    classifier_mode: str,
    include_image: bool = True,
    reasoning_effort: str = "high",
) -> dict:
    """Run agentic tool-calling loop for one sample.

    Returns dict with: final_text, tool_call_count, rounds, reasoning_summaries,
    messages (serialized conversation).
    """
    # --- Build classifier context ---
    oracle = None
    time_series = None
    recording_start = sample["recording_time_start"]
    recording_end = sample["recording_time_end"]
    context_length = sample.get("context_length_samples", 0)

    if classifier_mode == "oracle":
        oracle = OracleClassifier(
            needles=sample["needles"],
            difficulty_config=sample["difficulty_config"],
            recording_time_start=recording_start,
            recording_time_end=recording_end,
            context_length_samples=context_length,
        )
        recording_duration_sec = oracle._total_seconds
    else:
        x = np.asarray(sample["x_axis"], dtype=np.float32)
        y = np.asarray(sample["y_axis"], dtype=np.float32)
        z = np.asarray(sample["z_axis"], dtype=np.float32)
        time_series = np.stack([x, y, z], axis=0)
        if context_length == 0:
            context_length = time_series.shape[1]

        from datetime import timedelta

        from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string
        start_dt = parse_time_string(recording_start)
        end_dt = parse_time_string(recording_end)
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        recording_duration_sec = (end_dt - start_dt).total_seconds()

    # Build provider-native messages
    pre_prompt, post_prompt, img_b64 = build_user_content(
        sample, classifier_mode, recording_duration_sec, include_image=include_image,
    )
    messages = provider.build_messages(SYSTEM_PROMPT, pre_prompt, post_prompt, img_b64)
    tools = provider.get_tool_config(get_classify_bout_tool(classifier_mode))

    tool_call_count = 0
    tool_calls_success = 0
    tool_calls_error = 0
    total_segments_classified = 0
    total_seconds_classified = 0.0
    rounds = 0
    reasoning_summaries: list[str] = []
    # Track the full conversation for logging (text only — no base64 images)
    conversation_log: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "[image + text]" if img_b64 else f"{pre_prompt}\n{post_prompt}"},
    ]

    final_text_parts: list[str] = []

    for round_num in range(max_tool_rounds + 1):
        rounds = round_num + 1

        try:
            response = provider.call(messages, tools, reasoning_effort, max_tokens=16384)
        except Exception as e:
            return {
                "final_text": f"API error: {e}",
                "tool_call_count": tool_call_count,
                "tool_calls_success": tool_calls_success,
                "tool_calls_error": tool_calls_error,
                "total_segments_classified": total_segments_classified,
                "total_seconds_classified": round(total_seconds_classified, 1),
                "rounds": rounds,
                "reasoning_summaries": reasoning_summaries,
                "messages": conversation_log,
            }

        # Log reasoning summaries
        for summary in response.reasoning_summaries:
            reasoning_summaries.append(summary)
            conversation_log.append({"role": "reasoning_summary", "content": summary})

        # Log text
        for text in response.text_parts:
            final_text_parts.append(text)
        if response.text_parts:
            conversation_log.append({"role": "assistant", "content": "\n".join(response.text_parts)})

        # Append assistant turn to provider-native messages
        provider.append_response(messages, response)

        # Process tool calls
        has_tool_calls = bool(response.tool_calls)
        for tc in response.tool_calls:
            tool_call_count += 1

            start_time = tc.arguments.get("start_time", "")
            end_time = tc.arguments.get("end_time", "")

            tool_result = execute_classify_bout(
                start_time, end_time, recording_duration_sec,
                classifier_mode=classifier_mode,
                oracle=oracle,
                time_series=time_series,
                recording_start=recording_start,
                recording_end=recording_end,
                context_length=context_length,
            )

            # Track classifier call stats
            try:
                result_data = json.loads(tool_result)
                if "error" in result_data:
                    tool_calls_error += 1
                else:
                    tool_calls_success += 1
                    total_segments_classified += len(result_data.get("classifications", []))
                    from datetime import timedelta

                    from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string
                    _s = parse_time_string(start_time)
                    _e = parse_time_string(end_time)
                    if _e < _s:
                        _e += timedelta(days=1)
                    total_seconds_classified += (_e - _s).total_seconds()
            except (json.JSONDecodeError, AttributeError, Exception):
                tool_calls_error += 1

            conversation_log.append({
                "role": "tool_call",
                "name": "classify_bout",
                "arguments": tc.arguments,
                "call_id": tc.call_id,
            })
            conversation_log.append({
                "role": "tool_result",
                "call_id": tc.call_id,
                "content": tool_result,
            })

            provider.append_tool_result(messages, tc.call_id, tool_result)

        if not has_tool_calls:
            final_text = "\n".join(final_text_parts)
            return {
                "final_text": final_text,
                "tool_call_count": tool_call_count,
                "tool_calls_success": tool_calls_success,
                "tool_calls_error": tool_calls_error,
                "total_segments_classified": total_segments_classified,
                "total_seconds_classified": round(total_seconds_classified, 1),
                "rounds": rounds,
                "reasoning_summaries": reasoning_summaries,
                "messages": conversation_log,
            }

    # Exhausted tool rounds — force a final answer by calling without tools
    final_text = "\n".join(final_text_parts) if final_text_parts else ""
    if not final_text or "Answer:" not in final_text:
        forced_prompt = (
            "You have used all available tool calls. Based on the classifier results "
            "you have received so far, provide your final answer now. "
            "Answer with the format specified in the instructions."
        )
        provider.append_user_text(messages, forced_prompt)
        conversation_log.append({"role": "user", "content": "[forced answer prompt]"})

        try:
            forced = provider.call(messages, [], reasoning_effort, max_tokens=16384)
            if forced.text_parts:
                final_text = "\n".join(forced.text_parts)
                conversation_log.append({"role": "assistant", "content": final_text})
        except Exception:
            pass  # Keep whatever text we had

    return {
        "final_text": final_text or "Max tool rounds exceeded",
        "tool_call_count": tool_call_count,
        "tool_calls_success": tool_calls_success,
        "tool_calls_error": tool_calls_error,
        "total_segments_classified": total_segments_classified,
        "total_seconds_classified": round(total_seconds_classified, 1),
        "rounds": rounds,
        "reasoning_summaries": reasoning_summaries,
        "messages": conversation_log,
    }


# ---------------------------------------------------------------------------
# Sample processing
# ---------------------------------------------------------------------------


def process_sample(
    sample: dict,
    idx: int,
    provider: OpenAIProvider | AnthropicProvider,
    max_tool_rounds: int,
    classifier_mode: str,
    include_image: bool = True,
    reasoning_effort: str = "high",
) -> dict:
    """Process a single sample: run agentic loop, extract answer, evaluate."""
    result = run_tool_call_sample(
        sample, provider, max_tool_rounds,
        classifier_mode=classifier_mode,
        include_image=include_image, reasoning_effort=reasoning_effort,
    )

    final_text = result["final_text"]
    answer_type = sample.get("answer_type", "category")
    predicted_answer = extract_final_answer(final_text, answer_type)

    ground_truth = sample["answer"]
    eval_result = evaluate_answer(ground_truth, predicted_answer, answer_type)

    return {
        "index": idx,
        "question": sample["question"],
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "final_text": final_text,
        "correct": eval_result["correct"],
        "iou": eval_result["iou"],
        "normalized_gt": eval_result["normalized_gt"],
        "normalized_pred": eval_result["normalized_pred"],
        "tool_call_count": result["tool_call_count"],
        "tool_calls_success": result["tool_calls_success"],
        "tool_calls_error": result["tool_calls_error"],
        "total_segments_classified": result["total_segments_classified"],
        "total_seconds_classified": result["total_seconds_classified"],
        "rounds": result["rounds"],
        "reasoning_summaries": result.get("reasoning_summaries", []),
        "messages": result["messages"],
        "task_type": sample.get("task_type", ""),
        "answer_type": answer_type,
        "context_length_samples": sample.get("context_length_samples", 0),
        "classifier_mode": classifier_mode,
    }


# ---------------------------------------------------------------------------
# Parquet file processing with resume support
# ---------------------------------------------------------------------------


def load_completed_indices(jsonl_path: Path) -> set[int]:
    """Load indices of already-completed samples from JSONL file."""
    completed = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(record["index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def process_parquet_file(
    parquet_path: Path,
    output_path: Path,
    provider: OpenAIProvider | AnthropicProvider,
    max_samples: int | None,
    max_workers: int,
    max_tool_rounds: int,
    classifier_mode: str,
    include_image: bool = True,
    reasoning_effort: str = "high",
) -> dict[str, int]:
    """Process a parquet file with parallel API calls and resume support."""
    df = pd.read_parquet(parquet_path)
    if max_samples is not None:
        df = df.head(max_samples)

    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "trajectories.jsonl"

    completed = load_completed_indices(jsonl_path)
    todo_indices = [i for i in range(len(df)) if i not in completed]

    stats = {"total": len(df), "success": 0, "failed": 0, "skipped": len(completed)}

    if not todo_indices:
        print(f"  Already complete ({len(completed)}/{len(df)}), skipping")
        stats["success"] = len(completed)
        return stats

    if completed:
        print(f"  Resuming: {len(completed)}/{len(df)} already done")

    write_lock = Lock()

    def _process_one(idx: int) -> dict:
        row = df.iloc[idx].to_dict()
        return process_sample(
            row, idx, provider, max_tool_rounds,
            classifier_mode=classifier_mode,
            include_image=include_image, reasoning_effort=reasoning_effort,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, idx): idx for idx in todo_indices}

        for future in tqdm(as_completed(futures), total=len(todo_indices), desc="  evaluating"):
            try:
                result = future.result()
                with write_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                stats["success"] += 1
            except Exception as e:
                idx = futures[future]
                print(f"  Error processing sample {idx}: {e}")
                stats["failed"] += 1

    return stats


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def compute_summary(output_dir: Path) -> dict:
    """Aggregate JSONL results into per-task/context/answer_type metrics."""
    summary: dict = {"by_task": {}, "by_context": {}, "by_answer_type": {}, "overall": {}}

    all_results = []
    for jsonl_path in sorted(output_dir.rglob("trajectories.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not all_results:
        return summary

    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("correct", False))
    avg_tool_calls = sum(r.get("tool_call_count", 0) for r in all_results) / total
    avg_errors = sum(r.get("tool_calls_error", 0) for r in all_results) / total
    avg_segments = sum(r.get("total_segments_classified", 0) for r in all_results) / total
    avg_seconds = sum(r.get("total_seconds_classified", 0) for r in all_results) / total
    summary["overall"] = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "avg_tool_calls": round(avg_tool_calls, 2),
        "avg_tool_errors": round(avg_errors, 2),
        "avg_segments_classified": round(avg_segments, 2),
        "avg_seconds_classified": round(avg_seconds, 1),
    }

    tasks: dict[str, list] = {}
    for r in all_results:
        tasks.setdefault(r.get("task_type", "unknown"), []).append(r)
    for task, results in sorted(tasks.items()):
        n = len(results)
        c = sum(1 for r in results if r.get("correct", False))
        tc = sum(r.get("tool_call_count", 0) for r in results) / n
        te = sum(r.get("tool_calls_error", 0) for r in results) / n
        sc = sum(r.get("total_segments_classified", 0) for r in results) / n
        ss = sum(r.get("total_seconds_classified", 0) for r in results) / n
        summary["by_task"][task] = {
            "total": n, "correct": c, "accuracy": c / n,
            "avg_tool_calls": round(tc, 2), "avg_tool_errors": round(te, 2),
            "avg_segments_classified": round(sc, 2), "avg_seconds_classified": round(ss, 1),
        }

    contexts: dict[int, list] = {}
    for r in all_results:
        contexts.setdefault(r.get("context_length_samples", 0), []).append(r)
    for ctx, results in sorted(contexts.items()):
        n = len(results)
        c = sum(1 for r in results if r.get("correct", False))
        summary["by_context"][ctx] = {"total": n, "correct": c, "accuracy": c / n}

    answer_types: dict[str, list] = {}
    for r in all_results:
        answer_types.setdefault(r.get("answer_type", "unknown"), []).append(r)
    for at, results in sorted(answer_types.items()):
        n = len(results)
        c = sum(1 for r in results if r.get("correct", False))
        ious = [r["iou"] for r in results if r.get("iou") is not None]
        avg_iou = sum(ious) / len(ious) if ious else None
        entry: dict = {"total": n, "correct": c, "accuracy": c / n}
        if avg_iou is not None:
            entry["avg_iou"] = round(avg_iou, 3)
        summary["by_answer_type"][at] = entry

    return summary


def print_summary_table(summary: dict) -> None:
    """Print summary as a formatted terminal table."""
    overall = summary.get("overall", {})
    if not overall:
        print("No results to summarize.")
        return

    print(f"\n{'=' * 70}")
    print("LLM Classifier Benchmark Results")
    print(f"{'=' * 70}")
    print(f"Overall: {overall['correct']}/{overall['total']} correct "
          f"({overall['accuracy'] * 100:.1f}%)")
    print(f"  Tool calls: {overall['avg_tool_calls']} avg "
          f"({overall.get('avg_tool_errors', 0)} errors), "
          f"segments: {overall.get('avg_segments_classified', 'N/A')} avg, "
          f"seconds: {overall.get('avg_seconds_classified', 'N/A')}s avg")

    by_task = summary.get("by_task", {})
    if by_task:
        print(f"\n{'Task':<25} {'Acc':>7} {'Tools':>6} {'Errs':>5} {'Segs':>5} {'Secs':>7}")
        print(f"{'-' * 25} {'-' * 7} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 7}")
        for task, m in sorted(by_task.items()):
            print(f"{task:<25} {m['accuracy'] * 100:>6.1f}% {m['avg_tool_calls']:>6.1f}"
                  f" {m.get('avg_tool_errors', 0):>5.1f}"
                  f" {m.get('avg_segments_classified', 0):>5.1f}"
                  f" {m.get('avg_seconds_classified', 0):>6.1f}s")

    by_at = summary.get("by_answer_type", {})
    if by_at:
        print(f"\n{'Answer Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Avg IoU':>10}")
        print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}")
        for at, m in sorted(by_at.items()):
            iou_str = f"{m['avg_iou']:.3f}" if "avg_iou" in m else "N/A"
            print(f"{at:<20} {m['correct']:>8} {m['total']:>8} "
                  f"{m['accuracy'] * 100:>9.1f}% {iou_str:>10}")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs with classifier tool on TS-Haystack"
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--context-lengths", type=float, nargs="+", default=[100])
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"])
    parser.add_argument("--splits", type=str, nargs="+", default=["test"])
    parser.add_argument("--model", type=str, default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-tool-rounds", type=int, default=None,
        help="Max tool-call rounds per sample. If unset, scales automatically "
             "with recording length (15 for <=300s, up to 30 for 2h).",
    )
    parser.add_argument("--no-image", action="store_true", help="Disable time series image input")
    parser.add_argument(
        "--reasoning-effort", type=str, default="high",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level for the model (default: high)",
    )
    # Classifier mode
    parser.add_argument(
        "--classifier-mode", type=str, default="oracle",
        choices=["oracle", "real"],
        help="Classifier mode: 'oracle' uses ground-truth labels, "
             "'real' uses a trained HAR classifier (default: oracle)",
    )
    parser.add_argument(
        "--classifier-checkpoint", type=str, default=None,
        help="Path to trained HARClassifier checkpoint (required for --classifier-mode real)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for real classifier inference (default: cuda)",
    )
    # Provider selection
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["openai", "anthropic", "bedrock"],
        help="API provider. Auto-detected from model name if not set "
             "(anthropic.*/global.anthropic.* -> bedrock, claude-* -> anthropic, else openai).",
    )
    parser.add_argument(
        "--aws-region", type=str, default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=None,
        help="Override thinking budget tokens for Anthropic/Bedrock (default: derived from --reasoning-effort)",
    )
    args = parser.parse_args()

    # Auto-detect provider from model name
    if args.provider is not None:
        provider_name = args.provider
    elif any(args.model.startswith(p) for p in ("anthropic.", "us.anthropic.", "eu.anthropic.", "global.anthropic.")):
        provider_name = "bedrock"
    elif args.model.startswith("claude-"):
        provider_name = "anthropic"
    else:
        provider_name = "openai"

    # Validate credentials and create provider
    if provider_name == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY not set")
            sys.exit(1)
        provider = OpenAIProvider(model=args.model)
    elif provider_name == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        provider = AnthropicProvider(
            model=args.model,
            thinking_budget=args.thinking_budget,
        )
    else:  # bedrock
        provider = AnthropicProvider(
            model=args.model,
            thinking_budget=args.thinking_budget,
            bedrock=True,
            aws_region=args.aws_region,
        )

    if args.classifier_mode == "real" and args.classifier_checkpoint is None:
        print("Error: --classifier-checkpoint is required for --classifier-mode real")
        sys.exit(1)

    # Load real classifier if needed
    if args.classifier_mode == "real":
        import torch
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"
        load_real_classifier(args.classifier_checkpoint, device=device)

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    input_dir = Path(args.input_dir)
    include_image = not args.no_image
    model_slug = args.model.replace("/", "_")
    # Include ablation flags in output dir for easy comparison
    suffix_parts = [args.classifier_mode]
    if args.classifier_mode == "real" and args.classifier_checkpoint:
        classifier_name = Path(args.classifier_checkpoint).parent.name
        suffix_parts.append(classifier_name)
    if not include_image:
        suffix_parts.append("no_image")
    if args.reasoning_effort != "high":
        suffix_parts.append(f"reason_{args.reasoning_effort}")
    if args.thinking_budget is not None:
        suffix_parts.append(f"budget_{args.thinking_budget}")
    dir_name = model_slug + "_" + "_".join(suffix_parts)
    output_dir = Path(args.output_dir) / dir_name

    print("=" * 60)
    print("LLM Classifier Benchmark")
    print("=" * 60)
    print(f"Provider: {provider_name}")
    print(f"Model: {args.model}")
    print(f"Classifier mode: {args.classifier_mode}")
    if args.classifier_mode == "real":
        print(f"Classifier checkpoint: {args.classifier_checkpoint}")
        print(f"Classifier device: {args.device}")
    if provider_name == "bedrock":
        print(f"AWS Region: {args.aws_region}")
    if args.thinking_budget:
        print(f"Thinking budget: {args.thinking_budget}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tasks:  {tasks}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Workers: {args.max_workers}")
    print(f"Max tool rounds: {args.max_tool_rounds or 'auto'}")
    print(f"Image input: {include_image}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    if args.max_samples:
        print(f"Max samples per file: {args.max_samples}")
    print()

    total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for ctx_seconds in args.context_lengths:
        ctx_dir = format_context_dir(ctx_seconds)
        tool_rounds = get_tool_rounds(ctx_seconds, cli_override=args.max_tool_rounds)

        for task in tasks:
            for split in args.splits:
                parquet_path = input_dir / ctx_dir / task / split / "data.parquet"
                if not parquet_path.exists():
                    continue

                n_rows = len(pd.read_parquet(parquet_path, columns=["question"]))
                effective = min(n_rows, args.max_samples) if args.max_samples else n_rows
                print(f"\n{ctx_dir}/{task}/{split} ({effective} samples, {tool_rounds} tool rounds)")

                result_path = output_dir / ctx_dir / task / split
                stats = process_parquet_file(
                    parquet_path, result_path, provider,
                    args.max_samples, args.max_workers, tool_rounds,
                    classifier_mode=args.classifier_mode,
                    include_image=include_image, reasoning_effort=args.reasoning_effort,
                )

                for k in total_stats:
                    total_stats[k] += stats[k]

                print(f"  success={stats['success']}, failed={stats['failed']}, "
                      f"skipped={stats['skipped']}")

    # Compute and save summary
    summary = compute_summary(output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print_summary_table(summary)

    print(f"\nTotal: processed={total_stats['success']}, "
          f"failed={total_stats['failed']}, skipped={total_stats['skipped']}")


if __name__ == "__main__":
    main()
