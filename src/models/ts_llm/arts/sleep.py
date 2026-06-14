#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark LLMs (OpenAI / Anthropic / Bedrock) on the Sleep PSG haystack with a
``classify_bout`` tool wired to either an oracle (ground-truth annotations) or a
trained :class:`SleepClassifier`.

Shares the orchestration / provider logic with the ``capture24`` module but
replaces the data layer (parquet metadata + memmap signal slicing) and the
classify_bout tool implementation with sleep-PSG-specific code.

Examples:
    # Oracle baseline on sleep_stages
    python3 -m src.models.ts_llm.arts.sleep \\
        --label-class sleep_stages --classifier-mode oracle \\
        --context-lengths 900 --tasks existence --max-samples 5 --max-workers 1

    # Real classifier on arousals
    python3 -m src.models.ts_llm.arts.sleep \\
        --label-class arousals --classifier-mode real \\
        --arousals-checkpoint results/sleep_classifier/arousals/best_classifier.pt \\
        --context-lengths 900 --tasks all --split test

    # Real classifier on sleep_stages (loads both classifiers; arousals needed for state_query)
    python3 -m src.models.ts_llm.arts.sleep \\
        --label-class sleep_stages --classifier-mode real \\
        --stages-checkpoint results/sleep_classifier/sleep_stages/best_classifier.pt \\
        --arousals-checkpoint results/sleep_classifier/arousals/best_classifier.pt \\
        --context-lengths 900 --tasks all --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Reuse provider adapters and the response dataclasses from the main benchmark.
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
from src.datasets.sleep_psg_haystack.loader import (
    CHANNEL_NAMES,
    EFFECTIVE_HZ,
    SOURCE_HZ,
    load_annotations,
    load_window,
)
from src.datasets.sleep_psg_haystack.qa_loader import (
    ALL_CONTEXT_LENGTHS_PER_LABEL,
    ALL_TASKS,
    BASE_DIR,
    REQUIRED_COLUMNS,
    _format_ctx_dir,
)
from src.datasets.capture24_haystack.qa_loader import LazyParquetDataset
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.models.classifiers.sleep.model import (
    AROUSAL_CLASS_NAMES,
    SLEEP_STAGE_CLASS_NAMES,
    SleepClassifier,
)


# ---------------------------------------------------------------------------
# Constants & prompts
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("results") / "gpt_sleep_benchmark"

# Synthetic recording origin so we can use HH:MM:SS-style timestamps and reuse
# parse_time_string from the existing utils. The agent talks in offsets from
# this anchor; we convert back to milliseconds within the parquet window.
RECORDING_ANCHOR = datetime(2000, 1, 1, 0, 0, 0)


def _format_anchor_time(seconds: float) -> str:
    dt = RECORDING_ANCHOR + timedelta(seconds=seconds)
    return dt.strftime("%I:%M:%S.") + f"{dt.microsecond // 1000:03d} {dt.strftime('%p')}"


def _parse_anchor_time(ts: str) -> float:
    """Parse 'H:MM:SS.mmm AM/PM' (relative to RECORDING_ANCHOR) → seconds offset."""
    from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string
    parsed = parse_time_string(ts)
    parsed = parsed.replace(year=RECORDING_ANCHOR.year, month=RECORDING_ANCHOR.month, day=RECORDING_ANCHOR.day)
    delta = (parsed - RECORDING_ANCHOR).total_seconds()
    if delta < 0:
        delta += 24 * 3600
    return delta


SYSTEM_PROMPT = (
    f"You are an expert polysomnography (PSG) analyst. You are given a 13-channel PSG "
    f"recording sampled at {EFFECTIVE_HZ} Hz. Channels (in order): {', '.join(CHANNEL_NAMES)}.\n\n"
    f"Possible sleep stages: {', '.join(SLEEP_STAGE_CLASS_NAMES)}.\n"
    f"Possible arousal/respiratory event types: {', '.join(AROUSAL_CLASS_NAMES)}."
)


_INSTRUCTIONS_TAIL = """\
- After receiving classifier results, continue reasoning and query more bouts if needed.
- When ready, provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting answer with just the number. \
For time-based answers provide a range: "H:MM:SS.mmm AM/PM to H:MM:SS.mmm AM/PM"."""


def get_instructions(classifier_mode: str, recording_duration_sec: float, tool_names: list[str]) -> str:
    tool_list = " and ".join(f"`{n}`" for n in tool_names)
    if classifier_mode == "oracle":
        head = (
            "Instructions:\n"
            f"- Use the {tool_list} tool(s) to classify time bouts by start/end timestamps.\n"
            "- Returned 'confidence' is the fraction of the queried segment containing each class. "
            "Anything > 0 is present.\n"
        )
    else:
        stage_part = (
            f"`classify_sleep_stage` splits bouts into non-overlapping {STAGE_CHUNK_SEC}-second "
            "chunks"
        )
        arousal_part = (
            f"`classify_arousal` splits bouts into non-overlapping {AROUSAL_CHUNK_SEC}-second "
            "chunks"
        )
        if "classify_sleep_stage" in tool_names and "classify_arousal" in tool_names:
            chunk_desc = f"{stage_part}; {arousal_part}."
        elif "classify_sleep_stage" in tool_names:
            chunk_desc = f"{stage_part}."
        else:
            chunk_desc = f"{arousal_part}."
        head = (
            "Instructions:\n"
            f"- Use the {tool_list} tool(s) to classify time bouts by start/end timestamps.\n"
            f"- {chunk_desc} Each chunk is classified independently and returned as a list "
            "entry with its own sub-timestamps, top-class prediction, and softmax confidence.\n"
            "- To see a bout at its native resolution, query exactly one chunk width at a "
            "time; to scan faster, query a longer bout and read the per-chunk list.\n"
            "- The classifier may make mistakes — cross-check predictions against the question.\n"
        )
    max_q = get_max_query_seconds(recording_duration_sec)
    strategy = (
        f"- Recording duration: {recording_duration_sec:.0f}s. Max query length: {max_q:.0f}s.\n"
    )
    return f"{head}{strategy}{_INSTRUCTIONS_TAIL}"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Tasks that require both classifiers simultaneously (cross-timeline queries).
BOTH_TOOLS_TASKS: set[tuple[str, str]] = {("sleep_stages", "state_query")}

_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "start_time": {
            "type": "string",
            "description": "Segment start in 'H:MM:SS.mmm AM/PM' format",
        },
        "end_time": {
            "type": "string",
            "description": "Segment end in 'H:MM:SS.mmm AM/PM' format",
        },
    },
    "required": ["start_time", "end_time"],
    "additionalProperties": False,
}


# Native classifier window sizes (seconds) — bouts are split into non-overlapping
# chunks of this size in real mode.
STAGE_CHUNK_SEC = 30
AROUSAL_CHUNK_SEC = 20


def get_sleep_stage_tool(classifier_mode: str) -> dict:
    if classifier_mode == "oracle":
        desc = (
            "Classify a time segment of the PSG recording for sleep stages. "
            "Returns each sleep stage class with the fraction of the segment it occupies (0.0-1.0)."
        )
    else:
        desc = (
            "Classify a time segment of the PSG recording for sleep stages. The segment is "
            f"split into non-overlapping {STAGE_CHUNK_SEC}-second chunks, each classified "
            "independently with a top sleep-stage prediction and softmax confidence. "
            "Returns a list of chunks with their sub-timestamps."
        )
    return {"type": "function", "name": "classify_sleep_stage", "description": desc,
            "parameters": _TOOL_PARAMS, "strict": True}


def get_arousal_tool(classifier_mode: str) -> dict:
    if classifier_mode == "oracle":
        desc = (
            "Classify a time segment of the PSG recording for arousal/respiratory events. "
            "Returns each arousal class with the fraction of the segment it occupies (0.0-1.0)."
        )
    else:
        desc = (
            "Classify a time segment of the PSG recording for arousal/respiratory events. The "
            f"segment is split into non-overlapping {AROUSAL_CHUNK_SEC}-second chunks, each "
            "classified independently with a top arousal-class prediction and softmax "
            "confidence. Returns a list of chunks with their sub-timestamps."
        )
    return {"type": "function", "name": "classify_arousal", "description": desc,
            "parameters": _TOOL_PARAMS, "strict": True}


def tools_for_sample(label_class: str, task_type: str, classifier_mode: str) -> list[dict]:
    """Return the tool list the agent should receive for this sample."""
    if (label_class, task_type) in BOTH_TOOLS_TASKS:
        return [get_sleep_stage_tool(classifier_mode), get_arousal_tool(classifier_mode)]
    if label_class == "sleep_stages":
        return [get_sleep_stage_tool(classifier_mode)]
    return [get_arousal_tool(classifier_mode)]


# ---------------------------------------------------------------------------
# Real classifiers — one per label class, thread-safe access
# ---------------------------------------------------------------------------

_classifier_lock = Lock()
_stages_classifier: Optional[SleepClassifier] = None
_arousals_classifier: Optional[SleepClassifier] = None


def load_real_classifiers(
    stages_checkpoint: Optional[str],
    arousals_checkpoint: Optional[str],
    device: str = "cuda",
) -> None:
    global _stages_classifier, _arousals_classifier
    for ckpt, attr, name in [
        (stages_checkpoint, "_stages_classifier", "sleep_stages"),
        (arousals_checkpoint, "_arousals_classifier", "arousals"),
    ]:
        if ckpt is None:
            continue
        print(f"Loading {name} SleepClassifier from {ckpt} on {device}...")
        clf = SleepClassifier.load(ckpt, device=device)
        print(f"  Classes: {clf.class_names} | window_samples: {clf.window_samples}")
        globals()[attr] = clf


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _classify_bout_oracle(
    sub_start_sec: float,
    sub_end_sec: float,
    window_start_ms: int,
    annotations: list[tuple[int, int, str]],
    class_names: list[str],
) -> dict:
    """Compute per-class overlap fractions in the queried sub-range.

    ``annotations`` are at SOURCE_HZ samples — we convert them to ms relative
    to the parquet window for comparison.
    """
    q_start_ms = window_start_ms + int(round(sub_start_sec * 1000))
    q_end_ms = window_start_ms + int(round(sub_end_sec * 1000))
    span_ms = max(1, q_end_ms - q_start_ms)

    overlap = {c: 0 for c in class_names}
    for s_native, e_native, label in annotations:
        if label not in overlap:
            continue
        s_ms = int(round(s_native * 1000 / SOURCE_HZ))
        e_ms = int(round(e_native * 1000 / SOURCE_HZ))
        lo = max(q_start_ms, s_ms)
        hi = min(q_end_ms, e_ms)
        if hi > lo:
            overlap[label] += hi - lo

    classifications = [
        {"activity": c, "confidence": round(overlap[c] / span_ms, 3)}
        for c in class_names
        if overlap[c] > 0
    ]
    return {
        "segment": f"{_format_anchor_time(sub_start_sec)} - {_format_anchor_time(sub_end_sec)}",
        "classifications": classifications,
    }


def _classify_bout_real(
    sub_start_sec: float,
    sub_end_sec: float,
    signal: np.ndarray,  # (13, L) at EFFECTIVE_HZ — already loaded for the parquet window
    classifier: "SleepClassifier",
) -> dict:
    """Split the queried bout into non-overlapping chunks of the classifier's
    native window size and classify each independently. Returns one entry per
    chunk with its own sub-segment timestamps."""
    import torch

    L = signal.shape[1]
    s_idx = max(0, int(round(sub_start_sec * EFFECTIVE_HZ)))
    e_idx = min(L, int(round(sub_end_sec * EFFECTIVE_HZ)))
    if e_idx <= s_idx:
        return {
            "segment": f"{_format_anchor_time(sub_start_sec)} - {_format_anchor_time(sub_end_sec)}",
            "error": "Empty segment",
        }

    chunk_samples = int(classifier.window_samples)
    chunk_sec = chunk_samples / EFFECTIVE_HZ
    chunks: list[dict] = []
    cur = s_idx
    while cur < e_idx:
        nxt = min(cur + chunk_samples, e_idx)
        sub = torch.from_numpy(signal[:, cur:nxt]).float()
        with _classifier_lock:
            label, conf = classifier.classify_window(sub)
        chunk_start_sec = cur / EFFECTIVE_HZ
        chunk_end_sec = nxt / EFFECTIVE_HZ
        chunks.append({
            "start": _format_anchor_time(chunk_start_sec),
            "end": _format_anchor_time(chunk_end_sec),
            "activity": label,
            "confidence": round(float(conf), 3),
        })
        cur = nxt
    return {
        "segment": f"{_format_anchor_time(sub_start_sec)} - {_format_anchor_time(sub_end_sec)}",
        "chunk_seconds": chunk_sec,
        "classifications": chunks,
    }


def execute_classify_bout(
    start_time: str,
    end_time: str,
    *,
    label_class: str,
    classifier_mode: str,
    recording_duration_sec: float,
    window_start_ms: int,
    annotations: Optional[list],
    signal: Optional[np.ndarray],
) -> str:
    try:
        s = _parse_anchor_time(start_time)
        e = _parse_anchor_time(end_time)
        if e < s:
            e += 24 * 3600
        max_sec = get_max_query_seconds(recording_duration_sec)
        if e - s > max_sec:
            return json.dumps({
                "error": (
                    f"Segment too long: {e - s:.1f}s exceeds maximum {max_sec:.1f}s. "
                    f"Query segments of {max_sec:.1f}s or shorter."
                ),
                "segment": f"{start_time} - {end_time}",
            })
        if s < 0 or e > recording_duration_sec + 1e-3:
            return json.dumps({
                "error": (
                    f"Segment {start_time}-{end_time} is outside the recording "
                    f"window [0, {recording_duration_sec:.1f}s]"
                ),
                "segment": f"{start_time} - {end_time}",
            })

        if classifier_mode == "oracle":
            classes = SLEEP_STAGE_CLASS_NAMES if label_class == "sleep_stages" else AROUSAL_CLASS_NAMES
            return json.dumps(_classify_bout_oracle(s, e, window_start_ms, annotations or [], classes))
        else:
            classifier = _stages_classifier if label_class == "sleep_stages" else _arousals_classifier
            assert signal is not None and classifier is not None, (
                f"Real classifier for '{label_class}' not loaded."
            )
            return json.dumps(_classify_bout_real(s, e, signal, classifier))
    except Exception as exc:
        return json.dumps({"error": str(exc), "segment": f"{start_time} - {end_time}"})


# ---------------------------------------------------------------------------
# Per-sample agentic loop
# ---------------------------------------------------------------------------


def build_user_prompts(
    sample: dict,
    classifier_mode: str,
    recording_duration_sec: float,
    tool_names: list[str],
) -> tuple[str, str]:
    rec_start = _format_anchor_time(0.0)
    rec_end = _format_anchor_time(recording_duration_sec)
    pre = (
        f"You are given a 13-channel polysomnography window for subject "
        f"{sample.get('subject_id')}. The window spans {rec_start} to {rec_end} "
        f"({recording_duration_sec:.0f} seconds at {EFFECTIVE_HZ} Hz).\n\n"
        f"Question: {sample['question']}"
    )
    post = get_instructions(classifier_mode, recording_duration_sec, tool_names)
    return pre, post


def _filter_annotations(
    all_anns: list[tuple[int, int, str]],
    window_start_ms: int,
    window_end_ms: int,
) -> list[tuple[int, int, str]]:
    """Clip annotations to the parquet window and shift to window-local sample space."""
    ws_native = int(round(window_start_ms * SOURCE_HZ / 1000))
    we_native = int(round(window_end_ms * SOURCE_HZ / 1000))
    return [
        (max(0, s - ws_native), min(we_native - ws_native, e - ws_native), lbl)
        for s, e, lbl in all_anns
        if e > ws_native and s < we_native
    ]


def run_sample(
    sample: dict,
    idx: int,
    provider,
    label_class: str,
    classifier_mode: str,
    max_tool_rounds: int,
    reasoning_effort: str,
) -> dict:
    window_start_ms = int(sample["window_start_ms"])
    window_end_ms = int(sample["window_end_ms"])
    recording_duration_sec = (window_end_ms - window_start_ms) / 1000.0
    task_type = sample.get("task_type", "")

    active_tools = tools_for_sample(label_class, task_type, classifier_mode)
    needs_stages = any(t["name"] == "classify_sleep_stage" for t in active_tools)
    needs_arousals = any(t["name"] == "classify_arousal" for t in active_tools)

    annotations_stages: Optional[list] = None
    annotations_arousals: Optional[list] = None
    signal = None

    if classifier_mode == "oracle":
        try:
            if needs_stages:
                annotations_stages = _filter_annotations(
                    load_annotations(sample["subject_id"], "sleep_stages"),
                    window_start_ms, window_end_ms,
                )
            if needs_arousals:
                annotations_arousals = _filter_annotations(
                    load_annotations(sample["subject_id"], "arousals"),
                    window_start_ms, window_end_ms,
                )
        except Exception as e:
            return {"index": idx, "error": f"load_annotations failed: {e}"}
    else:
        signal = load_window(sample["subject_id"], window_start_ms, window_end_ms)

    # window_start_ms = 0 because annotations were shifted to window-local space above.
    local_window_start_ms = 0

    tool_names = [t["name"] for t in active_tools]
    pre, post = build_user_prompts(sample, classifier_mode, recording_duration_sec, tool_names)
    messages = provider.build_messages(SYSTEM_PROMPT, pre, post, None)

    tool_call_count = 0
    tool_calls_success = 0
    tool_calls_error = 0
    rounds = 0
    final_text_parts: list[str] = []
    reasoning_summaries: list[str] = []

    for round_num in range(max_tool_rounds + 1):
        rounds = round_num + 1
        try:
            response: ModelResponse = provider.call(
                messages, active_tools, reasoning_effort, max_tokens=16384,
            )
        except Exception as e:
            return {
                "index": idx,
                "question": sample.get("question", ""),
                "error": f"API error: {e}",
                "rounds": rounds,
            }

        for s in response.reasoning_summaries:
            reasoning_summaries.append(s)
        for t in response.text_parts:
            final_text_parts.append(t)
        provider.append_response(messages, response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_call_count += 1
            args = tc.arguments or {}
            # Determine label class from the tool name.
            if tc.name == "classify_sleep_stage":
                tc_label_class = "sleep_stages"
                tc_annotations = annotations_stages
            else:
                tc_label_class = "arousals"
                tc_annotations = annotations_arousals
            tool_result = execute_classify_bout(
                args.get("start_time", ""),
                args.get("end_time", ""),
                label_class=tc_label_class,
                classifier_mode=classifier_mode,
                recording_duration_sec=recording_duration_sec,
                window_start_ms=local_window_start_ms,
                annotations=tc_annotations,
                signal=signal,
            )
            try:
                if "error" in json.loads(tool_result):
                    tool_calls_error += 1
                else:
                    tool_calls_success += 1
            except Exception:
                tool_calls_error += 1
            provider.append_tool_result(messages, tc.call_id, tool_result)

    if rounds > max_tool_rounds and (not final_text_parts or "Answer:" not in "\n".join(final_text_parts)):
        # Force a final answer
        provider.append_user_text(
            messages,
            "You have used all available tool calls. Provide your final answer now "
            "in the format specified in the instructions.",
        )
        try:
            forced = provider.call(messages, [], reasoning_effort, max_tokens=4096)
            for t in forced.text_parts:
                final_text_parts.append(t)
        except Exception:
            pass

    final_text = "\n".join(final_text_parts)
    answer_type = sample.get("answer_type", "category")
    predicted = extract_final_answer(final_text, answer_type)
    eval_result = evaluate_answer(sample["answer"], predicted, answer_type)

    return {
        "index": idx,
        "subject_id": sample.get("subject_id"),
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "predicted_answer": predicted,
        "final_text": final_text,
        "correct": eval_result["correct"],
        "iou": eval_result.get("iou"),
        "tool_call_count": tool_call_count,
        "tool_calls_success": tool_calls_success,
        "tool_calls_error": tool_calls_error,
        "rounds": rounds,
        "reasoning_summaries": reasoning_summaries,
        "task_type": sample.get("task_type", ""),
        "answer_type": answer_type,
        "context_length_s": sample.get("context_length_s", 0),
        "label_class": label_class,
        "classifier_mode": classifier_mode,
    }


# ---------------------------------------------------------------------------
# Parquet driver
# ---------------------------------------------------------------------------


def load_completed_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    completed = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                completed.add(json.loads(line)["index"])
            except Exception:
                continue
    return completed


def process_parquet(
    parquet_path: Path,
    output_path: Path,
    provider,
    label_class: str,
    classifier_mode: str,
    max_samples: Optional[int],
    max_workers: int,
    max_tool_rounds: int,
    reasoning_effort: str,
) -> dict:
    df = pd.read_parquet(parquet_path)
    if max_samples is not None:
        df = df.head(max_samples)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "trajectories.jsonl"
    completed = load_completed_indices(jsonl_path)
    todo = [i for i in range(len(df)) if i not in completed]
    stats = {"total": len(df), "success": 0, "failed": 0, "skipped": len(completed)}
    if not todo:
        stats["success"] = len(completed)
        return stats

    write_lock = Lock()

    def _one(idx: int) -> dict:
        row = df.iloc[idx].to_dict()
        return run_sample(
            row, idx, provider, label_class, classifier_mode,
            max_tool_rounds, reasoning_effort,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, i): i for i in todo}
        for fut in tqdm(as_completed(futs), total=len(todo), desc="  evaluating"):
            try:
                result = fut.result()
                with write_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                stats["success"] += 1
            except Exception as e:
                idx = futs[fut]
                print(f"  Error sample {idx}: {e}")
                stats["failed"] += 1
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-class", choices=["sleep_stages", "arousals"], required=True)
    ap.add_argument("--classifier-mode", choices=["oracle", "real"], default="oracle")
    ap.add_argument("--stages-checkpoint", type=str, default=None,
                    help="Checkpoint for sleep_stages SleepClassifier (required for real mode "
                         "when running sleep_stages, including state_query)")
    ap.add_argument("--arousals-checkpoint", type=str, default=None,
                    help="Checkpoint for arousals SleepClassifier (required for real mode "
                         "when running arousals or sleep_stages/state_query)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input-dir", type=str, default=str(BASE_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--context-lengths", type=str, nargs="+", default=None,
                    help="Context length grid (e.g. 100 900 3600 full). Defaults per label-class.")
    ap.add_argument("--tasks", type=str, nargs="+", default=["all"])
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", type=str, default="gpt-5.4-2026-03-05")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--max-tool-rounds", type=int, default=None)
    ap.add_argument("--reasoning-effort", type=str, default="high",
                    choices=["none", "low", "medium", "high"])
    ap.add_argument("--provider", type=str, default=None,
                    choices=["openai", "anthropic", "bedrock"])
    ap.add_argument("--aws-region", type=str, default="us-east-1")
    ap.add_argument("--thinking-budget", type=int, default=None)
    args = ap.parse_args()

    # Provider selection
    if args.provider is not None:
        provider_name = args.provider
    elif any(args.model.startswith(p) for p in ("anthropic.", "us.anthropic.", "eu.anthropic.", "global.anthropic.")):
        provider_name = "bedrock"
    elif args.model.startswith("claude-"):
        provider_name = "anthropic"
    else:
        provider_name = "openai"

    if provider_name == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY not set"); sys.exit(1)
        provider = OpenAIProvider(model=args.model)
    elif provider_name == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("Error: ANTHROPIC_API_KEY not set"); sys.exit(1)
        provider = AnthropicProvider(model=args.model, thinking_budget=args.thinking_budget)
    else:
        provider = AnthropicProvider(
            model=args.model, thinking_budget=args.thinking_budget,
            bedrock=True, aws_region=args.aws_region,
        )

    if args.classifier_mode == "real":
        # Determine which checkpoints are needed for the requested label-class.
        # state_query on sleep_stages needs both classifiers.
        need_stages = args.label_class == "sleep_stages"
        need_arousals = args.label_class == "arousals"
        # state_query can appear in sleep_stages runs and requires the arousals classifier too.
        if args.label_class == "sleep_stages":
            need_arousals = True  # always load both for sleep_stages runs (state_query may be present)
        if need_stages and not args.stages_checkpoint:
            print("Error: --stages-checkpoint is required for --classifier-mode real with sleep_stages")
            sys.exit(1)
        if need_arousals and not args.arousals_checkpoint:
            print("Error: --arousals-checkpoint is required for --classifier-mode real "
                  "(needed for arousals tasks and sleep_stages/state_query)")
            sys.exit(1)
        import torch
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        load_real_classifiers(
            stages_checkpoint=args.stages_checkpoint if need_stages else None,
            arousals_checkpoint=args.arousals_checkpoint if need_arousals else None,
            device=device,
        )

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    ctx_grid = args.context_lengths or [str(c) for c in ALL_CONTEXT_LENGTHS_PER_LABEL[args.label_class]]
    # Normalize numeric strings to int.
    normalized_ctx = []
    for c in ctx_grid:
        if c == "full":
            normalized_ctx.append("full")
        else:
            normalized_ctx.append(int(float(c)))

    base_dir = Path(args.input_dir) / args.label_class / "tasks"
    model_slug = args.model.replace("/", "_")
    suffix = [args.label_class, args.classifier_mode]
    if args.classifier_mode == "real":
        ckpt = args.stages_checkpoint or args.arousals_checkpoint
        if ckpt:
            suffix.append(Path(ckpt).parent.name)
    output_root = Path(args.output_dir) / (model_slug + "_" + "_".join(suffix))

    print("=" * 60)
    print(f"Sleep PSG GPT benchmark — provider={provider_name} model={args.model}")
    print(f"label_class={args.label_class} classifier_mode={args.classifier_mode}")
    print(f"Input:  {base_dir}")
    print(f"Output: {output_root}")
    print(f"Tasks:  {tasks}")
    print(f"Ctx:    {normalized_ctx}")
    print("=" * 60)

    total = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    for ctx in normalized_ctx:
        ctx_dir = _format_ctx_dir(ctx)
        if ctx == "full":
            ctx_seconds = 8 * 3600  # rough upper bound for tool-round scheduling
        else:
            ctx_seconds = float(ctx)
        tool_rounds = get_tool_rounds(ctx_seconds, cli_override=args.max_tool_rounds)
        for task in tasks:
            parquet_path = base_dir / ctx_dir / task / args.split / "data.parquet"
            if not parquet_path.exists():
                continue
            n_rows = pd.read_parquet(parquet_path, columns=["question"]).shape[0]
            effective = min(n_rows, args.max_samples) if args.max_samples else n_rows
            print(f"\n{ctx_dir}/{task}/{args.split}: {effective} samples, {tool_rounds} tool rounds")
            stats = process_parquet(
                parquet_path,
                output_root / ctx_dir / task / args.split,
                provider,
                args.label_class,
                args.classifier_mode,
                args.max_samples,
                args.max_workers,
                tool_rounds,
                args.reasoning_effort,
            )
            for k in total:
                total[k] += stats[k]
            print(f"  success={stats['success']}, failed={stats['failed']}, skipped={stats['skipped']}")

    # Aggregate
    summary = {"by_task": {}, "by_context": {}, "overall": {}}
    rows = []
    for jsonl in sorted(output_root.rglob("trajectories.jsonl")):
        with open(jsonl) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if rows:
        n = len(rows)
        c = sum(1 for r in rows if r.get("correct"))
        summary["overall"] = {"total": n, "correct": c, "accuracy": c / n}
        per_task: dict = {}
        for r in rows:
            per_task.setdefault(r.get("task_type", "unknown"), []).append(r)
        for t, lst in sorted(per_task.items()):
            ck = sum(1 for r in lst if r.get("correct"))
            summary["by_task"][t] = {
                "total": len(lst), "correct": ck, "accuracy": ck / len(lst),
            }
        per_ctx: dict = {}
        for r in rows:
            per_ctx.setdefault(str(r.get("context_length_s", "?")), []).append(r)
        for ctx, lst in sorted(per_ctx.items()):
            ck = sum(1 for r in lst if r.get("correct"))
            summary["by_context"][ctx] = {
                "total": len(lst), "correct": ck, "accuracy": ck / len(lst),
            }
    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    if summary.get("overall"):
        ov = summary["overall"]
        print(f"Overall: {ov['correct']}/{ov['total']} = {ov['accuracy'] * 100:.1f}%")
    print(f"\nTotal: success={total['success']}, failed={total['failed']}, skipped={total['skipped']}")


if __name__ == "__main__":
    main()
