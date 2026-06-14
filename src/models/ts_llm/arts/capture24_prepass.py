#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Oracle pre-pass benchmark for TS-Haystack (Capture24).

Injects the ground-truth activity timeline from each sample's
``needles`` + ``difficulty_config`` as text into the user prompt. The LLM
answers by reasoning over the timestamped labels directly, with no tools.

Unlike :mod:`src.models.ts_llm.arts.capture24` oracle mode — which hands the LLM a
``classify_bout`` tool backed by an :class:`OracleClassifier` and requires
multi-turn tool calls to discover activities — this script emits the full
timeline up front. Labels are *not* binned to a fixed grid; segments keep
their original start/end boundaries from the sample's needles / global
timeline.

Mirrors :mod:`src.models.ts_llm.arts.sleep_prepass` oracle mode for the
Capture24 dataset.

Example:
    python3 -m src.models.ts_llm.arts.capture24_prepass \\
        --context-lengths 100 --tasks existence --max-samples 3 --max-workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from src.models.ts_llm.arts.capture24 import (
    ACTIVITY_CLASSES,
    ALL_TASKS,
    ANOMALY_INSTRUCTIONS,
    DEFAULT_INPUT_DIR,
    SYSTEM_PROMPT,
)
from src.models.ts_llm.arts.providers import (
    AnthropicProvider,
    OpenAIProvider,
)
from src.datasets.capture24_haystack.utils import format_context_dir
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.datasets.capture24_haystack.utils.oracle_utils import format_oracle_timeline


DEFAULT_OUTPUT_DIR = Path("results") / "gpt_prepass_benchmark"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_INSTRUCTIONS_HEAD = """\
Instructions:
- A ground-truth activity timeline of the full recording is provided above \
(activities with exact timestamps; inserted segments are marked [inserted]).
- Use the provided timeline directly to answer the question — the labels are \
authoritative; no classifier calls are available.
- Provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting questions answer \
with just the number. For time-based answers provide a range: \
"H:MM:SS.mmm AM/PM to H:MM:SS.mmm AM/PM"."""


def get_instructions(task_type: str) -> str:
    instr = f"{_INSTRUCTIONS_HEAD}\n\nValid activity classes: {ACTIVITY_CLASSES}"
    if task_type in ("anomaly_detection", "anomaly_localization"):
        instr += ANOMALY_INSTRUCTIONS
    return instr


def build_user_prompts(sample: dict, timeline_text: str) -> tuple[str, str]:
    pre = (
        f"You are given accelerometer data in all three dimensions from a "
        f"wrist-worn sensor. The recording spans from "
        f"{sample['recording_time_start']} to {sample['recording_time_end']}.\n\n"
        f"{timeline_text}\n\n"
        f"Question: {sample['question']}"
    )
    post = get_instructions(sample.get("task_type", ""))
    return pre, post


# ---------------------------------------------------------------------------
# Per-sample inference (single call — no agentic loop)
# ---------------------------------------------------------------------------


def run_sample(
    sample: dict,
    idx: int,
    provider,
    reasoning_effort: str,
) -> dict:
    timeline_text = format_oracle_timeline(
        needles=sample.get("needles"),
        difficulty_config=sample.get("difficulty_config"),
        recording_time_start=sample["recording_time_start"],
        recording_time_end=sample["recording_time_end"],
        context_length_samples=int(sample.get("context_length_samples", 0) or 0),
    )
    pre, post = build_user_prompts(sample, timeline_text)
    messages = provider.build_messages(SYSTEM_PROMPT, pre, post, None)

    input_char_lengths = {
        "system": len(SYSTEM_PROMPT),
        "pre_prompt": len(pre),
        "post_prompt": len(post),
        "timeline_text": len(timeline_text),
        "question": len(sample.get("question", "") or ""),
        "total": len(SYSTEM_PROMPT) + len(pre) + len(post),
    }

    try:
        response = provider.call(messages, [], reasoning_effort, max_tokens=16384)
    except Exception as exc:
        return {
            "index": idx,
            "question": sample.get("question", ""),
            "error": f"API error: {exc}",
            "task_type": sample.get("task_type", ""),
            "context_length_samples": sample.get("context_length_samples", 0),
        }

    final_text = "\n".join(response.text_parts)
    answer_type = sample.get("answer_type", "category")
    predicted = extract_final_answer(final_text, answer_type)
    eval_result = evaluate_answer(sample["answer"], predicted, answer_type)

    return {
        "index": idx,
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "predicted_answer": predicted,
        "final_text": final_text,
        "correct": eval_result["correct"],
        "iou": eval_result.get("iou"),
        "normalized_gt": eval_result.get("normalized_gt"),
        "normalized_pred": eval_result.get("normalized_pred"),
        "reasoning_summaries": response.reasoning_summaries,
        "task_type": sample.get("task_type", ""),
        "answer_type": answer_type,
        "context_length_samples": sample.get("context_length_samples", 0),
        "classifier_mode": "oracle_prepass",
        "timeline_text": timeline_text,
        "input_char_lengths": input_char_lengths,
    }


# ---------------------------------------------------------------------------
# Parquet driver with resume support
# ---------------------------------------------------------------------------


def load_completed_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    out: set[int] = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                out.add(json.loads(line)["index"])
            except Exception:
                continue
    return out


def process_parquet(
    parquet_path: Path,
    output_path: Path,
    provider,
    max_samples: Optional[int],
    max_workers: int,
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
        print(f"  Already complete ({len(completed)}/{len(df)}), skipping")
        return stats

    if completed:
        print(f"  Resuming: {len(completed)}/{len(df)} already done")

    write_lock = Lock()
    running = {"correct": 0, "total": 0, "chars_sum": 0, "chars_n": 0}

    def _one(idx: int) -> dict:
        row = df.iloc[idx].to_dict()
        return run_sample(row, idx, provider, reasoning_effort)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, i): i for i in todo}
        pbar = tqdm(as_completed(futs), total=len(todo), desc="  evaluating")
        for fut in pbar:
            try:
                result = fut.result()
                with write_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                    if "error" not in result:
                        running["total"] += 1
                        if result.get("correct"):
                            running["correct"] += 1
                        ic = result.get("input_char_lengths") or {}
                        if ic.get("total"):
                            running["chars_sum"] += int(ic["total"])
                            running["chars_n"] += 1
                stats["success"] += 1
                postfix: dict[str, str] = {}
                if running["total"]:
                    postfix["acc"] = (
                        f"{running['correct'] / running['total'] * 100:.0f}%"
                    )
                if running["chars_n"]:
                    postfix["in_chars"] = (
                        f"{running['chars_sum'] / running['chars_n'] / 1000:.1f}k"
                    )
                if postfix:
                    pbar.set_postfix(postfix)
            except Exception as exc:
                idx = futs[fut]
                print(f"  Error sample {idx}: {exc}")
                stats["failed"] += 1
        pbar.close()

    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def compute_summary(output_dir: Path) -> dict:
    summary: dict = {"by_task": {}, "by_context": {}, "by_answer_type": {}, "overall": {}}
    rows: list[dict] = []
    for jsonl in sorted(output_dir.rglob("trajectories.jsonl")):
        with open(jsonl) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    rows = [r for r in rows if "error" not in r]
    if not rows:
        return summary

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
        per_ctx.setdefault(str(r.get("context_length_samples", "?")), []).append(r)
    for ctx, lst in sorted(per_ctx.items()):
        ck = sum(1 for r in lst if r.get("correct"))
        summary["by_context"][ctx] = {
            "total": len(lst), "correct": ck, "accuracy": ck / len(lst),
        }

    per_at: dict = {}
    for r in rows:
        per_at.setdefault(r.get("answer_type", "unknown"), []).append(r)
    for at, lst in sorted(per_at.items()):
        ck = sum(1 for r in lst if r.get("correct"))
        ious = [r["iou"] for r in lst if r.get("iou") is not None]
        entry = {"total": len(lst), "correct": ck, "accuracy": ck / len(lst)}
        if ious:
            entry["avg_iou"] = sum(ious) / len(ious)
        summary["by_answer_type"][at] = entry

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Oracle pre-pass benchmark for TS-Haystack (no tools; "
                    "LLM reasons over the ground-truth timeline in context)."
    )
    ap.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--context-lengths", type=float, nargs="+", default=[100])
    ap.add_argument("--tasks", type=str, nargs="+", default=["all"])
    ap.add_argument("--splits", type=str, nargs="+", default=["test"])
    ap.add_argument("--model", type=str, default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument(
        "--reasoning-effort", type=str, default="high",
        choices=["none", "low", "medium", "high"],
    )
    # Oracle is the only pre-pass mode currently; the flag is kept for symmetry
    # with benchmark_gpt_sleep_prepass.py and to leave room for a future
    # real-classifier pre-pass.
    ap.add_argument(
        "--classifier-mode", type=str, default="oracle", choices=["oracle"],
        help="Pre-pass classifier mode (only 'oracle' supported).",
    )
    ap.add_argument(
        "--provider", type=str, default=None,
        choices=["openai", "anthropic", "bedrock"],
    )
    ap.add_argument("--aws-region", type=str, default="us-east-1")
    ap.add_argument("--thinking-budget", type=int, default=None)
    args = ap.parse_args()

    # Provider selection
    if args.provider is not None:
        provider_name = args.provider
    elif any(args.model.startswith(p) for p in
             ("anthropic.", "us.anthropic.", "eu.anthropic.", "global.anthropic.")):
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

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    input_dir = Path(args.input_dir)
    model_slug = args.model.replace("/", "_")
    suffix = ["oracle_prepass"]
    if args.reasoning_effort != "high":
        suffix.append(f"reason_{args.reasoning_effort}")
    if args.thinking_budget is not None:
        suffix.append(f"budget_{args.thinking_budget}")
    output_dir = Path(args.output_dir) / (model_slug + "_" + "_".join(suffix))

    print("=" * 60)
    print("TS-Haystack Oracle Pre-pass Benchmark")
    print("=" * 60)
    print(f"Provider: {provider_name}")
    print(f"Model: {args.model}")
    print(f"Classifier mode: {args.classifier_mode}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tasks:  {tasks}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Workers: {args.max_workers}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    if args.max_samples:
        print(f"Max samples per file: {args.max_samples}")
    print()

    total = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for ctx_seconds in args.context_lengths:
        ctx_dir = format_context_dir(ctx_seconds)
        for task in tasks:
            for split in args.splits:
                parquet_path = input_dir / ctx_dir / task / split / "data.parquet"
                if not parquet_path.exists():
                    continue
                n_rows = pd.read_parquet(parquet_path, columns=["question"]).shape[0]
                effective = min(n_rows, args.max_samples) if args.max_samples else n_rows
                print(f"\n{ctx_dir}/{task}/{split} ({effective} samples)")
                result_path = output_dir / ctx_dir / task / split
                stats = process_parquet(
                    parquet_path, result_path, provider,
                    args.max_samples, args.max_workers, args.reasoning_effort,
                )
                for k in total:
                    total[k] += stats[k]
                print(f"  success={stats['success']}, failed={stats['failed']}, "
                      f"skipped={stats['skipped']}")

    summary = compute_summary(output_dir)
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    ov = summary.get("overall") or {}
    if ov:
        print(f"Overall: {ov['correct']}/{ov['total']} = {ov['accuracy']*100:.1f}%")
        bt = summary.get("by_task") or {}
        if bt:
            print(f"\n{'Task':<25} {'Acc':>7} {'N':>5}")
            print(f"{'-'*25} {'-'*7} {'-'*5}")
            for t, m in sorted(bt.items()):
                print(f"{t:<25} {m['accuracy']*100:>6.1f}% {m['total']:>5d}")

    print(f"\nTotal: success={total['success']}, failed={total['failed']}, "
          f"skipped={total['skipped']}")


if __name__ == "__main__":
    main()
