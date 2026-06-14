#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark LLMs on the Sleep-PSG haystack using the trained sleep classifiers
as a *pre-pass context provider* + agentic tool back-end.

Unlike ``benchmark_gpt_sleep.py`` — which hands ``classify_sleep_stage`` /
``classify_arousal`` tools to the agent and lets it discover the recording
through tool calls alone — this script runs the full QA window through both
sleep classifiers at their native resolutions *before* the LLM sees the
question:

    * ``SleepClassifier(sleep_stages)`` — non-overlapping 30 s windows → a
      sleep-stage timeline (bouts of consecutive same-label chunks) covering
      the full QA window.
    * ``SleepClassifier(arousals)``     — non-overlapping 20 s windows → a
      list of detected arousal / respiratory events (chunks whose argmax is
      not ``none``).

The two pre-pass outputs are formatted as a compact textual context block and
injected into the user prompt. The agent still has ``classify_sleep_stage``
and ``classify_arousal`` tools so it can drill into sub-ranges; the tool
implementations are thin filters over the cached pre-pass predictions.

Additionally, every sample logs how well the classifier pre-pass did against
the ground-truth annotations inside the window (accuracy, macro F1, per-class
F1, confusion) so you can separate "did the classifier get it right" from
"did the LLM use the classifier right" at analysis time.

Examples:
    # Sleep stages on 900 s windows
    python3 -m src.models.ts_llm.arts.sleep_prepass \\
        --label-class sleep_stages \\
        --stages-checkpoint results/sleep_classifier/sleep_stages/best_classifier.pt \\
        --arousals-checkpoint results/sleep_classifier/arousals/best_classifier.pt \\
        --context-lengths 900 --tasks existence --max-samples 5 --max-workers 1

    # Arousals on 900 s windows
    python3 -m src.models.ts_llm.arts.sleep_prepass \\
        --label-class arousals \\
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

# Reuse provider adapters, response dataclasses, and tool-scheduling helpers
# from the main benchmark.
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
    _format_ctx_dir,
)
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.datasets.capture24_haystack.utils.timestamp_utils import parse_time_string
from src.models.classifiers.sleep.model import (
    AROUSAL_CLASS_NAMES,
    SLEEP_STAGE_CLASS_NAMES,
    SleepClassifier,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("results") / "gpt_sleep_prepass_benchmark"

# Native classifier window sizes (seconds).
STAGE_CHUNK_SEC = 30
AROUSAL_CHUNK_SEC = 20

# Synthetic anchor for "H:MM:SS.mmm AM/PM" timestamps — matches
# benchmark_gpt_sleep.py so trajectories line up across the two scripts.
RECORDING_ANCHOR = datetime(2000, 1, 1, 0, 0, 0)


def _format_anchor_time(seconds: float) -> str:
    dt = RECORDING_ANCHOR + timedelta(seconds=max(0.0, float(seconds)))
    return dt.strftime("%I:%M:%S.") + f"{dt.microsecond // 1000:03d} {dt.strftime('%p')}"


def _parse_anchor_time(ts: str) -> float:
    parsed = parse_time_string(ts)
    parsed = parsed.replace(
        year=RECORDING_ANCHOR.year, month=RECORDING_ANCHOR.month,
        day=RECORDING_ANCHOR.day,
    )
    delta = (parsed - RECORDING_ANCHOR).total_seconds()
    if delta < 0:
        delta += 24 * 3600
    return delta


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    f"You are an expert polysomnography (PSG) analyst. You are given a "
    f"13-channel PSG recording sampled at {EFFECTIVE_HZ} Hz. Channels (in order): "
    f"{', '.join(CHANNEL_NAMES)}.\n\n"
    f"Possible sleep stages: {', '.join(SLEEP_STAGE_CLASS_NAMES)}.\n"
    f"Possible arousal / respiratory event types: {', '.join(AROUSAL_CLASS_NAMES)}."
)


_INSTRUCTIONS_TAIL = """\
- After consulting the classifier context and (optionally) tool results, provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting answer with just the number. \
For time-based answers provide a range: "H:MM:SS.mmm AM/PM to H:MM:SS.mmm AM/PM"."""


def get_instructions(
    recording_duration_sec: float,
    tool_names: list[str],
    classifier_mode: str = "real",
) -> str:
    if classifier_mode == "oracle":
        oracle_tail = (
            "- After consulting the timeline, provide your final answer prefixed "
            "with \"Answer: \".\n"
            "- For boolean questions answer \"Yes\" or \"No\". For counting answer "
            "with just the number. For time-based answers provide a range: "
            "\"H:MM:SS.mmm AM/PM to H:MM:SS.mmm AM/PM\"."
        )
        head = (
            "Instructions:\n"
            "- A ground-truth analysis of the full recording is provided below "
            "(sleep stages and/or arousal events, with exact timestamps).\n"
            "- Use the provided timeline directly to answer the question; "
            "the labels are authoritative.\n"
        )
        return f"{head}{oracle_tail}"
    tool_list = " and ".join(f"`{n}`" for n in tool_names)
    head = (
        "Instructions:\n"
        "- A pre-computed classifier analysis of the full recording is provided "
        "below. Consult it first.\n"
        f"- You may also call {tool_list} on any sub-range for a finer look; "
        "these tools return the same pre-computed predictions filtered to that range.\n"
        f"- `classify_sleep_stage` predictions are at {STAGE_CHUNK_SEC}-second "
        f"resolution; `classify_arousal` is at {AROUSAL_CHUNK_SEC}-second resolution.\n"
        "- Classifier predictions may be wrong — cross-check against the question "
        "and prefer evidence from multiple chunks when possible.\n"
    )
    max_q = get_max_query_seconds(recording_duration_sec)
    strategy = (
        f"- Recording duration: {recording_duration_sec:.0f}s. "
        f"Max tool query length: {max_q:.0f}s.\n"
    )
    return f"{head}{strategy}{_INSTRUCTIONS_TAIL}"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

# Tasks that require both tool types simultaneously (cross-timeline queries).
BOTH_TOOLS_TASKS: set[tuple[str, str]] = {("sleep_stages", "state_query")}

_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "start_time": {
            "type": "string",
            "description": "Sub-range start in 'H:MM:SS.mmm AM/PM' format (window-relative)",
        },
        "end_time": {
            "type": "string",
            "description": "Sub-range end in 'H:MM:SS.mmm AM/PM' format (window-relative)",
        },
    },
    "required": ["start_time", "end_time"],
    "additionalProperties": False,
}


def get_sleep_stage_tool() -> dict:
    return {
        "type": "function",
        "name": "classify_sleep_stage",
        "description": (
            f"Return the per-chunk sleep-stage predictions ({STAGE_CHUNK_SEC}-second "
            "non-overlapping windows) inside the given sub-range, as produced by the "
            "sleep-stages SleepClassifier in the pre-pass. Each entry has start/end "
            "timestamps, top predicted class, and softmax confidence."
        ),
        "parameters": _TOOL_PARAMS,
        "strict": True,
    }


def get_arousal_tool() -> dict:
    return {
        "type": "function",
        "name": "classify_arousal",
        "description": (
            f"Return the per-chunk arousal predictions ({AROUSAL_CHUNK_SEC}-second "
            "non-overlapping windows) inside the given sub-range, as produced by the "
            "arousals SleepClassifier in the pre-pass. Each entry has start/end "
            "timestamps, top predicted class (or 'none' for no event), and softmax "
            "confidence."
        ),
        "parameters": _TOOL_PARAMS,
        "strict": True,
    }


def tools_for_sample(label_class: str, task_type: str) -> list[dict]:
    if (label_class, task_type) in BOTH_TOOLS_TASKS:
        return [get_sleep_stage_tool(), get_arousal_tool()]
    if label_class == "sleep_stages":
        return [get_sleep_stage_tool()]
    return [get_arousal_tool()]


# ---------------------------------------------------------------------------
# Classifier loading — thread-safe inference
# ---------------------------------------------------------------------------

_classifier_lock = Lock()
_stages_classifier: Optional[SleepClassifier] = None
_arousals_classifier: Optional[SleepClassifier] = None


def load_classifiers(
    stages_checkpoint: Optional[str],
    arousals_checkpoint: Optional[str],
    device: str,
) -> None:
    global _stages_classifier, _arousals_classifier
    if stages_checkpoint is not None:
        print(f"Loading sleep_stages SleepClassifier from {stages_checkpoint} on {device}...")
        _stages_classifier = SleepClassifier.load(stages_checkpoint, device=device)
        print(f"  classes: {_stages_classifier.class_names}  "
              f"window_samples: {_stages_classifier.window_samples}")
    if arousals_checkpoint is not None:
        print(f"Loading arousals SleepClassifier from {arousals_checkpoint} on {device}...")
        _arousals_classifier = SleepClassifier.load(arousals_checkpoint, device=device)
        print(f"  classes: {_arousals_classifier.class_names}  "
              f"window_samples: {_arousals_classifier.window_samples}")


# ---------------------------------------------------------------------------
# Pre-pass — classify the full window with each model at its native window
# ---------------------------------------------------------------------------


def _pre_pass_chunks(
    signal: np.ndarray,
    classifier: SleepClassifier,
    chunk_sec: float,
    device: str,
    batch_size: int = 128,
) -> list[dict]:
    """Classify non-overlapping ``chunk_sec``-second windows of the signal.

    signal: (C, L) float32 at EFFECTIVE_HZ, already per-channel z-scored by
    ``load_window``. Chunks shorter than ``classifier.window_samples`` are
    right-padded with zeros; chunks longer are truncated.

    Inference is split into mini-batches of ``batch_size`` chunks to keep GPU
    memory bounded — important for long windows (7200 s → 240/360 chunks,
    enough to OOM Chronos-2 in a single forward). If a sub-batch still OOMs,
    we halve and retry until we fall back to one-chunk-at-a-time.
    """
    import torch
    import torch.nn.functional as F

    C, L = signal.shape
    chunk_samples = int(round(chunk_sec * EFFECTIVE_HZ))
    native_samples = int(classifier.window_samples)
    if L < chunk_samples or chunk_samples <= 0:
        return []

    n_chunks = L // chunk_samples
    batch = np.zeros((n_chunks, C, native_samples), dtype=np.float32)
    for i in range(n_chunks):
        s = i * chunk_samples
        take = min(native_samples, chunk_samples)
        batch[i, :, :take] = signal[:, s:s + take]

    classes = classifier.class_names
    out: list[dict] = [None] * n_chunks  # type: ignore[list-item]

    with _classifier_lock:
        with torch.no_grad():
            i = 0
            bs = max(1, int(batch_size))
            while i < n_chunks:
                j = min(i + bs, n_chunks)
                try:
                    x = torch.from_numpy(batch[i:j]).to(device)
                    logits = classifier(x)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    del x, logits
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    continue
                labels = probs.argmax(-1)
                for k in range(j - i):
                    idx = i + k
                    out[idx] = {
                        "start_s": idx * chunk_sec,
                        "end_s": (idx + 1) * chunk_sec,
                        "label": classes[int(labels[k])],
                        "conf": float(probs[k, labels[k]]),
                    }
                i = j
            torch.cuda.empty_cache()
    return out  # type: ignore[return-value]


def _compress_bouts(chunks: list[dict]) -> list[dict]:
    """Merge adjacent same-label chunks into bouts (mean confidence)."""
    if not chunks:
        return []
    bouts: list[dict] = []
    cur_start = chunks[0]["start_s"]
    cur_end = chunks[0]["end_s"]
    cur_label = chunks[0]["label"]
    confs = [chunks[0]["conf"]]
    for c in chunks[1:]:
        if c["label"] == cur_label and c["start_s"] == cur_end:
            cur_end = c["end_s"]
            confs.append(c["conf"])
        else:
            bouts.append({
                "start_s": cur_start, "end_s": cur_end, "label": cur_label,
                "conf": float(np.mean(confs)), "n_chunks": len(confs),
            })
            cur_start, cur_end, cur_label, confs = (
                c["start_s"], c["end_s"], c["label"], [c["conf"]],
            )
    bouts.append({
        "start_s": cur_start, "end_s": cur_end, "label": cur_label,
        "conf": float(np.mean(confs)), "n_chunks": len(confs),
    })
    return bouts


def _filter_annotations(
    all_anns: list[tuple[int, int, str]],
    window_start_ms: int,
    window_end_ms: int,
) -> list[tuple[int, int, str]]:
    """Clip annotations to the parquet window and shift to window-local SOURCE_HZ samples."""
    ws_native = int(round(window_start_ms * SOURCE_HZ / 1000))
    we_native = int(round(window_end_ms * SOURCE_HZ / 1000))
    return [
        (max(0, s - ws_native), min(we_native - ws_native, e - ws_native), lbl)
        for s, e, lbl in all_anns
        if e > ws_native and s < we_native
    ]


def _gt_label_for_chunk(
    anns: list[tuple[int, int, str]],
    chunk_start_s: float,
    chunk_end_s: float,
    classes: list[str],
    default_label: Optional[str] = None,
) -> Optional[str]:
    """Return the label with the greatest overlap with this chunk, restricted
    to ``classes``. If no class from the vocabulary overlaps the chunk:
        - return ``default_label`` if not None (used for arousals → "none"),
        - else return ``None`` (used for sleep stages — treat as unscored).

    Annotations are in window-local SOURCE_HZ samples.
    """
    chunk_s_native = int(round(chunk_start_s * SOURCE_HZ))
    chunk_e_native = int(round(chunk_end_s * SOURCE_HZ))
    totals: dict[str, int] = {}
    for s, e, lbl in anns:
        if lbl not in classes:
            continue
        lo = max(chunk_s_native, s)
        hi = min(chunk_e_native, e)
        if hi > lo:
            totals[lbl] = totals.get(lbl, 0) + (hi - lo)
    if not totals:
        return default_label
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _oracle_chunks(
    anns: list[tuple[int, int, str]],
    window_duration_sec: float,
    chunk_sec: float,
    classes: list[str],
    default_label: Optional[str],
) -> list[dict]:
    """Build per-chunk ground-truth labels on the same non-overlapping grid the
    real classifier pre-pass uses. Chunks with no matching annotation fall back
    to ``default_label`` (``"none"`` for arousals, ``None`` → ``"unknown"`` for
    sleep stages). ``conf`` is 1.0 by construction."""
    n_chunks = int(window_duration_sec // chunk_sec)
    out: list[dict] = []
    for i in range(n_chunks):
        start_s = i * chunk_sec
        end_s = (i + 1) * chunk_sec
        gt = _gt_label_for_chunk(
            anns, start_s, end_s, classes, default_label=default_label,
        )
        out.append({
            "start_s": start_s,
            "end_s": end_s,
            "label": gt if gt is not None else "unknown",
            "conf": 1.0,
            "gt": gt,
        })
    return out


def _oracle_arousal_events(
    anns: list[tuple[int, int, str]],
    classes: list[str],
) -> list[dict]:
    """Convert filtered arousal annotations (native SOURCE_HZ samples, already
    window-local) into per-event dicts in seconds. Preserves true start/end
    boundaries and keeps distinct events separate — the chunk grid is lossy for
    irregular event data and was causing ordinal collapse on localization."""
    out: list[dict] = []
    for s, e, lbl in anns:
        if lbl not in classes:
            continue
        out.append({
            "start_s": s / SOURCE_HZ,
            "end_s": e / SOURCE_HZ,
            "label": lbl,
            "conf": 1.0,
        })
    out.sort(key=lambda ev: (ev["start_s"], ev["end_s"]))
    return out


def run_pre_pass(
    sample: dict,
    label_class: str,
    need_stages: bool,
    need_arousals: bool,
    device: str,
    batch_size: int = 128,
    classifier_mode: str = "real",
) -> dict:
    """Full pre-pass — produce per-chunk labels for the requested streams.

    In ``real`` mode: run each SleepClassifier at its native resolution and
    attach per-chunk ground-truth labels (for metrics).
    In ``oracle`` mode: read annotations directly, no classifier forward pass
    and no raw-signal loading.

    Returns ``{'stages': [chunks...]?, 'arousals': [chunks...]?,
               'stages_bouts': [...], 'arousals_bouts': [...]}``.
    """
    subject_id = str(sample["subject_id"])
    window_start_ms = int(sample["window_start_ms"])
    window_end_ms = int(sample["window_end_ms"])
    window_duration_sec = (window_end_ms - window_start_ms) / 1000.0

    signal = (
        load_window(subject_id, window_start_ms, window_end_ms)
        if classifier_mode == "real" else None
    )

    out: dict[str, list[dict]] = {}

    if need_stages:
        try:
            anns = _filter_annotations(
                load_annotations(subject_id, "sleep_stages"),
                window_start_ms, window_end_ms,
            )
        except Exception:
            anns = []
        if classifier_mode == "oracle":
            chunks = _oracle_chunks(
                anns, window_duration_sec, STAGE_CHUNK_SEC,
                SLEEP_STAGE_CLASS_NAMES, default_label=None,
            )
        else:
            assert _stages_classifier is not None, "sleep_stages classifier not loaded"
            chunks = _pre_pass_chunks(
                signal, _stages_classifier, STAGE_CHUNK_SEC, device,
                batch_size=batch_size,
            )
            for c in chunks:
                c["gt"] = _gt_label_for_chunk(
                    anns, c["start_s"], c["end_s"], SLEEP_STAGE_CLASS_NAMES,
                    default_label=None,
                )
        out["stages"] = chunks
        out["stages_bouts"] = _compress_bouts(chunks)

    if need_arousals:
        try:
            anns = _filter_annotations(
                load_annotations(subject_id, "arousals"),
                window_start_ms, window_end_ms,
            )
        except Exception:
            anns = []
        if classifier_mode == "oracle":
            # Emit raw events (true start/end) rather than a 20s chunk grid —
            # the grid quantizes boundaries and merges distinct events that
            # share a chunk, which broke localization / ordinal questions.
            out["arousals_events"] = _oracle_arousal_events(
                anns, AROUSAL_CLASS_NAMES,
            )
        else:
            assert _arousals_classifier is not None, "arousals classifier not loaded"
            chunks = _pre_pass_chunks(
                signal, _arousals_classifier, AROUSAL_CHUNK_SEC, device,
                batch_size=batch_size,
            )
            for c in chunks:
                c["gt"] = _gt_label_for_chunk(
                    anns, c["start_s"], c["end_s"], AROUSAL_CLASS_NAMES,
                    default_label="none",
                )
            out["arousals"] = chunks
            out["arousals_bouts"] = _compress_bouts(chunks)

    return out


# ---------------------------------------------------------------------------
# Pre-pass metrics — classifier output vs. ground truth
# ---------------------------------------------------------------------------


def _per_class_f1(preds: list[str], gts: list[str], classes: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def _confusion(preds: list[str], gts: list[str], classes: list[str]) -> dict:
    m: dict[str, dict[str, int]] = {g: {p: 0 for p in classes} for g in classes}
    for p, g in zip(preds, gts):
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def _metrics_for_chunks(chunks: list[dict], classes: list[str]) -> dict:
    preds = [c["label"] for c in chunks if c.get("gt") is not None]
    gts = [c["gt"] for c in chunks if c.get("gt") is not None]
    if not preds:
        return {
            "n_chunks_scored": 0, "n_chunks_total": len(chunks),
            "accuracy": None, "macro_f1": None, "per_class_f1": {},
            "support": {c: 0 for c in classes},
            "confusion": {g: {p: 0 for p in classes} for g in classes},
        }
    per_class = _per_class_f1(preds, gts, classes)
    return {
        "n_chunks_scored": len(preds),
        "n_chunks_total": len(chunks),
        "accuracy": sum(1 for p, g in zip(preds, gts) if p == g) / len(preds),
        "macro_f1": float(np.mean(list(per_class.values()))),
        "per_class_f1": per_class,
        "support": {c: gts.count(c) for c in classes},
        "confusion": _confusion(preds, gts, classes),
    }


def compute_pre_pass_metrics(pre_pass: dict) -> dict:
    out: dict[str, dict] = {}
    if "stages" in pre_pass:
        out["stages"] = _metrics_for_chunks(pre_pass["stages"], SLEEP_STAGE_CLASS_NAMES)
    if "arousals" in pre_pass:
        out["arousals"] = _metrics_for_chunks(pre_pass["arousals"], AROUSAL_CLASS_NAMES)
    return out


def _aggregate_pre_pass(rows: list[dict]) -> dict:
    """Pool per-sample confusion matrices into corpus-level rhythm/beat metrics."""
    agg: dict[str, dict] = {}
    for kind, classes in (
        ("stages", SLEEP_STAGE_CLASS_NAMES),
        ("arousals", AROUSAL_CLASS_NAMES),
    ):
        confusion = {g: {p: 0 for p in classes} for g in classes}
        support = {c: 0 for c in classes}
        n_scored = 0
        for r in rows:
            m = (r.get("pre_pass_metrics") or {}).get(kind) or {}
            n_scored += int(m.get("n_chunks_scored") or 0)
            for c, v in (m.get("support") or {}).items():
                if c in support:
                    support[c] += int(v)
            for g, prow in (m.get("confusion") or {}).items():
                if g not in confusion:
                    continue
                for p, v in prow.items():
                    if p in confusion[g]:
                        confusion[g][p] += int(v)
        if n_scored == 0:
            continue
        total = sum(confusion[g][p] for g in classes for p in classes)
        correct = sum(confusion[g][g] for g in classes)
        per_class: dict[str, float] = {}
        f1s = []
        for c in classes:
            tp = confusion[c][c]
            fp = sum(confusion[g][c] for g in classes if g != c)
            fn = sum(confusion[c][p] for p in classes if p != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[c] = f1
            if support[c] > 0:
                f1s.append(f1)
        agg[kind] = {
            "n_chunks_scored": n_scored,
            "accuracy": (correct / total) if total else None,
            "macro_f1": float(np.mean(f1s)) if f1s else None,
            "per_class_f1": per_class,
            "support": support,
            "confusion": confusion,
        }
    return agg


# ---------------------------------------------------------------------------
# Context-file rendering
# ---------------------------------------------------------------------------


def render_context_file(
    pre_pass: dict,
    window_duration_sec: float,
    max_arousal_events: int = 200,
    classifier_mode: str = "real",
) -> str:
    is_oracle = classifier_mode == "oracle"
    source_label = "ground truth" if is_oracle else "SleepClassifier"
    header_label = "Ground-truth" if is_oracle else "Classifier"
    lines: list[str] = [
        f"# {header_label} pre-pass summary (window "
        f"{_format_anchor_time(0.0)} - {_format_anchor_time(window_duration_sec)})"
    ]

    if "stages_bouts" in pre_pass:
        bouts = pre_pass["stages_bouts"]
        lines.append("")
        lines.append(
            f"## Sleep-stage timeline ({source_label}, {STAGE_CHUNK_SEC}s resolution, "
            f"{len(bouts)} bouts)"
        )
        if not bouts:
            lines.append("  (no sleep-stage labels — window shorter than one chunk)")
        for b in bouts:
            dur = b["end_s"] - b["start_s"]
            conf_part = "" if is_oracle else f"  conf={b['conf']:.2f}"
            lines.append(
                f"  {_format_anchor_time(b['start_s'])} - "
                f"{_format_anchor_time(b['end_s'])}  "
                f"{b['label']:<4s}{conf_part}  "
                f"({b['n_chunks']} chunk{'s' if b['n_chunks'] != 1 else ''}, {dur:.0f}s)"
            )

    if "arousals_events" in pre_pass:
        events = pre_pass["arousals_events"]
        lines.append("")
        lines.append(
            f"## Arousal / respiratory-event timeline (ground truth, "
            f"{len(events)} event{'s' if len(events) != 1 else ''}): "
            + "  ".join(
                f"{cls}={sum(1 for ev in events if ev['label'] == cls)}"
                for cls in AROUSAL_CLASS_NAMES if cls != "none"
            )
        )
        if not events:
            lines.append("  (no arousal or respiratory events)")
        else:
            if len(events) > max_arousal_events:
                lines.append(
                    f"  (listing first {max_arousal_events} of {len(events)} events)"
                )
                events = events[:max_arousal_events]
            for ev in events:
                lines.append(
                    f"  {_format_anchor_time(ev['start_s'])} - "
                    f"{_format_anchor_time(ev['end_s'])}  "
                    f"{ev['label']:<18s}"
                )
    elif "arousals" in pre_pass:
        chunks = pre_pass["arousals"]
        events = [c for c in chunks if c["label"] != "none"]
        n_none = len(chunks) - len(events)
        lines.append("")
        lines.append(
            f"## Arousal / respiratory-event timeline ({source_label}, "
            f"{AROUSAL_CHUNK_SEC}s resolution): "
            f"{len(chunks)} chunks analyzed  | "
            + "  ".join(f"{cls}={sum(1 for c in events if c['label']==cls)}"
                        for cls in AROUSAL_CLASS_NAMES if cls != "none")
            + f"  none={n_none}"
        )
        if not events:
            lines.append(f"  (no arousal or respiratory events detected)")
        else:
            if len(events) > max_arousal_events:
                lines.append(
                    f"  (listing first {max_arousal_events} of {len(events)} detected events — "
                    f"call classify_arousal on a sub-range to see more)"
                )
                events = events[:max_arousal_events]
            for c in events:
                lines.append(
                    f"  {_format_anchor_time(c['start_s'])} - "
                    f"{_format_anchor_time(c['end_s'])}  "
                    f"{c['label']:<18s}  conf={c['conf']:.2f}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool execution (cached lookups into pre-pass results)
# ---------------------------------------------------------------------------


def _tool_classify(
    start_time: str, end_time: str,
    chunks: list[dict],
    chunk_sec: float,
    recording_duration_sec: float,
) -> str:
    try:
        s = _parse_anchor_time(start_time)
        e = _parse_anchor_time(end_time)
    except Exception as exc:
        return json.dumps({"error": f"bad timestamp: {exc}",
                           "segment": f"{start_time} - {end_time}"})
    if e < s:
        e += 24 * 3600
    if e < s:
        return json.dumps({"error": "end_time < start_time",
                           "segment": f"{start_time} - {end_time}"})
    max_sec = get_max_query_seconds(recording_duration_sec)
    if e - s > max_sec:
        return json.dumps({
            "error": f"Segment too long: {e - s:.1f}s exceeds maximum {max_sec:.1f}s.",
            "segment": f"{start_time} - {end_time}",
        })
    if s < 0 or e > recording_duration_sec + 1e-3:
        return json.dumps({
            "error": f"Segment outside recording window [0, {recording_duration_sec:.1f}s]",
            "segment": f"{start_time} - {end_time}",
        })
    hits = [
        {
            "start": _format_anchor_time(c["start_s"]),
            "end": _format_anchor_time(c["end_s"]),
            "activity": c["label"],
            "confidence": round(c["conf"], 3),
        }
        for c in chunks
        if c["end_s"] > s and c["start_s"] < e
    ]
    return json.dumps({
        "segment": f"{start_time} - {end_time}",
        "chunk_seconds": chunk_sec,
        "classifications": hits,
    })


# ---------------------------------------------------------------------------
# Per-sample agentic loop
# ---------------------------------------------------------------------------


def build_user_prompts(
    sample: dict,
    recording_duration_sec: float,
    context_text: str,
    tool_names: list[str],
    classifier_mode: str = "real",
) -> tuple[str, str]:
    rec_start = _format_anchor_time(0.0)
    rec_end = _format_anchor_time(recording_duration_sec)
    pre = (
        f"You are given a 13-channel polysomnography window for subject "
        f"{sample.get('subject_id')}. The window spans {rec_start} to {rec_end} "
        f"({recording_duration_sec:.0f} seconds at {EFFECTIVE_HZ} Hz).\n\n"
        f"{context_text}\n\n"
        f"Question: {sample['question']}"
    )
    post = get_instructions(recording_duration_sec, tool_names, classifier_mode=classifier_mode)
    return pre, post


def run_sample(
    sample: dict,
    idx: int,
    provider,
    label_class: str,
    device: str,
    max_tool_rounds: int,
    reasoning_effort: str,
    prepass_batch_size: int = 128,
    classifier_mode: str = "real",
) -> dict:
    window_start_ms = int(sample["window_start_ms"])
    window_end_ms = int(sample["window_end_ms"])
    recording_duration_sec = (window_end_ms - window_start_ms) / 1000.0
    task_type = sample.get("task_type", "")

    active_tools = tools_for_sample(label_class, task_type)
    needs_stages = any(t["name"] == "classify_sleep_stage" for t in active_tools)
    needs_arousals = any(t["name"] == "classify_arousal" for t in active_tools)
    # Oracle mode reasons over the text prompt alone — no tool access.
    llm_tools = [] if classifier_mode == "oracle" else active_tools

    try:
        pre_pass = run_pre_pass(
            sample, label_class,
            need_stages=needs_stages, need_arousals=needs_arousals,
            device=device, batch_size=prepass_batch_size,
            classifier_mode=classifier_mode,
        )
    except Exception as exc:
        return {"index": idx, "error": f"pre-pass failed: {exc}",
                "question": sample.get("question", "")}

    pre_pass_metrics = (
        {} if classifier_mode == "oracle" else compute_pre_pass_metrics(pre_pass)
    )
    context_text = render_context_file(
        pre_pass, recording_duration_sec, classifier_mode=classifier_mode,
    )

    tool_names = [t["name"] for t in llm_tools]
    pre, post = build_user_prompts(
        sample, recording_duration_sec, context_text, tool_names,
        classifier_mode=classifier_mode,
    )
    messages = provider.build_messages(SYSTEM_PROMPT, pre, post, None)

    input_char_lengths = {
        "system": len(SYSTEM_PROMPT),
        "pre_prompt": len(pre),
        "post_prompt": len(post),
        "context_text": len(context_text),
        "question": len(sample.get("question", "") or ""),
        "total": len(SYSTEM_PROMPT) + len(pre) + len(post),
    }

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
                messages, llm_tools, reasoning_effort, max_tokens=16384,
            )
        except Exception as exc:
            return {
                "index": idx,
                "question": sample.get("question", ""),
                "error": f"API error: {exc}",
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
            st = args.get("start_time", "")
            et = args.get("end_time", "")
            if tc.name == "classify_sleep_stage" and "stages" in pre_pass:
                tool_result = _tool_classify(
                    st, et, pre_pass["stages"], STAGE_CHUNK_SEC, recording_duration_sec,
                )
            elif tc.name == "classify_arousal" and "arousals" in pre_pass:
                tool_result = _tool_classify(
                    st, et, pre_pass["arousals"], AROUSAL_CHUNK_SEC, recording_duration_sec,
                )
            else:
                tool_result = json.dumps({
                    "error": f"tool {tc.name} has no pre-pass predictions "
                             "(was the corresponding classifier loaded?)"
                })
            try:
                if "error" in json.loads(tool_result):
                    tool_calls_error += 1
                else:
                    tool_calls_success += 1
            except Exception:
                tool_calls_error += 1
            provider.append_tool_result(messages, tc.call_id, tool_result)

    if rounds > max_tool_rounds and (
        not final_text_parts or "Answer:" not in "\n".join(final_text_parts)
    ):
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
        "task_type": task_type,
        "answer_type": answer_type,
        "context_length_s": sample.get("context_length_s", 0),
        "label_class": label_class,
        "classifier_mode": classifier_mode,
        "pre_pass": {
            "stages_bouts": pre_pass.get("stages_bouts"),
            "arousals_n_events": (
                len(pre_pass["arousals_events"])
                if "arousals_events" in pre_pass
                else (
                    sum(1 for c in pre_pass["arousals"] if c["label"] != "none")
                    if "arousals" in pre_pass else None
                )
            ),
            "stages_n_chunks": len(pre_pass.get("stages", [])) or None,
            "arousals_n_chunks": len(pre_pass.get("arousals", [])) or None,
        },
        "pre_pass_metrics": pre_pass_metrics,
        "input_char_lengths": input_char_lengths,
    }


# ---------------------------------------------------------------------------
# Parquet driver
# ---------------------------------------------------------------------------


def load_completed_indices(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    completed: set[int] = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                completed.add(json.loads(line)["index"])
            except Exception:
                continue
    return completed


def _init_running_totals() -> dict:
    return {
        "stages": {
            "confusion": {g: {p: 0 for p in SLEEP_STAGE_CLASS_NAMES}
                          for g in SLEEP_STAGE_CLASS_NAMES},
            "n_scored": 0,
        },
        "arousals": {
            "confusion": {g: {p: 0 for p in AROUSAL_CLASS_NAMES}
                          for g in AROUSAL_CLASS_NAMES},
            "n_scored": 0,
        },
        "answers": {"correct": 0, "total": 0},
        "input_chars": {"sum": 0, "n": 0, "max": 0, "min": None},
    }


def _update_running_totals(running: dict, result: dict) -> None:
    running["answers"]["total"] += 1
    if result.get("correct"):
        running["answers"]["correct"] += 1
    m = result.get("pre_pass_metrics") or {}
    for kind in ("stages", "arousals"):
        sub = m.get(kind) or {}
        if not sub:
            continue
        running[kind]["n_scored"] += int(sub.get("n_chunks_scored") or 0)
        for g, prow in (sub.get("confusion") or {}).items():
            if g not in running[kind]["confusion"]:
                continue
            for p, v in prow.items():
                if p in running[kind]["confusion"][g]:
                    running[kind]["confusion"][g][p] += int(v)
    icl = result.get("input_char_lengths") or {}
    total_chars = int(icl.get("total") or 0)
    if total_chars > 0:
        rc = running["input_chars"]
        rc["sum"] += total_chars
        rc["n"] += 1
        if total_chars > rc["max"]:
            rc["max"] = total_chars
        if rc["min"] is None or total_chars < rc["min"]:
            rc["min"] = total_chars


def _running_summary(running: dict) -> dict:
    """Return accuracy + macro F1 for each classifier stream plus answer acc."""
    out: dict = {}
    for kind, classes in (("stages", SLEEP_STAGE_CLASS_NAMES),
                          ("arousals", AROUSAL_CLASS_NAMES)):
        conf = running[kind]["confusion"]
        total = sum(conf[g][p] for g in classes for p in classes)
        if total == 0:
            out[kind] = None
            continue
        correct = sum(conf[g][g] for g in classes)
        f1s = []
        for c in classes:
            tp = conf[c][c]
            fp = sum(conf[g][c] for g in classes if g != c)
            fn = sum(conf[c][p] for p in classes if p != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            support = sum(conf[c][p] for p in classes)
            if support > 0:
                f1s.append(f1)
        out[kind] = {
            "n": total,
            "accuracy": correct / total,
            "macro_f1": float(np.mean(f1s)) if f1s else None,
        }
    ans = running["answers"]
    out["answers"] = {
        "n": ans["total"],
        "accuracy": (ans["correct"] / ans["total"]) if ans["total"] else None,
    }
    ic = running["input_chars"]
    out["input_chars"] = {
        "n": ic["n"],
        "mean": (ic["sum"] / ic["n"]) if ic["n"] else None,
        "max": ic["max"] if ic["n"] else None,
        "min": ic["min"],
    }
    return out


def _print_block_summary(running: dict, header: str) -> None:
    s = _running_summary(running)
    lines = [f"  [{header}]"]
    ans = s.get("answers") or {}
    if ans.get("n"):
        lines.append(f"    answer accuracy: {ans['accuracy']*100:.1f}% "
                     f"({running['answers']['correct']}/{ans['n']})")
    for kind, label in (("stages", "sleep_stages"), ("arousals", "arousals")):
        sub = s.get(kind)
        if not sub:
            continue
        lines.append(f"    {label}: acc={sub['accuracy']*100:.1f}%  "
                     f"macro_f1={sub['macro_f1']:.3f}  (n={sub['n']} chunks)")
    ic = s.get("input_chars") or {}
    if ic.get("n"):
        lines.append(
            f"    input chars: mean={ic['mean']:.0f}  "
            f"min={ic['min']}  max={ic['max']}  (n={ic['n']})"
        )
    print("\n".join(lines))


def process_parquet(
    parquet_path: Path,
    output_path: Path,
    provider,
    label_class: str,
    device: str,
    max_samples: Optional[int],
    max_workers: int,
    max_tool_rounds: int,
    reasoning_effort: str,
    prepass_batch_size: int = 128,
    classifier_mode: str = "real",
) -> dict:
    df = pd.read_parquet(parquet_path)
    if max_samples is not None:
        df = df.head(max_samples)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "trajectories.jsonl"
    completed = load_completed_indices(jsonl_path)
    todo = [i for i in range(len(df)) if i not in completed]
    stats = {"total": len(df), "success": 0, "failed": 0, "skipped": len(completed)}

    # Seed running totals with anything already on disk so the postfix reflects
    # the full accumulated state, not just the current partial run.
    running = _init_running_totals()
    if completed:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    _update_running_totals(running, json.loads(line))
                except Exception:
                    continue

    if not todo:
        stats["success"] = len(completed)
        _print_block_summary(running, f"already-complete {parquet_path.parent.name}")
        return stats

    write_lock = Lock()

    def _one(idx: int) -> dict:
        row = df.iloc[idx].to_dict()
        return run_sample(
            row, idx, provider, label_class, device,
            max_tool_rounds, reasoning_effort,
            prepass_batch_size=prepass_batch_size,
            classifier_mode=classifier_mode,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, i): i for i in todo}
        pbar = tqdm(as_completed(futs), total=len(todo), desc="  evaluating")
        for fut in pbar:
            try:
                result = fut.result()
                with write_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result, default=str) + "\n")
                    _update_running_totals(running, result)
                    live = _running_summary(running)
                stats["success"] += 1
                postfix: dict[str, str] = {}
                ans = live.get("answers") or {}
                if ans.get("accuracy") is not None:
                    postfix["ans"] = f"{ans['accuracy']*100:.0f}%"
                stg = live.get("stages")
                if stg:
                    postfix["stage_f1"] = f"{stg['macro_f1']:.2f}"
                    postfix["stage_acc"] = f"{stg['accuracy']*100:.0f}%"
                aro = live.get("arousals")
                if aro:
                    postfix["arous_f1"] = f"{aro['macro_f1']:.2f}"
                    postfix["arous_acc"] = f"{aro['accuracy']*100:.0f}%"
                ic = live.get("input_chars") or {}
                if ic.get("n"):
                    postfix["in_chars"] = f"{ic['mean']/1000:.1f}k"
                if postfix:
                    pbar.set_postfix(postfix)
            except Exception as exc:
                idx = futs[fut]
                print(f"  Error sample {idx}: {exc}")
                stats["failed"] += 1
        pbar.close()

    _print_block_summary(running, parquet_path.parent.name)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-class", choices=["sleep_stages", "arousals"], required=True)
    ap.add_argument("--classifier-mode", choices=["real", "oracle"], default="real",
                    help="'real' runs the SleepClassifier pre-pass (default). "
                         "'oracle' fills the timeline with ground-truth labels "
                         "at the same 30s/20s grid and disables tools — the LLM "
                         "reasons over the text prompt alone.")
    ap.add_argument("--stages-checkpoint", type=str, default=None,
                    help="Checkpoint for sleep_stages SleepClassifier "
                         "(required in real mode; ignored in oracle mode).")
    ap.add_argument("--arousals-checkpoint", type=str, default=None,
                    help="Checkpoint for arousals SleepClassifier "
                         "(required in real mode; ignored in oracle mode).")
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
    ap.add_argument("--prepass-batch-size", type=int, default=128,
                    help="Mini-batch size for the classifier pre-pass forward. "
                         "Lower this if you hit CUDA OOM on long windows "
                         "(7200/full). Auto-halves on OOM down to 1.")
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

    if args.classifier_mode == "real":
        # Both classifiers must be loaded for sleep_stages (state_query task uses
        # the arousals classifier as well); for arousals runs only the arousals
        # classifier is strictly required, but we allow loading stages too for
        # symmetry / future cross-tasks.
        need_stages = args.label_class == "sleep_stages" or bool(args.stages_checkpoint)
        need_arousals = True  # state_query may be present for sleep_stages; arousals always needs it.
        if need_stages and not args.stages_checkpoint:
            print("Error: --stages-checkpoint is required for real mode "
                  "(sleep_stages / state_query need it)")
            sys.exit(1)
        if need_arousals and not args.arousals_checkpoint:
            print("Error: --arousals-checkpoint is required for real mode "
                  "(needed for arousals tasks and sleep_stages/state_query)")
            sys.exit(1)

        import torch
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        load_classifiers(
            stages_checkpoint=args.stages_checkpoint if need_stages else None,
            arousals_checkpoint=args.arousals_checkpoint if need_arousals else None,
            device=device,
        )
    else:
        # Oracle mode: no classifiers, no raw-signal loading, no GPU needed.
        device = "cpu"
        print("Oracle mode: using ground-truth labels (no classifiers loaded).")

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    ctx_grid = args.context_lengths or [
        str(c) for c in ALL_CONTEXT_LENGTHS_PER_LABEL[args.label_class]
    ]
    normalized_ctx: list = []
    for c in ctx_grid:
        if c == "full":
            normalized_ctx.append("full")
        else:
            normalized_ctx.append(int(float(c)))

    base_dir = Path(args.input_dir) / args.label_class / "tasks"
    model_slug = args.model.replace("/", "_")
    suffix = [args.label_class, "prepass"]
    if args.classifier_mode == "oracle":
        suffix.append("oracle")
    else:
        ckpt = args.stages_checkpoint or args.arousals_checkpoint
        if ckpt:
            suffix.append(Path(ckpt).parent.name)
    output_root = Path(args.output_dir) / (model_slug + "_" + "_".join(suffix))

    print("=" * 60)
    print(f"Sleep PSG pre-pass GPT benchmark — provider={provider_name} model={args.model}")
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
            print(f"\n{ctx_dir}/{task}/{args.split}: {effective} samples, "
                  f"{tool_rounds} tool rounds")
            stats = process_parquet(
                parquet_path,
                output_root / ctx_dir / task / args.split,
                provider, args.label_class, device,
                args.max_samples, args.max_workers,
                tool_rounds, args.reasoning_effort,
                prepass_batch_size=args.prepass_batch_size,
                classifier_mode=args.classifier_mode,
            )
            for k in total:
                total[k] += stats[k]
            print(f"  success={stats['success']}, failed={stats['failed']}, "
                  f"skipped={stats['skipped']}")

    # Aggregate summary
    summary = {"by_task": {}, "by_context": {}, "overall": {}, "pre_pass": {}}
    rows: list[dict] = []
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
        summary["pre_pass"] = _aggregate_pre_pass(rows)

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    if summary.get("overall"):
        ov = summary["overall"]
        print(f"Overall: {ov['correct']}/{ov['total']} = {ov['accuracy'] * 100:.1f}%")
    pp = summary.get("pre_pass") or {}
    for kind, label in (("stages", "sleep_stages"), ("arousals", "arousals")):
        sub = pp.get(kind) or {}
        if sub.get("n_chunks_scored"):
            print(f"Pre-pass {label}: acc={sub['accuracy']:.3f}  "
                  f"macro_f1={sub['macro_f1']:.3f}  "
                  f"(n={sub['n_chunks_scored']} chunks)")
    print(f"\nTotal: success={total['success']}, failed={total['failed']}, "
          f"skipped={total['skipped']}")


if __name__ == "__main__":
    main()
