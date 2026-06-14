#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark LLMs on the LTAF-Haystack QA split using the trained LTAF classifiers
as a *pre-pass context provider* + agentic tool back-end.

Unlike ``benchmark_gpt_sleep.py`` / ``benchmark_gpt.py`` which hand a
``classify_bout`` tool to the agent and let it discover the recording through
tool calls alone, this script runs the full QA window through the LTAF
classifiers at their native windows *before* the LLM ever sees the question:

    * ``RhythmResNet1D`` — non-overlapping 10 s windows → a rhythm timeline
      (bouts of consecutive same-label chunks) covering the full QA window.
    * ``EcgBeatHTFClassifier`` — 2 s R-peak-centered windows over every beat
      in the record's beat timeline (from ``data/ltafdb/ltaf_haystack/
      beat_timelines/<rid>.parquet``); the label-history stream is fed
      autoregressively with predicted labels.

The two pre-pass outputs are formatted as a compact textual **context file**
and injected into the user prompt. The agent still has the same tools as
``benchmark_gpt_sleep.py`` (``classify_rhythm``, ``classify_beats``) so it can
drill into finer sub-ranges; the tool implementations are thin filters over
the cached pre-pass results.

Examples:
    # Real classifiers on rhythms/existence (100 s windows)
    python3 -m src.models.ts_llm.arts.ltaf \\
        --rhythm-checkpoint checkpoints/ltaf/rhythm_resnet1d/best_classifier.pt \\
        --beat-checkpoint checkpoints/ltaf/beats_htf/best_classifier.pt \\
        --context-lengths 100 --tasks existence --max-samples 5 --max-workers 1

    # Full grid, 900 s windows, test split
    python3 -m src.models.ts_llm.arts.ltaf \\
        --rhythm-checkpoint checkpoints/ltaf/rhythm_resnet1d/best_classifier.pt \\
        --beat-checkpoint checkpoints/ltaf/beats_htf/best_classifier.pt \\
        --context-lengths 900 --tasks all --split test

If the checkpoints are missing, first run:
    python3 scripts/download_classifiers.py --domain ecg
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

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Reuse provider adapters and response dataclasses from the main benchmark.
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
from src.datasets.constants import RAW_DATA
from src.datasets.ltaf_haystack.loader import (
    CHANNEL_NAMES,
    SOURCE_HZ,
    load_window_ms,
)
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.models.classifiers.ecg.beat_htf import BEAT_CLASS_NAMES, EcgBeatHTFClassifier
from src.models.classifiers.ecg.rhythm import (
    RHYTHM_CLASS_NAMES_6,
    RhythmResNet1D,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("results") / "gpt_ltaf_benchmark"
BASE_DIR = Path(RAW_DATA) / "ltafdb" / "ltaf_haystack" / "rhythms" / "tasks"
BEAT_TIMELINES_DIR = Path(RAW_DATA) / "ltafdb" / "ltaf_haystack" / "beat_timelines"
RHYTHM_TIMELINES_DIR = Path(RAW_DATA) / "ltafdb" / "ltaf_haystack" / "timelines"

RHYTHM_CHUNK_SEC = 10
BEAT_WINDOW_SEC = 2
BEAT_HALF_SAMPLES = BEAT_WINDOW_SEC * SOURCE_HZ // 2  # 128

ALL_TASKS = [
    "existence", "localization", "counting", "ordering",
    "state_query", "antecedent", "comparison", "multi_hop",
    "anomaly_detection", "anomaly_localization",
]
ALL_CONTEXT_LENGTHS = [100, 900, 3600, 7200]


def _format_ctx_dir(ctx: int | float | str) -> str:
    if isinstance(ctx, str):
        return ctx if ctx.endswith("s") else f"{ctx}s"
    return f"{float(ctx)}".replace(".", "_") + "s"


# ---------------------------------------------------------------------------
# Timestamp helpers — LTAF QA uses window-relative 24h "HH:MM:SS".
# ---------------------------------------------------------------------------


def _format_hms(seconds: float) -> str:
    s_total = max(0.0, float(seconds))
    h = int(s_total // 3600)
    m = int((s_total % 3600) // 60)
    s = s_total - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse_hms(ts: str) -> float:
    t = ts.strip().upper().replace("AM", "").replace("PM", "").strip()
    parts = t.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    f"You are an expert cardiologist analyzing 2-lead ECG recordings sampled at "
    f"{SOURCE_HZ} Hz. Channels: {', '.join(CHANNEL_NAMES)}.\n\n"
    f"Rhythm classes: {', '.join(RHYTHM_CLASS_NAMES_6)} "
    "(NSR=normal sinus, AFIB=atrial fibrillation, SBR=sinus bradycardia, "
    "AB=atrial bigeminy, SVTA=supraventricular tachyarrhythmia, B=ventricular bigeminy).\n"
    f"Beat classes: {', '.join(BEAT_CLASS_NAMES)} "
    "(N=normal, A=PAC/premature atrial, V=PVC/premature ventricular)."
)


_INSTRUCTIONS_TAIL = """\
- After consulting the classifier context and (optionally) tool results, provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting answer with just the number. \
For category answer with the class name. For timestamps use "HH:MM:SS.mmm" with \
millisecond precision; for time ranges use "HH:MM:SS.mmm-HH:MM:SS.mmm". Prefer the \
sub-second timestamps emitted by the classifier tools — rounding to whole seconds may \
fall outside the scoring tolerance."""


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
            "with just the number. For category answer with the class name. For "
            "timestamps use \"HH:MM:SS.mmm\"; for time ranges use "
            "\"HH:MM:SS.mmm-HH:MM:SS.mmm\". The timestamps in the timeline are exact."
        )
        head = (
            "Instructions:\n"
            "- A ground-truth analysis of the full recording is provided below "
            "(rhythm bouts and per-beat symbols, with exact timestamps).\n"
            "- Use the provided timeline directly to answer the question; "
            "the labels are authoritative.\n"
        )
        return f"{head}{oracle_tail}"
    tool_list = " and ".join(f"`{n}`" for n in tool_names)
    head = (
        "Instructions:\n"
        f"- A pre-computed classifier analysis of the full recording is provided below. "
        f"Consult it first.\n"
        f"- You may also call {tool_list} on any sub-range for a finer look; "
        "these tools return the same pre-computed predictions filtered to that range.\n"
        "- Classifier predictions may be wrong — cross-check against the question "
        "and prefer evidence from multiple chunks/beats when possible.\n"
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

_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "start_time": {"type": "string", "description": "Sub-range start in 'HH:MM:SS.mmm' format (window-relative); millisecond precision accepted"},
        "end_time": {"type": "string", "description": "Sub-range end in 'HH:MM:SS.mmm' format (window-relative); millisecond precision accepted"},
    },
    "required": ["start_time", "end_time"],
    "additionalProperties": False,
}


def get_rhythm_tool() -> dict:
    return {
        "type": "function",
        "name": "classify_rhythm",
        "description": (
            "Return the per-chunk rhythm predictions (10-second non-overlapping windows) "
            "inside the given sub-range, as produced by the RhythmResNet1D classifier in "
            "the pre-pass. Each entry has start/end timestamps, top predicted class, and "
            "softmax confidence."
        ),
        "parameters": _TOOL_PARAMS,
        "strict": True,
    }


def get_beats_tool() -> dict:
    return {
        "type": "function",
        "name": "classify_beats",
        "description": (
            "Return the per-beat predictions (N/A/V) for every R-peak inside the given "
            "sub-range, as produced by the EcgBeatHTF classifier in the pre-pass. Each "
            "entry has the beat timestamp, predicted class, and softmax confidence."
        ),
        "parameters": _TOOL_PARAMS,
        "strict": True,
    }


def tools_for_sample() -> list[dict]:
    return [get_rhythm_tool(), get_beats_tool()]


# ---------------------------------------------------------------------------
# Real classifiers — load once, thread-safe inference
# ---------------------------------------------------------------------------

_classifier_lock = Lock()
_rhythm_model: Optional[RhythmResNet1D] = None
_beat_model: Optional[EcgBeatHTFClassifier] = None
_beat_timeline_cache: dict[str, pd.DataFrame] = {}
_rhythm_timeline_cache: dict[str, pd.DataFrame] = {}


def load_classifiers(rhythm_checkpoint: str, beat_checkpoint: str, device: str) -> None:
    global _rhythm_model, _beat_model
    print(f"Loading RhythmResNet1D from {rhythm_checkpoint} on {device}...")
    _rhythm_model = RhythmResNet1D.load(rhythm_checkpoint, device=device)
    print(f"  classes: {_rhythm_model.class_names}  params: "
          f"{sum(p.numel() for p in _rhythm_model.parameters()):,}")
    print(f"Loading EcgBeatHTFClassifier from {beat_checkpoint} on {device}...")
    _beat_model = EcgBeatHTFClassifier.load(beat_checkpoint, device=device)
    print(f"  classes: {_beat_model.class_names}  history_k: {_beat_model.history_k}  "
          f"window_samples: {_beat_model.window_samples}  params: "
          f"{sum(p.numel() for p in _beat_model.parameters()):,}")


def _get_beat_timeline(record_id: str) -> pd.DataFrame:
    if record_id in _beat_timeline_cache:
        return _beat_timeline_cache[record_id]
    path = BEAT_TIMELINES_DIR / f"{record_id}.parquet"
    if not path.exists():
        _beat_timeline_cache[record_id] = pd.DataFrame(columns=["sample", "symbol"])
        return _beat_timeline_cache[record_id]
    df = pd.read_parquet(path, columns=["sample", "symbol"]).sort_values("sample")
    df = df.reset_index(drop=True)
    _beat_timeline_cache[record_id] = df
    return df


def _get_rhythm_timeline(record_id: str) -> pd.DataFrame:
    if record_id in _rhythm_timeline_cache:
        return _rhythm_timeline_cache[record_id]
    path = RHYTHM_TIMELINES_DIR / f"{record_id}.parquet"
    if not path.exists():
        _rhythm_timeline_cache[record_id] = pd.DataFrame(
            columns=["start_sample", "end_sample", "activity"]
        )
        return _rhythm_timeline_cache[record_id]
    df = pd.read_parquet(path, columns=["start_sample", "end_sample", "activity"])
    df = df.sort_values("start_sample").reset_index(drop=True)
    _rhythm_timeline_cache[record_id] = df
    return df


def _gt_rhythm_for_chunk(
    rhythm_df: pd.DataFrame,
    chunk_start_abs: int,
    chunk_end_abs: int,
) -> Optional[str]:
    """Return the rhythm label with the greatest overlap with this chunk,
    restricted to the 6-class vocabulary. None if no valid overlap."""
    if rhythm_df.empty:
        return None
    starts = rhythm_df["start_sample"].to_numpy()
    ends = rhythm_df["end_sample"].to_numpy()
    labels = rhythm_df["activity"].to_numpy()
    lo = np.maximum(starts, chunk_start_abs)
    hi = np.minimum(ends, chunk_end_abs)
    overlap = np.maximum(0, hi - lo)
    if overlap.sum() == 0:
        return None
    totals: dict[str, int] = {}
    for i in range(len(labels)):
        if overlap[i] <= 0:
            continue
        lbl = str(labels[i])
        if lbl not in RHYTHM_CLASS_NAMES_6:
            continue
        totals[lbl] = totals.get(lbl, 0) + int(overlap[i])
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Pre-pass — run the full window through the classifiers
# ---------------------------------------------------------------------------


def _zscore(chunk: np.ndarray) -> np.ndarray:
    mean = chunk.mean(axis=-1, keepdims=True)
    std = chunk.std(axis=-1, keepdims=True)
    return ((chunk - mean) / (std + 1e-6)).astype(np.float32, copy=False)


def _pre_pass_rhythm(
    signal: np.ndarray,
    device: str,
    record_id: Optional[str] = None,
    window_start_sample: int = 0,
    batch_size: int = 64,
) -> list[dict]:
    """Classify non-overlapping 10 s chunks of the window signal.

    signal: (L, 2) float32 at SOURCE_HZ. If ``record_id`` is provided, each
    chunk is annotated with the ground-truth rhythm label from the record's
    rhythm timeline (for pre-pass accuracy metrics).

    Inference is split into mini-batches of ``batch_size`` chunks to keep GPU
    memory bounded — important for long windows (7200 s → 720 chunks). If a
    sub-batch OOMs we halve and retry until we fall back to one-at-a-time.
    """
    import torch
    import torch.nn.functional as F

    L = signal.shape[0]
    chunk_samples = RHYTHM_CHUNK_SEC * SOURCE_HZ
    if L < chunk_samples:
        return []

    n_chunks = L // chunk_samples
    chunks = np.empty((n_chunks, 2, chunk_samples), dtype=np.float32)
    for i in range(n_chunks):
        s = i * chunk_samples
        e = s + chunk_samples
        c = np.ascontiguousarray(signal[s:e].T)  # (2, L)
        chunks[i] = _zscore(c)

    probs = np.empty((n_chunks, _rhythm_model.num_classes), dtype=np.float32)
    with _classifier_lock:
        with torch.no_grad():
            i = 0
            bs = max(1, int(batch_size))
            while i < n_chunks:
                j = min(i + bs, n_chunks)
                try:
                    x = torch.from_numpy(chunks[i:j]).to(device)
                    logits = _rhythm_model(x)
                    probs[i:j] = F.softmax(logits, dim=-1).cpu().numpy()
                    del x, logits
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    continue
                i = j
    labels = probs.argmax(-1)
    classes = _rhythm_model.class_names

    rhythm_df = _get_rhythm_timeline(record_id) if record_id is not None else pd.DataFrame()

    out: list[dict] = []
    for i in range(n_chunks):
        chunk_start_abs = window_start_sample + i * chunk_samples
        chunk_end_abs = chunk_start_abs + chunk_samples
        gt = _gt_rhythm_for_chunk(rhythm_df, chunk_start_abs, chunk_end_abs) \
            if not rhythm_df.empty else None
        out.append({
            "start_s": i * RHYTHM_CHUNK_SEC,
            "end_s": (i + 1) * RHYTHM_CHUNK_SEC,
            "label": classes[int(labels[i])],
            "conf": float(probs[i, labels[i]]),
            "gt": gt,
        })
    return out


def _compress_rhythm_bouts(chunks: list[dict]) -> list[dict]:
    """Merge adjacent same-label chunks into bouts with mean confidence."""
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
            cur_start, cur_end, cur_label, confs = c["start_s"], c["end_s"], c["label"], [c["conf"]]
    bouts.append({
        "start_s": cur_start, "end_s": cur_end, "label": cur_label,
        "conf": float(np.mean(confs)), "n_chunks": len(confs),
    })
    return bouts


def _oracle_rhythm_bouts(
    record_id: str,
    window_start_sample: int,
    window_end_sample: int,
) -> list[dict]:
    """Return the real rhythm-timeline bouts that overlap the window, clipped
    to the window bounds and shifted to window-local seconds. No chunking,
    no majority-overlap approximation — each entry is an actual annotation
    bout with its real activity label and timestamps."""
    df = _get_rhythm_timeline(record_id)
    if df.empty:
        return []
    starts = df["start_sample"].to_numpy()
    ends = df["end_sample"].to_numpy()
    activities = df["activity"].to_numpy()
    out: list[dict] = []
    for s, e, a in zip(starts, ends, activities):
        clip_s = max(int(s), window_start_sample)
        clip_e = min(int(e), window_end_sample)
        if clip_e <= clip_s:
            continue
        label = str(a)
        out.append({
            "start_s": (clip_s - window_start_sample) / SOURCE_HZ,
            "end_s": (clip_e - window_start_sample) / SOURCE_HZ,
            "label": label,
            "conf": 1.0,
            "gt": label,
        })
    out.sort(key=lambda b: b["start_s"])
    return out


def _oracle_beat_predictions(
    record_id: str,
    window_start_sample: int,
    window_end_sample: int,
) -> list[dict]:
    """Return every annotated beat whose R-peak sample falls inside the window,
    with its real symbol as the label. No 2s-fit edge mask — GT labels exist
    regardless of whether a classifier input window would fit."""
    df = _get_beat_timeline(record_id)
    if df.empty:
        return []
    samples = df["sample"].to_numpy(dtype=np.int64)
    symbols = df["symbol"].to_numpy()
    mask = (samples >= window_start_sample) & (samples < window_end_sample)
    out: list[dict] = []
    for pos in np.flatnonzero(mask):
        sym = str(symbols[pos])
        time_s = (int(samples[pos]) - window_start_sample) / SOURCE_HZ
        out.append({
            "sample_abs": int(samples[pos]),
            "time_s": float(time_s),
            "label": sym,
            "conf": 1.0,
            "gt": sym,
        })
    return out


def _pre_pass_beats(
    record_id: str,
    window_start_sample: int,
    window_end_sample: int,
    device: str,
    batch_size: int = 64,
) -> list[dict]:
    """Classify every beat whose 2 s R-peak window lies inside the QA window.

    Uses the record's full beat timeline so RR history reflects prior beats
    (including any before the QA window). Label history is autoregressive —
    predictions from prior in-window beats feed forward; beats outside the
    window use a neutral -1 placeholder.
    """
    import torch
    import torch.nn.functional as F

    df = _get_beat_timeline(record_id)
    if df.empty:
        return []

    samples = df["sample"].to_numpy(dtype=np.int64)
    in_window_mask = (samples >= window_start_sample + BEAT_HALF_SAMPLES) & \
                     (samples + BEAT_HALF_SAMPLES <= window_end_sample)
    target_idx = np.flatnonzero(in_window_mask)
    if target_idx.size == 0:
        return []

    history_k = _beat_model.history_k
    num_classes = _beat_model.num_classes
    use_labels = _beat_model.history_use_labels

    # Predicted label per beat position in `df` (aligned with `samples`).
    # -1 means unknown; we only fill positions we actually classify.
    pred_labels = np.full(len(samples), -1, dtype=np.int64)

    window_sig_len = window_end_sample - window_start_sample
    predictions: list[dict] = []

    for batch_start in range(0, target_idx.size, batch_size):
        batch_pos = target_idx[batch_start: batch_start + batch_size]

        x_list: list[np.ndarray] = []
        rr_list: list[np.ndarray] = []
        lbl_list: list[np.ndarray] = []
        kept_pos: list[int] = []

        for pos in batch_pos:
            center = int(samples[pos])
            abs_s = center - BEAT_HALF_SAMPLES
            abs_e = center + BEAT_HALF_SAMPLES
            # Slice from the record's memmap directly (load_window_ms would
            # shift timestamps; here we want absolute record samples).
            from src.datasets.ltaf_haystack.loader import load_bout_signal
            try:
                sig = load_bout_signal(record_id, abs_s, abs_e)  # (L, 2)
            except Exception:
                continue
            if sig.shape[0] != 2 * BEAT_HALF_SAMPLES:
                continue
            sig_t = np.ascontiguousarray(sig.T)  # (2, L)
            sig_t = _zscore(sig_t)

            # RR history — use real prior beats from the record timeline.
            rr = np.zeros(history_k, dtype=np.float32)
            lbl = np.full(history_k, -1, dtype=np.int64)
            for k in range(history_k):
                prev_pos = pos - 1 - k
                if prev_pos < 0:
                    break
                nxt_pos = prev_pos + 1
                rr[k] = (int(samples[nxt_pos]) - int(samples[prev_pos])) / SOURCE_HZ
                # Autoregressive label: use our own prediction if available.
                lbl[k] = int(pred_labels[prev_pos])
            x_list.append(sig_t)
            rr_list.append(rr)
            lbl_list.append(lbl)
            kept_pos.append(int(pos))

        if not x_list:
            continue

        x_arr = np.stack(x_list, axis=0)
        rr_arr = np.stack(rr_list, axis=0)
        lbl_arr = np.stack(lbl_list, axis=0)

        n_kept = len(kept_pos)
        probs = np.empty((n_kept, num_classes), dtype=np.float32)
        with _classifier_lock:
            with torch.no_grad():
                i = 0
                bs = max(1, int(batch_size))
                while i < n_kept:
                    j = min(i + bs, n_kept)
                    try:
                        x_t = torch.from_numpy(x_arr[i:j]).float().to(device)
                        rr_t = torch.from_numpy(rr_arr[i:j]).float().to(device)
                        lbl_t = torch.from_numpy(lbl_arr[i:j]).long().to(device)
                        logits = _beat_model(x_t, rr_t, lbl_t if use_labels else None)
                        probs[i:j] = F.softmax(logits, dim=-1).cpu().numpy()
                        del x_t, rr_t, lbl_t, logits
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if bs == 1:
                            raise
                        bs = max(1, bs // 2)
                        continue
                    i = j

        preds = probs.argmax(-1)
        symbols = df["symbol"].to_numpy()
        for k, pos in enumerate(kept_pos):
            pred_labels[pos] = int(preds[k])
            time_s = (int(samples[pos]) - window_start_sample) / SOURCE_HZ
            gt_sym = str(symbols[pos])
            predictions.append({
                "sample_abs": int(samples[pos]),
                "time_s": float(time_s),
                "label": _beat_model.class_names[int(preds[k])],
                "conf": float(probs[k, preds[k]]),
                "gt": gt_sym if gt_sym in BEAT_CLASS_NAMES else None,
            })

    predictions.sort(key=lambda r: r["time_s"])
    return predictions


def run_pre_pass(
    sample: dict,
    device: str,
    batch_size: int = 64,
    classifier_mode: str = "real",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Full pre-pass. Returns (rhythm_chunks, rhythm_bouts, beat_predictions).

    In ``real`` mode: run classifiers at their native resolutions.
    In ``oracle`` mode: surface the raw annotation timelines directly — no
    classifier forward, no chunking, no raw-signal loading. ``rhythm_chunks``
    aliases ``rhythm_bouts`` since there is no underlying chunk grid.
    """
    record_id = str(sample["record_id"])
    window_start_ms = int(sample["window_start_ms"])
    window_end_ms = int(sample["window_end_ms"])
    source_hz = int(sample.get("source_hz", SOURCE_HZ))

    window_start_sample = int(round(window_start_ms / 1000.0 * source_hz))
    window_end_sample = int(round(window_end_ms / 1000.0 * source_hz))

    if classifier_mode == "oracle":
        rhythm_bouts = _oracle_rhythm_bouts(
            record_id, window_start_sample, window_end_sample,
        )
        rhythm_chunks = rhythm_bouts
        beat_predictions = _oracle_beat_predictions(
            record_id, window_start_sample, window_end_sample,
        )
        return rhythm_chunks, rhythm_bouts, beat_predictions

    signal = load_window_ms(record_id, window_start_ms, window_end_ms, source_hz=source_hz)
    rhythm_chunks = _pre_pass_rhythm(
        signal, device, record_id=record_id,
        window_start_sample=window_start_sample, batch_size=batch_size,
    )
    rhythm_bouts = _compress_rhythm_bouts(rhythm_chunks)
    beat_predictions = _pre_pass_beats(
        record_id, window_start_sample, window_end_sample, device,
        batch_size=batch_size,
    )
    return rhythm_chunks, rhythm_bouts, beat_predictions


# ---------------------------------------------------------------------------
# Context-file rendering
# ---------------------------------------------------------------------------


def render_context_file(
    rhythm_bouts: list[dict],
    beat_predictions: list[dict],
    window_duration_sec: float,
    max_anomalous_beats: int = 200,
    classifier_mode: str = "real",
) -> str:
    is_oracle = classifier_mode == "oracle"
    header_label = "Ground-truth" if is_oracle else "Classifier"
    rhythm_source = (
        "ground truth, native annotation resolution"
        if is_oracle else f"RhythmResNet1D, {RHYTHM_CHUNK_SEC}s resolution"
    )
    beat_source = (
        "ground truth, per R-peak" if is_oracle
        else "EcgBeatHTF, per R-peak"
    )

    lines: list[str] = []
    lines.append(
        f"# {header_label} pre-pass summary (window 00:00:00 - "
        f"{_format_hms(window_duration_sec)})"
    )

    lines.append("")
    lines.append(
        f"## Rhythm timeline ({rhythm_source}, {len(rhythm_bouts)} bouts)"
    )
    if not rhythm_bouts:
        empty_msg = (
            "  (no rhythm annotations in this window)" if is_oracle
            else "  (no rhythm predictions — window shorter than one chunk)"
        )
        lines.append(empty_msg)
    for b in rhythm_bouts:
        dur = b["end_s"] - b["start_s"]
        if is_oracle:
            lines.append(
                f"  {_format_hms(b['start_s'])} - {_format_hms(b['end_s'])}  "
                f"{b['label']:<5s}  ({dur:.1f}s)"
            )
        else:
            lines.append(
                f"  {_format_hms(b['start_s'])} - {_format_hms(b['end_s'])}  "
                f"{b['label']:<5s}  conf={b['conf']:.2f}  "
                f"({b['n_chunks']} chunk{'s' if b['n_chunks'] != 1 else ''}, {dur:.0f}s)"
            )

    # Beats: list only non-normal beats. Report counts for whatever symbols are
    # present so oracle mode can surface e.g. 'Q', '+' when real annotations
    # contain them.
    n_normal = sum(1 for b in beat_predictions if b["label"] == "N")
    anomalous = [b for b in beat_predictions if b["label"] != "N"]
    lines.append("")
    if is_oracle:
        from collections import Counter
        counts = Counter(b["label"] for b in beat_predictions)
        # Show N first, then the rest by frequency.
        other = sorted(
            ((lbl, n) for lbl, n in counts.items() if lbl != "N"),
            key=lambda kv: (-kv[1], kv[0]),
        )
        count_str = "  ".join(
            [f"N={n_normal}"] + [f"{lbl}={n}" for lbl, n in other]
        )
        lines.append(
            f"## Beat timeline ({beat_source}): "
            f"{len(beat_predictions)} beats  | {count_str}"
        )
    else:
        lines.append(
            f"## Beat classifier ({beat_source}): "
            f"{len(beat_predictions)} beats analyzed  | "
            f"N={n_normal}  A(PAC)={sum(1 for b in anomalous if b['label']=='A')}  "
            f"V(PVC)={sum(1 for b in anomalous if b['label']=='V')}"
        )
    if not anomalous:
        noun = "non-normal beats" if is_oracle else "premature atrial or ventricular beats detected"
        lines.append(f"  (no {noun})")
    else:
        if len(anomalous) > max_anomalous_beats:
            if is_oracle:
                lines.append(
                    f"  (listing first {max_anomalous_beats} of {len(anomalous)} non-normal beats)"
                )
            else:
                lines.append(
                    f"  (listing first {max_anomalous_beats} of {len(anomalous)} anomalous beats — "
                    f"call classify_beats on a sub-range to see more)"
                )
            anomalous = anomalous[:max_anomalous_beats]
        for b in anomalous:
            if is_oracle:
                lines.append(f"  {_format_hms(b['time_s'])}  {b['label']}")
            else:
                lines.append(
                    f"  {_format_hms(b['time_s'])}  {b['label']}  conf={b['conf']:.2f}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-pass metrics — classifier output vs. ground truth
# ---------------------------------------------------------------------------


def _per_class_f1(preds: list[str], gts: list[str], classes: list[str]) -> dict[str, float]:
    """Macro per-class F1. Expects preds/gts over the same index set and every
    label in ``classes``. Returns a dict ``{class: f1}``."""
    out: dict[str, float] = {}
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = f1
    return out


def compute_pre_pass_metrics(
    rhythm_chunks: list[dict],
    beat_predictions: list[dict],
) -> dict:
    """Compute rhythm/beat pre-pass accuracy + macro F1 against ground truth.

    Only chunks/beats with a ground-truth label inside the classifier's class
    vocabulary are counted (everything else — e.g. rhythm classes outside the
    6-class subset, or beat symbols like 'Q' — is excluded).
    """
    r_preds = [c["label"] for c in rhythm_chunks if c.get("gt") is not None]
    r_gts = [c["gt"] for c in rhythm_chunks if c.get("gt") is not None]
    if r_preds:
        r_acc = sum(1 for p, g in zip(r_preds, r_gts) if p == g) / len(r_preds)
        r_per = _per_class_f1(r_preds, r_gts, RHYTHM_CLASS_NAMES_6)
        r_macro = float(np.mean(list(r_per.values())))
    else:
        r_acc = None
        r_per = {}
        r_macro = None

    b_preds = [b["label"] for b in beat_predictions if b.get("gt") is not None]
    b_gts = [b["gt"] for b in beat_predictions if b.get("gt") is not None]
    if b_preds:
        b_acc = sum(1 for p, g in zip(b_preds, b_gts) if p == g) / len(b_preds)
        b_per = _per_class_f1(b_preds, b_gts, BEAT_CLASS_NAMES)
        b_macro = float(np.mean(list(b_per.values())))
    else:
        b_acc = None
        b_per = {}
        b_macro = None

    return {
        "rhythm": {
            "n_chunks_scored": len(r_preds),
            "n_chunks_total": len(rhythm_chunks),
            "accuracy": r_acc,
            "macro_f1": r_macro,
            "per_class_f1": r_per,
            "support": {c: r_gts.count(c) for c in RHYTHM_CLASS_NAMES_6},
            "confusion": _confusion(r_preds, r_gts, RHYTHM_CLASS_NAMES_6),
        },
        "beats": {
            "n_beats_scored": len(b_preds),
            "n_beats_total": len(beat_predictions),
            "accuracy": b_acc,
            "macro_f1": b_macro,
            "per_class_f1": b_per,
            "support": {c: b_gts.count(c) for c in BEAT_CLASS_NAMES},
            "confusion": _confusion(b_preds, b_gts, BEAT_CLASS_NAMES),
        },
    }


def _confusion(preds: list[str], gts: list[str], classes: list[str]) -> dict:
    """Return confusion matrix as nested dict {gt: {pred: count}}."""
    m: dict[str, dict[str, int]] = {g: {p: 0 for p in classes} for g in classes}
    for p, g in zip(preds, gts):
        if g in m and p in m[g]:
            m[g][p] += 1
    return m


def _aggregate_pre_pass(rows: list[dict]) -> dict:
    """Pool per-sample confusion matrices into a single rhythm/beat metric set.

    Using confusion counts (not averaged per-sample F1) avoids inflation from
    samples with only 1 or 2 scored chunks/beats.
    """
    agg = {
        "rhythm": {
            "n_chunks_scored": 0,
            "support": {c: 0 for c in RHYTHM_CLASS_NAMES_6},
            "confusion": {g: {p: 0 for p in RHYTHM_CLASS_NAMES_6}
                          for g in RHYTHM_CLASS_NAMES_6},
        },
        "beats": {
            "n_beats_scored": 0,
            "support": {c: 0 for c in BEAT_CLASS_NAMES},
            "confusion": {g: {p: 0 for p in BEAT_CLASS_NAMES}
                          for g in BEAT_CLASS_NAMES},
        },
    }
    for r in rows:
        m = r.get("pre_pass_metrics") or {}
        for kind, classes in (("rhythm", RHYTHM_CLASS_NAMES_6),
                              ("beats", BEAT_CLASS_NAMES)):
            sub = m.get(kind) or {}
            agg[kind]["n_chunks_scored" if kind == "rhythm" else "n_beats_scored"] += int(
                sub.get("n_chunks_scored" if kind == "rhythm" else "n_beats_scored", 0) or 0
            )
            for c, v in (sub.get("support") or {}).items():
                if c in agg[kind]["support"]:
                    agg[kind]["support"][c] += int(v)
            for g, prow in (sub.get("confusion") or {}).items():
                if g not in agg[kind]["confusion"]:
                    continue
                for p, v in prow.items():
                    if p in agg[kind]["confusion"][g]:
                        agg[kind]["confusion"][g][p] += int(v)

    for kind, classes in (("rhythm", RHYTHM_CLASS_NAMES_6),
                          ("beats", BEAT_CLASS_NAMES)):
        conf = agg[kind]["confusion"]
        total = sum(conf[g][p] for g in classes for p in classes)
        correct = sum(conf[g][g] for g in classes)
        agg[kind]["accuracy"] = (correct / total) if total else None

        f1s = []
        per_class: dict[str, float] = {}
        for c in classes:
            tp = conf[c][c]
            fp = sum(conf[g][c] for g in classes if g != c)
            fn = sum(conf[c][p] for p in classes if p != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[c] = f1
            if agg[kind]["support"].get(c, 0) > 0:
                f1s.append(f1)
        agg[kind]["per_class_f1"] = per_class
        agg[kind]["macro_f1"] = (float(np.mean(f1s)) if f1s else None)
    return agg


# ---------------------------------------------------------------------------
# Tool execution (cached lookups into pre-pass results)
# ---------------------------------------------------------------------------


def _tool_classify_rhythm(
    start_time: str, end_time: str,
    rhythm_chunks: list[dict],
    recording_duration_sec: float,
) -> str:
    try:
        s = _parse_hms(start_time)
        e = _parse_hms(end_time)
    except Exception as exc:
        return json.dumps({"error": f"bad timestamp: {exc}",
                           "segment": f"{start_time} - {end_time}"})
    if e < s:
        return json.dumps({"error": "end_time < start_time",
                           "segment": f"{start_time} - {end_time}"})
    max_sec = get_max_query_seconds(recording_duration_sec)
    if e - s > max_sec:
        return json.dumps({
            "error": f"Segment too long: {e - s:.1f}s exceeds maximum {max_sec:.1f}s.",
            "segment": f"{start_time} - {end_time}",
        })
    hits = [
        {
            "start": _format_hms(c["start_s"]),
            "end": _format_hms(c["end_s"]),
            "activity": c["label"],
            "confidence": round(c["conf"], 3),
        }
        for c in rhythm_chunks
        if c["end_s"] > s and c["start_s"] < e
    ]
    return json.dumps({
        "segment": f"{start_time} - {end_time}",
        "chunk_seconds": RHYTHM_CHUNK_SEC,
        "classifications": hits,
    })


def _tool_classify_beats(
    start_time: str, end_time: str,
    beat_predictions: list[dict],
    recording_duration_sec: float,
    max_beats: int = 300,
) -> str:
    try:
        s = _parse_hms(start_time)
        e = _parse_hms(end_time)
    except Exception as exc:
        return json.dumps({"error": f"bad timestamp: {exc}",
                           "segment": f"{start_time} - {end_time}"})
    if e < s:
        return json.dumps({"error": "end_time < start_time",
                           "segment": f"{start_time} - {end_time}"})
    max_sec = get_max_query_seconds(recording_duration_sec)
    if e - s > max_sec:
        return json.dumps({
            "error": f"Segment too long: {e - s:.1f}s exceeds maximum {max_sec:.1f}s.",
            "segment": f"{start_time} - {end_time}",
        })
    beats_in = [b for b in beat_predictions if s <= b["time_s"] < e]
    truncated = False
    if len(beats_in) > max_beats:
        beats_in = beats_in[:max_beats]
        truncated = True
    out = {
        "segment": f"{start_time} - {end_time}",
        "n_beats": len(beats_in),
        "beats": [
            {"time": _format_hms(b["time_s"]), "label": b["label"],
             "confidence": round(b["conf"], 3)}
            for b in beats_in
        ],
    }
    if truncated:
        out["truncated"] = f"first {max_beats} beats only — call on a shorter range for more"
    return json.dumps(out)


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
    rec_end = _format_hms(recording_duration_sec)
    pre = (
        f"You are given a 2-lead ECG recording from subject {sample.get('record_id')}. "
        f"The window spans 00:00:00 to {rec_end} ({recording_duration_sec:.0f} seconds "
        f"at {SOURCE_HZ} Hz).\n\n"
        f"{context_text}\n\n"
        f"Question: {sample['question']}"
    )
    post = get_instructions(recording_duration_sec, tool_names, classifier_mode=classifier_mode)
    return pre, post


def run_sample(
    sample: dict,
    idx: int,
    provider,
    device: str,
    max_tool_rounds: int,
    reasoning_effort: str,
    prepass_batch_size: int = 64,
    classifier_mode: str = "real",
) -> dict:
    window_start_ms = int(sample["window_start_ms"])
    window_end_ms = int(sample["window_end_ms"])
    recording_duration_sec = (window_end_ms - window_start_ms) / 1000.0

    try:
        rhythm_chunks, rhythm_bouts, beat_predictions = run_pre_pass(
            sample, device, batch_size=prepass_batch_size,
            classifier_mode=classifier_mode,
        )
    except Exception as exc:
        return {"index": idx, "error": f"pre-pass failed: {exc}",
                "question": sample.get("question", "")}

    pre_pass_metrics = (
        {} if classifier_mode == "oracle"
        else compute_pre_pass_metrics(rhythm_chunks, beat_predictions)
    )
    context_text = render_context_file(
        rhythm_bouts, beat_predictions, recording_duration_sec,
        classifier_mode=classifier_mode,
    )

    active_tools = tools_for_sample()
    # Oracle mode reasons over the text prompt alone — no tool access.
    llm_tools = [] if classifier_mode == "oracle" else active_tools
    tool_names = [t["name"] for t in llm_tools]
    pre, post = build_user_prompts(
        sample, recording_duration_sec, context_text, tool_names,
        classifier_mode=classifier_mode,
    )
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
            if tc.name == "classify_rhythm":
                tool_result = _tool_classify_rhythm(st, et, rhythm_chunks, recording_duration_sec)
            elif tc.name == "classify_beats":
                tool_result = _tool_classify_beats(st, et, beat_predictions, recording_duration_sec)
            else:
                tool_result = json.dumps({"error": f"unknown tool {tc.name}"})
            try:
                if "error" in json.loads(tool_result):
                    tool_calls_error += 1
                else:
                    tool_calls_success += 1
            except Exception:
                tool_calls_error += 1
            provider.append_tool_result(messages, tc.call_id, tool_result)

    if rounds > max_tool_rounds and (not final_text_parts or
                                     "Answer:" not in "\n".join(final_text_parts)):
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
        "record_id": sample.get("record_id"),
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
        "context_length_samples": int(sample.get("context_length_samples", 0)),
        "classifier_mode": classifier_mode,
        "pre_pass": {
            "rhythm_bouts": rhythm_bouts,
            "n_beats": len(beat_predictions),
            "n_anomalous_beats": sum(1 for b in beat_predictions if b["label"] != "N"),
        },
        "pre_pass_metrics": pre_pass_metrics,
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


# ---------------------------------------------------------------------------
# Live running totals (printed via tqdm.set_postfix during eval)
# ---------------------------------------------------------------------------


def _init_running_totals() -> dict:
    return {
        "rhythm": {
            "confusion": {g: {p: 0 for p in RHYTHM_CLASS_NAMES_6}
                          for g in RHYTHM_CLASS_NAMES_6},
            "n_scored": 0,
        },
        "beats": {
            "confusion": {g: {p: 0 for p in BEAT_CLASS_NAMES}
                          for g in BEAT_CLASS_NAMES},
            "n_scored": 0,
        },
        "answers": {"correct": 0, "total": 0},
    }


def _update_running_totals(running: dict, result: dict) -> None:
    running["answers"]["total"] += 1
    if result.get("correct"):
        running["answers"]["correct"] += 1
    m = result.get("pre_pass_metrics") or {}
    for kind, n_key in (("rhythm", "n_chunks_scored"), ("beats", "n_beats_scored")):
        sub = m.get(kind) or {}
        if not sub:
            continue
        running[kind]["n_scored"] += int(sub.get(n_key) or 0)
        for g, prow in (sub.get("confusion") or {}).items():
            if g not in running[kind]["confusion"]:
                continue
            for p, v in prow.items():
                if p in running[kind]["confusion"][g]:
                    running[kind]["confusion"][g][p] += int(v)


def _running_summary(running: dict) -> dict:
    """Return accuracy + macro F1 for each classifier stream plus answer acc."""
    out: dict = {}
    for kind, classes in (("rhythm", RHYTHM_CLASS_NAMES_6),
                          ("beats", BEAT_CLASS_NAMES)):
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
    return out


def _print_block_summary(running: dict, header: str) -> None:
    s = _running_summary(running)
    lines = [f"  [{header}]"]
    ans = s.get("answers") or {}
    if ans.get("n"):
        lines.append(f"    answer accuracy: {ans['accuracy']*100:.1f}% "
                     f"({running['answers']['correct']}/{ans['n']})")
    for kind, label in (("rhythm", "rhythm"), ("beats", "beats")):
        sub = s.get(kind)
        if not sub:
            continue
        lines.append(f"    {label}: acc={sub['accuracy']*100:.1f}%  "
                     f"macro_f1={sub['macro_f1']:.3f}  (n={sub['n']})")
    print("\n".join(lines))


def process_parquet(
    parquet_path: Path,
    output_path: Path,
    provider,
    device: str,
    max_samples: Optional[int],
    max_workers: int,
    max_tool_rounds: int,
    reasoning_effort: str,
    prepass_batch_size: int = 64,
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
            row, idx, provider, device, max_tool_rounds, reasoning_effort,
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
                rhy = live.get("rhythm")
                if rhy:
                    postfix["rhy_f1"] = f"{rhy['macro_f1']:.2f}"
                    postfix["rhy_acc"] = f"{rhy['accuracy']*100:.0f}%"
                bts = live.get("beats")
                if bts:
                    postfix["beat_f1"] = f"{bts['macro_f1']:.2f}"
                    postfix["beat_acc"] = f"{bts['accuracy']*100:.0f}%"
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
    ap.add_argument("--classifier-mode", choices=["real", "oracle"], default="real",
                    help="'real' runs the rhythm + beat classifiers (default). "
                         "'oracle' surfaces the raw annotation timelines directly "
                         "(variable-width rhythm bouts + every annotated beat, "
                         "with actual labels) and disables tools — the LLM "
                         "reasons over the text prompt alone.")
    ap.add_argument("--rhythm-checkpoint", type=str, default=None,
                    help="Path to RhythmResNet1D best_classifier.pt "
                         "(required in real mode; ignored in oracle mode).")
    ap.add_argument("--beat-checkpoint", type=str, default=None,
                    help="Path to EcgBeatHTFClassifier best_classifier.pt "
                         "(required in real mode; ignored in oracle mode).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input-dir", type=str, default=str(BASE_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--context-lengths", type=str, nargs="+", default=None,
                    help=f"Context length grid (e.g. 100 900 3600 7200). "
                         f"Defaults to {ALL_CONTEXT_LENGTHS}.")
    ap.add_argument("--tasks", type=str, nargs="+", default=["all"])
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "validation", "test"])
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
    ap.add_argument("--prepass-batch-size", type=int, default=64,
                    help="Mini-batch size for the classifier pre-pass forward. "
                         "Lower this if you hit CUDA OOM on long windows "
                         "(3600/7200). Auto-halves on OOM down to 1.")
    args = ap.parse_args()

    # Provider selection (same logic as benchmark_gpt_sleep.py)
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
        if not args.rhythm_checkpoint or not args.beat_checkpoint:
            print("Error: --rhythm-checkpoint and --beat-checkpoint are required "
                  "for --classifier-mode real")
            sys.exit(1)
        import torch
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        load_classifiers(args.rhythm_checkpoint, args.beat_checkpoint, device)
    else:
        # Oracle mode: no classifiers, no raw-signal loading, no GPU needed.
        device = "cpu"
        print("Oracle mode: using ground-truth rhythm + beat timelines (no classifiers loaded).")

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    ctx_grid = args.context_lengths or [str(c) for c in ALL_CONTEXT_LENGTHS]
    normalized_ctx: list[int | float | str] = []
    for c in ctx_grid:
        try:
            normalized_ctx.append(float(c))
        except ValueError:
            normalized_ctx.append(c)

    base_dir = Path(args.input_dir)
    split_dir = "validation" if args.split == "val" else args.split
    model_slug = args.model.replace("/", "_")
    suffix = "ltaf_oracle" if args.classifier_mode == "oracle" else "ltaf"
    output_root = Path(args.output_dir) / f"{model_slug}_{suffix}"

    print("=" * 60)
    print(f"LTAF-Haystack GPT benchmark — provider={provider_name} model={args.model}")
    print(f"classifier_mode={args.classifier_mode}")
    print(f"Input:  {base_dir}")
    print(f"Output: {output_root}")
    print(f"Tasks:  {tasks}")
    print(f"Ctx:    {normalized_ctx}")
    print("=" * 60)

    total = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    for ctx in normalized_ctx:
        ctx_dir = _format_ctx_dir(ctx)
        try:
            ctx_seconds = float(ctx)
        except (TypeError, ValueError):
            ctx_seconds = 3600.0
        tool_rounds = get_tool_rounds(ctx_seconds, cli_override=args.max_tool_rounds)
        for task in tasks:
            parquet_path = base_dir / ctx_dir / task / split_dir / "data.parquet"
            if not parquet_path.exists():
                continue
            n_rows = pd.read_parquet(parquet_path, columns=["question"]).shape[0]
            effective = min(n_rows, args.max_samples) if args.max_samples else n_rows
            print(f"\n{ctx_dir}/{task}/{split_dir}: {effective} samples, "
                  f"{tool_rounds} tool rounds")
            stats = process_parquet(
                parquet_path,
                output_root / ctx_dir / task / split_dir,
                provider, device,
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
            per_ctx.setdefault(str(r.get("context_length_samples", "?")), []).append(r)
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
    r = pp.get("rhythm") or {}
    b = pp.get("beats") or {}
    if r.get("n_chunks_scored"):
        print(f"Pre-pass rhythm: acc={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}  "
              f"(n={r['n_chunks_scored']} chunks)")
    if b.get("n_beats_scored"):
        print(f"Pre-pass beats:  acc={b['accuracy']:.3f}  macro_f1={b['macro_f1']:.3f}  "
              f"(n={b['n_beats_scored']} beats)")
    print(f"\nTotal: success={total['success']}, failed={total['failed']}, "
          f"skipped={total['skipped']}")


if __name__ == "__main__":
    main()
