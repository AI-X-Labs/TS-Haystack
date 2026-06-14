#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Benchmark LLMs on the UK-DALE-Haystack QA split using the trained
``UKDaleClassifier`` as a *pre-pass context provider* + agentic tool back-end.

Unlike a raw-trace benchmark, this script first runs the full QA window
(reconstructed via ``reconstruct_sample_signal``) through the multi-label
appliance classifier, compresses per-sample predictions into per-appliance
bouts (run-length encoding above a threshold), and injects a textual bout
timeline into the user prompt. The agent can also call ``classify_appliance_window``
and ``find_appliance_bouts`` to drill into sub-ranges; both tools are thin
filters over the cached pre-pass predictions.

Two modes:
  - ``real``: run the trained UKDaleClassifier checkpoint on each window.
  - ``oracle``: surface ``needles_json`` + ``other_bouts_json`` directly (the
    ground-truth bouts that the haystack generator inserted / observed in the
    background trace). No classifier, no tools.

Examples:
    .venv/bin/python3 -m src.models.ts_llm.arts.uk_dale_prepass \\
        --classifier-mode real \\
        --classifier-checkpoint results/uk_dale_classifier/best_classifier.pt \\
        --context-lengths 3600 --tasks existence --max-samples 5 --max-workers 1

    .venv/bin/python3 -m src.models.ts_llm.arts.uk_dale_prepass \\
        --classifier-mode oracle \\
        --context-lengths 3600 --tasks all --split test
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
from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB
from src.datasets.uk_dale_haystack.plot_generator import reconstruct_sample_signal
from src.datasets.capture24_haystack.utils.answer_evaluation import (
    evaluate_answer,
    extract_final_answer,
)
from src.models.classifiers.uk_dale.model import (
    UK_DALE_CLASS_NAMES,
    UKDaleClassifier,
    featurize_power,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("results") / "gpt_uk_dale_prepass_benchmark"
BASE_DIR = Path(RAW_DATA) / "uk_dale" / "uk_dale_haystack" / "tasks"

ALL_TASKS = [
    "existence", "localization", "counting", "ordering",
    "state_query", "antecedent", "comparison", "multi_hop",
    "anomaly_detection", "anomaly_localization",
]
ALL_CONTEXT_LENGTHS = [900, 3600, 7200, 32400, 86400]


def _format_ctx_dir(ctx: int | float | str) -> str:
    if isinstance(ctx, str):
        return ctx if ctx.endswith("s") else f"{ctx}s"
    if float(ctx).is_integer():
        return f"{int(ctx)}s"
    return str(ctx).replace(".", "_") + "s"


# ---------------------------------------------------------------------------
# Timestamps — UK-DALE QA uses window-relative 24h "HH:MM:SS"
# ---------------------------------------------------------------------------


def _format_hms(seconds: float) -> str:
    s_total = max(0.0, float(seconds))
    h = int(s_total // 3600)
    m = int((s_total % 3600) // 60)
    s = s_total - h * 3600 - m * 60
    # GT in parquets is integer-second HH:MM:SS; emit the same form so
    # answer evaluation IOU lines up cleanly. Sub-second precision isn't
    # available either way (UK-DALE samples on 6 s grid).
    return f"{h:02d}:{m:02d}:{int(round(s)):02d}"


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
    "You are an expert energy analyst. You are given a UK domestic mains "
    "active-power trace (watts), sampled on a regular 6-second grid. "
    "Your task is to answer questions about which appliances were used and "
    "when, based on a pre-computed appliance-bout analysis.\n\n"
    f"Possible appliances: {', '.join(V1_VOCAB)}."
)


_INSTRUCTIONS_TAIL = """\
- After consulting the classifier context and (optionally) tool results, provide your final answer prefixed with "Answer: ".
- For boolean questions answer "Yes" or "No". For counting answer with just the number. \
For category answer with the canonical appliance label. For time ranges use \
"HH:MM:SS - HH:MM:SS"; for timestamps use "HH:MM:SS"."""


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
            "with just the number. For category answer with the canonical "
            "appliance label. For time ranges use \"HH:MM:SS - HH:MM:SS\"; "
            "for timestamps use \"HH:MM:SS\". The timestamps in the timeline "
            "are exact."
        )
        head = (
            "Instructions:\n"
            "- A ground-truth analysis of the full recording is provided below "
            "(per-appliance bouts with exact timestamps).\n"
            "- Use the provided timeline directly to answer the question; "
            "the labels are authoritative.\n"
        )
        return f"{head}{oracle_tail}"
    tool_list = " and ".join(f"`{n}`" for n in tool_names)
    head = (
        "Instructions:\n"
        "- A pre-computed classifier analysis of the full recording is provided "
        "below. Consult it first.\n"
        f"- You may also call {tool_list} on any sub-range or appliance for a "
        "finer look; these tools return the same pre-computed predictions "
        "filtered to your query.\n"
        "- Classifier predictions may be wrong — cross-check against the "
        "question and prefer evidence from multiple bouts when possible.\n"
    )
    strategy = (
        f"- Recording duration: {recording_duration_sec:.0f}s. "
        f"Tool query ranges may span the full recording; results are "
        f"per-appliance bouts, not raw samples.\n"
    )
    return f"{head}{strategy}{_INSTRUCTIONS_TAIL}"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_RANGE_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "start_time": {
            "type": "string",
            "description": "Sub-range start in 'HH:MM:SS' format (window-relative)",
        },
        "end_time": {
            "type": "string",
            "description": "Sub-range end in 'HH:MM:SS' format (window-relative)",
        },
    },
    "required": ["start_time", "end_time"],
    "additionalProperties": False,
}


def get_classify_window_tool() -> dict:
    return {
        "type": "function",
        "name": "classify_appliance_window",
        "description": (
            "Return the per-appliance bouts (start, end, peak watts, mean confidence) "
            "that overlap the given sub-range, as produced by the UKDaleClassifier "
            "in the pre-pass. Each entry has appliance label, start/end timestamps, "
            "peak power in watts, and mean prediction confidence."
        ),
        "parameters": _RANGE_TOOL_PARAMS,
        "strict": True,
    }


def get_find_bouts_tool() -> dict:
    return {
        "type": "function",
        "name": "find_appliance_bouts",
        "description": (
            "Return all bouts for a specific appliance inside the given sub-range, "
            "as produced by the UKDaleClassifier in the pre-pass."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "appliance": {
                    "type": "string",
                    "description": (
                        "Canonical appliance label, one of: " + ", ".join(V1_VOCAB)
                    ),
                    "enum": list(V1_VOCAB),
                },
                "start_time": _RANGE_TOOL_PARAMS["properties"]["start_time"],
                "end_time": _RANGE_TOOL_PARAMS["properties"]["end_time"],
            },
            "required": ["appliance", "start_time", "end_time"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def tools_for_sample() -> list[dict]:
    return [get_classify_window_tool(), get_find_bouts_tool()]


# ---------------------------------------------------------------------------
# Classifier loading — thread-safe inference
# ---------------------------------------------------------------------------

_classifier_lock = Lock()
_classifier: Optional[UKDaleClassifier] = None


def load_classifier(checkpoint: str, device: str) -> None:
    global _classifier
    print(f"Loading UKDaleClassifier from {checkpoint} on {device}...")
    _classifier = UKDaleClassifier.load(checkpoint, device=device)
    print(f"  classes: {_classifier.class_names}  params: "
          f"{sum(p.numel() for p in _classifier.parameters()):,}")


# ---------------------------------------------------------------------------
# Pre-pass — per-sample multi-label predictions, compressed to bouts
# ---------------------------------------------------------------------------

NUM_CLASSES = len(UK_DALE_CLASS_NAMES)
APP_TO_IDX = {a: i for i, a in enumerate(UK_DALE_CLASS_NAMES)}


def _classify_full_signal(
    power_w: np.ndarray,
    device: str,
    chunk_size: int = 16384,
) -> np.ndarray:
    """Run the classifier across a full 1-D mains trace.

    Returns ``probs`` of shape ``(NUM_CLASSES, L)`` — sigmoid probability per
    sample per class. Long traces are split into ``chunk_size`` segments with
    a small overlap so the dilated TCN's receptive field is honoured at the
    boundaries; only the centre of each chunk is kept.
    """
    import torch

    assert _classifier is not None, "classifier not loaded"
    L = int(power_w.shape[0])
    probs = np.zeros((NUM_CLASSES, L), dtype=np.float32)

    # Receptive-field padding around chunk borders.
    overlap = 64

    n_in_channels = int(_classifier.n_channels)

    with _classifier_lock:
        with torch.no_grad():
            i = 0
            while i < L:
                # Define chunk including overlap on each side
                lo = max(0, i - overlap)
                hi = min(L, i + chunk_size + overlap)
                seg = power_w[lo:hi].astype(np.float32)
                feats = featurize_power(seg)  # (2, L') if v2; (1, L') if legacy
                if feats.ndim == 1:
                    feats = feats[None, :]
                # If the classifier was trained with only the level channel
                # (legacy v1 ckpt), drop the edge row.
                if n_in_channels == 1 and feats.shape[0] == 2:
                    feats = feats[:1]
                x = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # (1,C_in,L')
                logits = _classifier(x)  # (1, C, L')
                p = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (C, L')
                # Place the centre region back into ``probs`` (drop overlap pads)
                center_lo = i - lo
                center_hi = center_lo + min(chunk_size, L - i)
                probs[:, i:i + (center_hi - center_lo)] = p[:, center_lo:center_hi]
                i += chunk_size
    return probs


def _bouts_from_probs(
    probs: np.ndarray,
    dt_s: float,
    threshold: float = 0.5,
    min_duration_s: float = 12.0,
) -> list[dict]:
    """Run-length encode per-sample probs into per-appliance bouts.

    Each bout dict: {appliance, start_s, end_s, peak_prob, mean_prob, duration_s}.
    Drops bouts shorter than ``min_duration_s`` (one or two grid steps of
    flicker — the classifier sometimes single-step blips).
    """
    out: list[dict] = []
    C, L = probs.shape
    for c in range(C):
        active = probs[c] >= threshold
        if not active.any():
            continue
        # RLE
        diffs = np.diff(active.astype(np.int8))
        starts = np.flatnonzero(diffs > 0) + 1
        ends = np.flatnonzero(diffs < 0) + 1
        if active[0]:
            starts = np.concatenate([[0], starts])
        if active[-1]:
            ends = np.concatenate([ends, [L]])
        for s, e in zip(starts, ends):
            duration_s = (e - s) * dt_s
            if duration_s < min_duration_s:
                continue
            seg = probs[c, s:e]
            out.append({
                "appliance": UK_DALE_CLASS_NAMES[c],
                "start_s": float(s * dt_s),
                "end_s": float(e * dt_s),
                "duration_s": float(duration_s),
                "peak_prob": float(seg.max()),
                "mean_prob": float(seg.mean()),
            })
    out.sort(key=lambda b: (b["start_s"], b["end_s"], b["appliance"]))
    return out


def _attach_peak_w(bouts: list[dict], mains_w: np.ndarray, dt_s: float) -> None:
    """Add ``peak_w`` / ``mean_w`` (mains stats over the bout span) in-place.

    These are diagnostic only — the classifier sees log1p-power and only
    predicts which appliance is on, but the LLM finds power magnitudes useful
    for things like "loudest microwave event".
    """
    L = mains_w.shape[0]
    for b in bouts:
        i0 = int(round(b["start_s"] / dt_s))
        i1 = int(round(b["end_s"] / dt_s))
        i0 = max(0, min(L, i0))
        i1 = max(i0 + 1, min(L, i1))
        seg = mains_w[i0:i1]
        b["peak_w"] = float(seg.max()) if seg.size else 0.0
        b["mean_w"] = float(seg.mean()) if seg.size else 0.0


def _oracle_bouts_from_row(row: dict, dt_s: float) -> list[dict]:
    """Build the GT per-appliance bout list from the row's needles + other_bouts.

    needles_json entries use ``insert_position_samples`` / ``insert_duration_samples``
    on the same window-local 6 s grid as the reconstructed mains.
    other_bouts_json entries use ``start_sample`` / ``end_sample``.
    """
    out: list[dict] = []
    needles = json.loads(row["needles_json"]) if row.get("needles_json") else []
    other = json.loads(row["other_bouts_json"]) if row.get("other_bouts_json") else []
    for n in needles:
        s = int(n["insert_position_samples"])
        e = s + int(n["insert_duration_samples"])
        out.append({
            "appliance": str(n["appliance"]),
            "start_s": float(s * dt_s),
            "end_s": float(e * dt_s),
            "duration_s": float((e - s) * dt_s),
            "peak_prob": 1.0,
            "mean_prob": 1.0,
            "peak_w": float(n.get("peak_w", 0.0) or 0.0),
            "is_inserted": True,
            "is_anomalous": bool(n.get("is_anomalous", False)),
            "anomaly_class": n.get("anomaly_class"),
        })
    for b in other:
        s = int(b["start_sample"])
        e = int(b["end_sample"])
        out.append({
            "appliance": str(b["appliance"]),
            "start_s": float(s * dt_s),
            "end_s": float(e * dt_s),
            "duration_s": float((e - s) * dt_s),
            "peak_prob": 1.0,
            "mean_prob": 1.0,
            "peak_w": float(b.get("peak_w", 0.0) or 0.0),
            "mean_w": float(b.get("mean_w", 0.0) or 0.0),
            "is_inserted": False,
        })
    out.sort(key=lambda b: (b["start_s"], b["end_s"], b["appliance"]))
    return out


def run_pre_pass(
    sample: dict,
    device: str,
    classifier_mode: str = "real",
    threshold: float = 0.5,
    min_bout_s: float = 12.0,
) -> tuple[list[dict], np.ndarray, np.ndarray | None]:
    """Returns (bouts, mains_w, probs_or_None).

    In ``real`` mode: classify the reconstructed mains, RLE → bouts.
    In ``oracle`` mode: read GT bouts from the row directly, no signal forward.
    """
    dt_s = float(sample.get("dt_s", 6.0) or 6.0)
    mains_w = reconstruct_sample_signal(sample)

    if classifier_mode == "oracle":
        bouts = _oracle_bouts_from_row(sample, dt_s)
        return bouts, mains_w, None

    probs = _classify_full_signal(mains_w, device)
    bouts = _bouts_from_probs(probs, dt_s, threshold=threshold, min_duration_s=min_bout_s)
    _attach_peak_w(bouts, mains_w, dt_s)
    return bouts, mains_w, probs


# ---------------------------------------------------------------------------
# Pre-pass metrics — per-appliance presence agreement (per 6 s sample)
# ---------------------------------------------------------------------------


def _bouts_to_per_sample(bouts: list[dict], n_samples: int, dt_s: float) -> np.ndarray:
    """(NUM_CLASSES, n_samples) bool array of presence per sample."""
    out = np.zeros((NUM_CLASSES, n_samples), dtype=bool)
    for b in bouts:
        idx = APP_TO_IDX.get(str(b["appliance"]))
        if idx is None:
            continue
        s = max(0, int(round(b["start_s"] / dt_s)))
        e = min(n_samples, int(round(b["end_s"] / dt_s)))
        if e > s:
            out[idx, s:e] = True
    return out


def compute_pre_pass_metrics(
    pred_bouts: list[dict],
    gt_bouts: list[dict],
    n_samples: int,
    dt_s: float,
) -> dict:
    """Per-appliance precision/recall/F1 + confusion (per-sample agreement)."""
    pred = _bouts_to_per_sample(pred_bouts, n_samples, dt_s)
    gt = _bouts_to_per_sample(gt_bouts, n_samples, dt_s)
    out: dict[str, dict] = {}
    confusion = {a: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for a in UK_DALE_CLASS_NAMES}
    for c, name in enumerate(UK_DALE_CLASS_NAMES):
        p = pred[c]
        g = gt[c]
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        tn = int((~p & ~g).sum())
        confusion[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support_gt": int(g.sum()),
            "support_pred": int(p.sum()),
        }
    macro_f1 = float(np.mean([out[a]["f1"] for a in UK_DALE_CLASS_NAMES if out[a]["support_gt"] > 0])) \
        if any(out[a]["support_gt"] > 0 for a in UK_DALE_CLASS_NAMES) else None
    return {
        "per_class": out,
        "macro_f1": macro_f1,
        "confusion": confusion,
        "n_samples_scored": int(n_samples),
    }


# ---------------------------------------------------------------------------
# Context-file rendering
# ---------------------------------------------------------------------------


def render_context_file(
    bouts: list[dict],
    window_duration_sec: float,
    classifier_mode: str = "real",
    max_bouts_listed: int = 200,
) -> str:
    is_oracle = classifier_mode == "oracle"
    header_label = "Ground-truth" if is_oracle else "Classifier"
    source = "ground truth" if is_oracle else "UKDaleClassifier (per-sample multi-label)"

    lines: list[str] = []
    lines.append(
        f"# {header_label} pre-pass summary (window 00:00:00 - "
        f"{_format_hms(window_duration_sec)})"
    )

    # Counts header
    counts: dict[str, int] = {a: 0 for a in UK_DALE_CLASS_NAMES}
    for b in bouts:
        if b["appliance"] in counts:
            counts[b["appliance"]] += 1
    counts_str = "  ".join(f"{a}={counts[a]}" for a in UK_DALE_CLASS_NAMES if counts[a] > 0)
    lines.append("")
    lines.append(
        f"## Appliance bouts ({source}, {len(bouts)} bouts)"
    )
    if counts_str:
        lines.append(f"  per-appliance counts: {counts_str}")
    if not bouts:
        lines.append("  (no appliance bouts detected)" if not is_oracle else
                     "  (no appliance bouts in this window)")

    if len(bouts) > max_bouts_listed:
        lines.append(
            f"  (listing first {max_bouts_listed} of {len(bouts)} bouts; "
            f"call `find_appliance_bouts` or `classify_appliance_window` for more)"
        )
        listed = bouts[:max_bouts_listed]
    else:
        listed = bouts

    for b in listed:
        peak_w = b.get("peak_w")
        peak_part = f"  peak={peak_w:.0f}W" if peak_w else ""
        if is_oracle:
            extra = ""
            if b.get("is_inserted"):
                extra = "  [inserted]"
            if b.get("is_anomalous"):
                extra += f"  [anomaly:{b.get('anomaly_class', '?')}]"
            lines.append(
                f"  {_format_hms(b['start_s'])} - {_format_hms(b['end_s'])}  "
                f"{b['appliance']:<18s}  ({b['duration_s']:.0f}s){peak_part}{extra}"
            )
        else:
            lines.append(
                f"  {_format_hms(b['start_s'])} - {_format_hms(b['end_s'])}  "
                f"{b['appliance']:<18s}  conf={b['mean_prob']:.2f}  "
                f"({b['duration_s']:.0f}s){peak_part}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool execution (cached lookups into pre-pass results)
# ---------------------------------------------------------------------------


# UK-DALE samples on a 6 s grid; the tool returns aggregated bouts, not
# per-sample data, so the per-query cap inherited from benchmark_gpt.py
# (120 s, calibrated for 100 Hz / 256 Hz signals) is too restrictive here.
# Allow the full recording duration as a query span.
def _max_query_seconds_uk_dale(recording_duration_sec: float) -> float:
    return float(recording_duration_sec)


def _validate_range(
    start_time: str,
    end_time: str,
    recording_duration_sec: float,
) -> tuple[float, float] | str:
    try:
        s = _parse_hms(start_time)
        e = _parse_hms(end_time)
    except Exception as exc:
        return json.dumps({"error": f"bad timestamp: {exc}",
                           "segment": f"{start_time} - {end_time}"})
    if e < s:
        return json.dumps({"error": "end_time < start_time",
                           "segment": f"{start_time} - {end_time}"})
    max_sec = _max_query_seconds_uk_dale(recording_duration_sec)
    if e - s > max_sec + 1e-3:
        return json.dumps({
            "error": f"Segment too long: {e - s:.1f}s exceeds maximum {max_sec:.1f}s.",
            "segment": f"{start_time} - {end_time}",
        })
    if s < 0 or e > recording_duration_sec + 1e-3:
        return json.dumps({
            "error": f"Segment outside recording window [0, {recording_duration_sec:.1f}s]",
            "segment": f"{start_time} - {end_time}",
        })
    return s, e


def _tool_classify_appliance_window(
    start_time: str,
    end_time: str,
    bouts: list[dict],
    recording_duration_sec: float,
    max_results: int = 200,
) -> str:
    rng = _validate_range(start_time, end_time, recording_duration_sec)
    if isinstance(rng, str):
        return rng
    s, e = rng
    hits = [b for b in bouts if b["end_s"] > s and b["start_s"] < e]
    truncated = False
    if len(hits) > max_results:
        hits = hits[:max_results]
        truncated = True
    out = {
        "segment": f"{start_time} - {end_time}",
        "n_bouts": len(hits),
        "bouts": [
            {
                "start": _format_hms(b["start_s"]),
                "end": _format_hms(b["end_s"]),
                "appliance": b["appliance"],
                "peak_w": round(float(b.get("peak_w") or 0.0), 1),
                "confidence": round(float(b.get("mean_prob") or 1.0), 3),
            }
            for b in hits
        ],
    }
    if truncated:
        out["truncated"] = f"first {max_results} bouts only — narrow the range or use `find_appliance_bouts` for one appliance"
    return json.dumps(out)


def _tool_find_appliance_bouts(
    appliance: str,
    start_time: str,
    end_time: str,
    bouts: list[dict],
    recording_duration_sec: float,
) -> str:
    rng = _validate_range(start_time, end_time, recording_duration_sec)
    if isinstance(rng, str):
        return rng
    s, e = rng
    if appliance not in V1_VOCAB:
        return json.dumps({
            "error": f"unknown appliance '{appliance}' — must be one of {V1_VOCAB}",
            "segment": f"{start_time} - {end_time}",
        })
    hits = [b for b in bouts
            if b["appliance"] == appliance and b["end_s"] > s and b["start_s"] < e]
    return json.dumps({
        "segment": f"{start_time} - {end_time}",
        "appliance": appliance,
        "n_bouts": len(hits),
        "bouts": [
            {
                "start": _format_hms(b["start_s"]),
                "end": _format_hms(b["end_s"]),
                "duration_s": round(float(b["duration_s"]), 1),
                "peak_w": round(float(b.get("peak_w") or 0.0), 1),
                "confidence": round(float(b.get("mean_prob") or 1.0), 3),
            }
            for b in hits
        ],
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
    rec_end = _format_hms(recording_duration_sec)
    house = sample.get("background_house_id")
    pre = (
        f"You are given a UK domestic mains active-power trace from house "
        f"{house}. The window spans 00:00:00 to {rec_end} "
        f"({recording_duration_sec:.0f} seconds, sampled every "
        f"{float(sample.get('dt_s', 6.0)):.0f} s).\n\n"
        f"{context_text}\n\n"
        f"Question: {sample['question']}"
    )
    post = get_instructions(
        recording_duration_sec, tool_names, classifier_mode=classifier_mode,
    )
    return pre, post


def run_sample(
    sample: dict,
    idx: int,
    provider,
    device: str,
    max_tool_rounds: int,
    reasoning_effort: str,
    classifier_mode: str = "real",
    threshold: float = 0.5,
    min_bout_s: float = 12.0,
) -> dict:
    recording_duration_sec = float(sample.get("context_length_s") or 0.0)
    if recording_duration_sec <= 0:
        # Derive from background bounds if context_length_s wasn't materialised.
        recording_duration_sec = (
            int(sample["background_end_ns"]) - int(sample["background_start_ns"])
        ) / 1e9
    dt_s = float(sample.get("dt_s", 6.0) or 6.0)

    try:
        pred_bouts, mains_w, _probs = run_pre_pass(
            sample, device, classifier_mode=classifier_mode,
            threshold=threshold, min_bout_s=min_bout_s,
        )
    except Exception as exc:
        return {"index": idx, "error": f"pre-pass failed: {exc}",
                "question": sample.get("question", "")}

    # GT for metrics (real mode only)
    pre_pass_metrics: dict = {}
    if classifier_mode == "real":
        try:
            gt_bouts = _oracle_bouts_from_row(sample, dt_s)
            pre_pass_metrics = compute_pre_pass_metrics(
                pred_bouts, gt_bouts, mains_w.shape[0], dt_s,
            )
        except Exception:
            pre_pass_metrics = {}

    context_text = render_context_file(
        pred_bouts, recording_duration_sec, classifier_mode=classifier_mode,
    )

    active_tools = tools_for_sample()
    llm_tools = [] if classifier_mode == "oracle" else active_tools
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
            if tc.name == "classify_appliance_window":
                tool_result = _tool_classify_appliance_window(
                    st, et, pred_bouts, recording_duration_sec,
                )
            elif tc.name == "find_appliance_bouts":
                tool_result = _tool_find_appliance_bouts(
                    args.get("appliance", ""), st, et,
                    pred_bouts, recording_duration_sec,
                )
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
    eval_result = evaluate_answer(
        sample["answer"], predicted, answer_type,
        timestamp_tolerance_s=float(sample.get("dt_s", 6.0) or 6.0),
    )

    return {
        "index": idx,
        "background_house_id": sample.get("background_house_id"),
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
        "context_length_s": float(sample.get("context_length_s", 0) or 0),
        "classifier_mode": classifier_mode,
        "pre_pass": {
            "n_bouts": len(pred_bouts),
            "n_appliances_active": len({b["appliance"] for b in pred_bouts}),
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
        "answers": {"correct": 0, "total": 0},
        "per_class": {a: {"tp": 0, "fp": 0, "fn": 0} for a in UK_DALE_CLASS_NAMES},
        "input_chars": {"sum": 0, "n": 0, "max": 0, "min": None},
    }


def _update_running_totals(running: dict, result: dict) -> None:
    running["answers"]["total"] += 1
    if result.get("correct"):
        running["answers"]["correct"] += 1
    pp = (result.get("pre_pass_metrics") or {}).get("confusion") or {}
    for a, cm in pp.items():
        if a in running["per_class"]:
            running["per_class"][a]["tp"] += int(cm.get("tp", 0))
            running["per_class"][a]["fp"] += int(cm.get("fp", 0))
            running["per_class"][a]["fn"] += int(cm.get("fn", 0))
    icl = result.get("input_char_lengths") or {}
    n_chars = int(icl.get("total") or 0)
    if n_chars:
        rc = running["input_chars"]
        rc["sum"] += n_chars
        rc["n"] += 1
        if n_chars > rc["max"]:
            rc["max"] = n_chars
        if rc["min"] is None or n_chars < rc["min"]:
            rc["min"] = n_chars


def _running_summary(running: dict) -> dict:
    out: dict = {}
    ans = running["answers"]
    out["answers"] = {
        "n": ans["total"],
        "accuracy": (ans["correct"] / ans["total"]) if ans["total"] else None,
    }
    f1s = []
    for a in UK_DALE_CLASS_NAMES:
        cm = running["per_class"][a]
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if tp + fn > 0:
            f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else None
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
    if s.get("macro_f1") is not None:
        lines.append(f"    pre-pass macro F1: {s['macro_f1']:.3f}")
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
    device: str,
    max_samples: Optional[int],
    max_workers: int,
    max_tool_rounds: int,
    reasoning_effort: str,
    classifier_mode: str = "real",
    threshold: float = 0.5,
    min_bout_s: float = 12.0,
) -> dict:
    df = pd.read_parquet(parquet_path)
    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(bool)].reset_index(drop=True)
    if max_samples is not None:
        df = df.head(max_samples)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "trajectories.jsonl"
    completed = load_completed_indices(jsonl_path)
    todo = [i for i in range(len(df)) if i not in completed]
    stats = {"total": len(df), "success": 0, "failed": 0, "skipped": len(completed)}

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
            classifier_mode=classifier_mode,
            threshold=threshold, min_bout_s=min_bout_s,
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
                if live.get("macro_f1") is not None:
                    postfix["pre_f1"] = f"{live['macro_f1']:.2f}"
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
    ap.add_argument("--classifier-mode", choices=["real", "oracle"], default="real")
    ap.add_argument("--classifier-checkpoint", type=str, default=None,
                    help="Path to UKDaleClassifier best_classifier.pt "
                         "(required in real mode; ignored in oracle mode).")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--input-dir", type=str, default=str(BASE_DIR))
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--context-lengths", type=str, nargs="+", default=None,
                    help=f"Context length grid (e.g. 900 3600 7200). "
                         f"Defaults to {ALL_CONTEXT_LENGTHS}.")
    ap.add_argument("--tasks", type=str, nargs="+", default=["all"])
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "validation", "test"])
    ap.add_argument("--model", type=str, default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--max-tool-rounds", type=int, default=None)
    ap.add_argument("--reasoning-effort", type=str, default="high",
                    choices=["none", "low", "medium", "high"])
    ap.add_argument("--provider", type=str, default=None,
                    choices=["openai", "anthropic", "bedrock"])
    ap.add_argument("--aws-region", type=str, default="us-east-1")
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Classifier sigmoid threshold for bout RLE")
    ap.add_argument("--min-bout-s", type=float, default=12.0,
                    help="Drop predicted bouts shorter than this (seconds)")
    args = ap.parse_args()

    # Provider selection (mirrors benchmark_gpt_sleep_prepass.py)
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
        if not args.classifier_checkpoint:
            print("Error: --classifier-checkpoint is required for real mode")
            sys.exit(1)
        import torch
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        load_classifier(args.classifier_checkpoint, device)
    else:
        device = "cpu"
        print("Oracle mode: using ground-truth bouts (no classifier loaded).")

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks
    ctx_grid = args.context_lengths or [str(c) for c in ALL_CONTEXT_LENGTHS]
    normalized_ctx: list = []
    for c in ctx_grid:
        try:
            normalized_ctx.append(float(c))
        except ValueError:
            normalized_ctx.append(c)

    base_dir = Path(args.input_dir)
    split_dir = "validation" if args.split == "val" else args.split
    model_slug = args.model.replace("/", "_")
    suffix = "uk_dale_oracle" if args.classifier_mode == "oracle" else "uk_dale"
    output_root = Path(args.output_dir) / f"{model_slug}_{suffix}"

    print("=" * 60)
    print(f"UK-DALE-Haystack pre-pass GPT benchmark — provider={provider_name} model={args.model}")
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
                classifier_mode=args.classifier_mode,
                threshold=args.threshold,
                min_bout_s=args.min_bout_s,
            )
            for k in total:
                total[k] += stats[k]
            print(f"  success={stats['success']}, failed={stats['failed']}, "
                  f"skipped={stats['skipped']}")

    # Aggregate summary
    summary: dict = {"by_task": {}, "by_context": {}, "overall": {}, "pre_pass": {}}
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

        # Pool per-class confusion across all samples
        per_class = {a: {"tp": 0, "fp": 0, "fn": 0} for a in UK_DALE_CLASS_NAMES}
        for r in rows:
            pp = (r.get("pre_pass_metrics") or {}).get("confusion") or {}
            for a, cm in pp.items():
                if a in per_class:
                    per_class[a]["tp"] += int(cm.get("tp", 0))
                    per_class[a]["fp"] += int(cm.get("fp", 0))
                    per_class[a]["fn"] += int(cm.get("fn", 0))
        per_class_metrics: dict = {}
        f1s = []
        for a in UK_DALE_CLASS_NAMES:
            tp, fp, fn = per_class[a]["tp"], per_class[a]["fp"], per_class[a]["fn"]
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            per_class_metrics[a] = {
                "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
            }
            if tp + fn > 0:
                f1s.append(f1)
        summary["pre_pass"] = {
            "macro_f1": float(np.mean(f1s)) if f1s else None,
            "per_class": per_class_metrics,
        }

    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    if summary.get("overall"):
        ov = summary["overall"]
        print(f"Overall: {ov['correct']}/{ov['total']} = {ov['accuracy'] * 100:.1f}%")
    pp = summary.get("pre_pass") or {}
    if pp.get("macro_f1") is not None:
        print(f"Pre-pass macro F1: {pp['macro_f1']:.3f}")
    print(f"\nTotal: success={total['success']}, failed={total['failed']}, "
          f"skipped={total['skipped']}")


if __name__ == "__main__":
    main()
