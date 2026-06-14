#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Visual verification script for LTAF-Haystack samples (natural-only).

For each enabled task at a given context length, draw ``n_samples``
valid examples and persist:

  * ``sample_{i}_window.png`` — full window with rhythm bands, beat
    markers, answer highlights, and Q/A text.
  * ``sample_{i}_zoomed.png`` — centred zoom of ~5 min around the answer.
  * ``sample_{i}.json``       — the sample's ``to_dict()`` payload
    (signals excluded to keep the file small).

Output layout::

  data/ltafdb/ltaf_haystack/rhythms/verification/
      ctx_{ctx_s}s/
          {task}/
              sample_00_window.png
              sample_00_zoomed.png
              sample_00.json
              ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFBoutRecord,
    LTAFGeneratedSample,
)
from src.datasets.ltaf_haystack.core.ltaf_prompt_templates import (
    LTAFPromptTemplateBank,
)
from src.datasets.ltaf_haystack.core.participant_splitter import (
    load_split_manifest,
)
from src.datasets.ltaf_haystack.core.recording_sampler import RecordingSampler
from src.datasets.ltaf_haystack.core.seed_manager import LTAFSeedManager
from src.datasets.ltaf_haystack.core.window_index import LTAFWindowIndex
from src.datasets.ltaf_haystack.loader import LTAF_DATA_DIR, CHANNEL_NAMES
from src.datasets.ltaf_haystack.plot_generator import create_ecg_plot
from src.datasets.ltaf_haystack.tasks import (
    get_task_generator,
    list_available_tasks,
)


LTAF_VERIFICATION_ROOT = LTAF_DATA_DIR / "ltaf_haystack" / "rhythms" / "verification"


def _answer_regions(
    sample: LTAFGeneratedSample, source_hz: int
) -> List[Tuple[int, int, str]]:
    """Return (start, end, label) regions to paint on the plot.

    Label drives the colour in :data:`plot_generator.HIGHLIGHT_COLOURS`:
      * ``"answer"``   — the bout/beat that *is* the answer (yellow band).
      * ``"context"``  — a region the question refers to but is not itself
                         the answer (teal band, e.g. ``antecedent``'s
                         target bout).

    Metadata key conventions:
      * ``start_sample``/``end_sample``                  — answer bout.
      * ``beat_sample``                                  — answer beat.
      * ``anomaly_beat_samples``                         — every matching
                                                          V/A beat (used by
                                                          ``anomaly_detection``
                                                          when the answer is
                                                          "yes").
      * ``bout_segments``                                — list of
                                                          ``[start, end]``
                                                          pairs, one per
                                                          counted bout
                                                          (``counting``).
      * ``context_start_sample``/``context_end_sample``  — secondary
                                                          question-context
                                                          bout, painted in a
                                                          different colour.
    """
    meta = sample.metadata or {}
    half = max(1, source_hz // 2)
    out: List[Tuple[int, int, str]] = []

    beat_samples = meta.get("anomaly_beat_samples") or []
    bout_segments = meta.get("bout_segments") or []
    if beat_samples:
        out.extend(
            (max(0, int(s) - half), int(s) + half, "answer") for s in beat_samples
        )
    elif bout_segments:
        out.extend(
            (int(seg[0]), int(seg[1]), "answer") for seg in bout_segments
        )
    elif "start_sample" in meta and "end_sample" in meta:
        out.append((int(meta["start_sample"]), int(meta["end_sample"]), "answer"))
    elif "beat_sample" in meta:
        s = int(meta["beat_sample"])
        out.append((max(0, s - half), s + half, "answer"))

    if "context_start_sample" in meta and "context_end_sample" in meta:
        out.append(
            (
                int(meta["context_start_sample"]),
                int(meta["context_end_sample"]),
                "context",
            )
        )
    return out


def _zoomed_bounds(
    sample: LTAFGeneratedSample,
    zoom_seconds: float,
) -> Tuple[int, int]:
    n = int(sample.signals.shape[0])
    zoom_samples = min(n, max(1, int(round(zoom_seconds * sample.source_hz))))
    regions = _answer_regions(sample, sample.source_hz)
    answer_regions = [r for r in regions if r[2] == "answer"]
    if answer_regions:
        # Center on the median answer region so multi-beat anomaly samples
        # land at least one highlight inside the zoom window.
        mid_region = answer_regions[len(answer_regions) // 2]
        mid = (mid_region[0] + mid_region[1]) // 2
    elif regions:
        mid_region = regions[len(regions) // 2]
        mid = (mid_region[0] + mid_region[1]) // 2
    else:
        mid = n // 2
    lo = max(0, mid - zoom_samples // 2)
    hi = min(n, lo + zoom_samples)
    lo = max(0, hi - zoom_samples)
    return lo, hi


def _render_sample(
    sample: LTAFGeneratedSample,
    output_dir: Path,
    index: int,
    zoom_seconds: float = 300.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = sample.signals
    highlights: List[Tuple[int, int, str]] = _answer_regions(
        sample, sample.source_hz
    )

    # Generated samples don't carry the rhythm timeline; we skip bands in
    # this view and rely on the zoomed plot + answer highlight.
    rhythm_bands: List[LTAFBoutRecord] = []
    beat_markers: dict = {}

    title = (
        f"{sample.task_type} · ctx={sample.context_length_samples / sample.source_hz:.0f}s "
        f"· record={sample.record_id}"
    )
    img = create_ecg_plot(
        signals=signals,
        channel_names=CHANNEL_NAMES,
        rhythm_bands=rhythm_bands,
        beat_markers=beat_markers,
        highlights=highlights,
        title=title,
        question=sample.question,
        answer=sample.answer,
        source_hz=sample.source_hz,
    )
    img.save(output_dir / f"sample_{index:02d}_window.png")

    lo, hi = _zoomed_bounds(sample, zoom_seconds)
    zoomed_signals = signals[lo:hi]
    zoom_highlights: List[Tuple[int, int, str]] = []
    for s, e, label in highlights:
        if e <= lo or s >= hi:
            continue
        zoom_highlights.append(
            (max(0, s - lo), min(hi - lo, e - lo), label)
        )

    img_z = create_ecg_plot(
        signals=zoomed_signals,
        channel_names=CHANNEL_NAMES,
        rhythm_bands=[],
        beat_markers={},
        highlights=zoom_highlights,
        title=title + f"  [zoom {(hi-lo)/sample.source_hz:.0f}s]",
        question=sample.question,
        answer=sample.answer,
        source_hz=sample.source_hz,
    )
    img_z.save(output_dir / f"sample_{index:02d}_zoomed.png")

    payload = sample.to_dict()
    payload.pop("signals", None)
    with (output_dir / f"sample_{index:02d}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _build_natural_sample(generator, rng) -> Optional[LTAFGeneratedSample]:
    try:
        recording = generator.recording_sampler.sample_recording(rng)
    except ValueError:
        return None
    sample = generator.generate_sample(recording, rng)
    return sample if sample.is_valid else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-length-seconds",
        type=float,
        default=900.0,
        help="Context length in seconds. Default 900 (15 min).",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of samples per task.",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--label-class", default="rhythms")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--zoom-seconds",
        type=float,
        default=300.0,
        help="Zoom window length for the zoomed plot. Default 300s.",
    )
    args = parser.parse_args()

    ctx_s = float(args.context_length_seconds)

    manifest = load_split_manifest()
    split_records = manifest.get(args.split, [])
    if not split_records:
        print(f"No records in split={args.split}", file=sys.stderr)
        sys.exit(1)

    window_index = LTAFWindowIndex.get_or_build(
        label_class=args.label_class,
        context_length_s=ctx_s,
        record_ids=split_records,
    )
    recording_sampler = RecordingSampler(split_records, args.label_class, window_index)
    template_bank = LTAFPromptTemplateBank()
    seed_manager = LTAFSeedManager(master_seed=args.seed)

    available = list_available_tasks()
    tasks = args.tasks or available
    tasks = [t for t in tasks if t in available]
    if not tasks:
        print(f"No valid tasks. Available: {available}", file=sys.stderr)
        sys.exit(1)

    output_root = args.output_root or LTAF_VERIFICATION_ROOT / f"ctx_{int(round(ctx_s))}s"
    output_root.mkdir(parents=True, exist_ok=True)

    for task_name in tasks:
        TaskClass = get_task_generator(task_name)
        if not TaskClass.supports_context_length(args.label_class, ctx_s):
            print(f"  [skip] {task_name} gated off at ctx={ctx_s}s")
            continue
        generator = TaskClass(
            recording_sampler=recording_sampler,
            template_bank=template_bank,
            seed_manager=seed_manager,
            label_class=args.label_class,
        )
        task_dir = output_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(args.seed + hash(task_name) % 100_000)
        produced = 0
        for _ in range(args.n_samples * 10):
            if produced >= args.n_samples:
                break
            sample = _build_natural_sample(generator, rng)
            if sample is None:
                continue
            _render_sample(sample, task_dir, index=produced, zoom_seconds=args.zoom_seconds)
            produced += 1

        print(
            f"  [{task_name}] ctx={ctx_s:.0f}s produced {produced} samples → {task_dir}"
        )


if __name__ == "__main__":
    main()
