#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Visual verification script for the windowed-natural Sleep PSG benchmark.

For each task, samples a few QA examples (using a windowed RecordingSampler
matching the requested context length) and saves two annotated PSG plots:

  - sample_NN_window.png : the entire window the sample was generated from
  - sample_NN_zoomed.png : a 5-minute zoom centered on the answer region

Usage:
    .venv/bin/python3 src/datasets/sleep_psg_haystack/scripts/verify_sleep_psg_haystack.py
    .venv/bin/python3 src/datasets/sleep_psg_haystack/scripts/verify_sleep_psg_haystack.py \
        --label-class arousals --context-length-seconds 900
    .venv/bin/python3 src/datasets/sleep_psg_haystack/scripts/verify_sleep_psg_haystack.py \
        --label-class sleep_stages --context-length-seconds full

Output: data/sleep_psg/ts_haystack/{label_class}/verification/{ctx_dir}/{task}/
"""

import argparse
import json

import numpy as np

from src.datasets.sleep_psg_haystack.loader import (
    CHANNEL_NAMES,
    SLEEP_PSG_DATA_DIR,
    SOURCE_HZ,
    load_subject_signals_mmap,
)
from src.datasets.sleep_psg_haystack.core.participant_splitter import split_participants
from src.datasets.sleep_psg_haystack.core.recording_sampler import RecordingSampler
from src.datasets.sleep_psg_haystack.core.window_index import SleepPSGWindowIndex
from src.datasets.sleep_psg_haystack.core.prompt_templates import SleepPromptTemplateBank
from src.datasets.sleep_psg_haystack.generation.generator import (
    DEFAULT_CONTEXT_LENGTHS_S,
    context_dir_name,
    parse_context_token,
)
from src.datasets.sleep_psg_haystack.plot_generator import create_psg_plot
from src.datasets.sleep_psg_haystack.tasks import PSG_TASK_REGISTRY, get_tasks_for_label_class

from src.datasets.capture24_haystack.core.seed_manager import SeedManager


# How much context (in seconds) to show in the zoomed plot around the answer
PLOT_CONTEXT_SECONDS = 300  # 5 min


def _ord(n):
    if n is None:
        return ""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th') }"


def _get_all_regions(task_type: str, metadata: dict, recording=None) -> list:
    """
    Extract (window-relative) annotatable regions per task with descriptive
    labels so the plot legend reads e.g. "anchor: 1st N3", not "target".
    Returns list of (start_ms, end_ms, label) tuples.

    `recording` is the windowed PSGRecordingSample matching this sample —
    used by the existence task to look up all bouts of the target activity
    when the answer is "yes".
    """
    regions = []

    if task_type == "existence":
        # Highlight every bout of the target activity in the window when the
        # answer is "yes" so the plot shows what the model is being asked
        # to verify the presence of. Negative samples (target absent) get
        # no regions — that absence IS the answer.
        if metadata.get("is_positive") and recording is not None:
            target = metadata.get("target_activity")
            bouts = recording.activity_index.get(target, [])
            for i, b in enumerate(bouts):
                regions.append((b.start_ms, b.end_ms, f"{target} #{i+1}"))
        return regions

    if task_type == "localization":
        s, e = metadata.get("start_ms"), metadata.get("end_ms")
        if s is not None and e is not None:
            n = metadata.get("ordinal")
            a = metadata.get("activity", "target")
            regions.append((s, e, f"target: {_ord(n)} {a}"))

    elif task_type == "counting":
        # Counting metadata sometimes carries a representative bout
        s, e = metadata.get("start_ms"), metadata.get("end_ms")
        if s is not None and e is not None:
            a = metadata.get("activity", "target")
            regions.append((s, e, f"sample {a}"))

    elif task_type == "ordering":
        s_a, e_a = metadata.get("start_ms"), metadata.get("end_ms")
        n_a = metadata.get("ordinal_a")
        a = metadata.get("activity_a", "A")
        if s_a is not None and e_a is not None:
            regions.append((s_a, e_a, f"A: {_ord(n_a)} {a}"))
        s_b, e_b = metadata.get("start_ms_b"), metadata.get("end_ms_b")
        n_b = metadata.get("ordinal_b")
        b = metadata.get("activity_b", "B")
        if s_b is not None and e_b is not None:
            regions.append((s_b, e_b, f"B: {_ord(n_b)} {b}"))

    elif task_type == "antecedent":
        # The TARGET bout (the one being asked about)
        s, e = metadata.get("start_ms"), metadata.get("end_ms")
        if s is not None and e is not None:
            n = metadata.get("ordinal")
            a = metadata.get("activity", "target")
            regions.append((s, e, f"target: {_ord(n)} {a}"))
        # The ANTECEDENT bout (the answer)
        s_ant = metadata.get("antecedent_start_ms")
        e_ant = metadata.get("antecedent_end_ms")
        if s_ant is not None and e_ant is not None:
            label_ant = metadata.get("antecedent_activity", "antecedent")
            regions.append((s_ant, e_ant, f"antecedent: {label_ant}"))

    elif task_type == "comparison":
        s, e = metadata.get("start_ms"), metadata.get("end_ms")
        if s is not None and e is not None:
            sup = metadata.get("superlative", "target")
            a = metadata.get("activity", "")
            regions.append((s, e, f"{sup} {a}"))

    elif task_type == "multi_hop":
        # ANCHOR bout (referenced in question, not the answer)
        s_anc = metadata.get("anchor_start_ms")
        e_anc = metadata.get("anchor_end_ms")
        if s_anc is not None and e_anc is not None:
            n_anc = metadata.get("anchor_ordinal")
            a_anc = metadata.get("anchor_activity", "anchor")
            regions.append((s_anc, e_anc, f"anchor: {_ord(n_anc)} {a_anc}"))
        # TARGET bout (the answer)
        s, e = metadata.get("start_ms"), metadata.get("end_ms")
        if s is not None and e is not None:
            n_t = metadata.get("target_ordinal")
            a_t = metadata.get("target_activity", "target")
            d = metadata.get("direction", "")
            regions.append((s, e, f"target: {_ord(n_t)} {a_t} ({d})"))

    elif task_type == "state_query":
        # Cross-timeline: highlight the arousal whose context we're asking about
        s = metadata.get("arousal_start_ms")
        e = metadata.get("arousal_end_ms")
        if s is not None and e is not None:
            n = metadata.get("ordinal")
            a = metadata.get("arousal_type", "arousal")
            stage = metadata.get("sleep_stage", "")
            regions.append((s, e, f"{_ord(n)} {a} → {stage}"))

    # existence has no region — answer is global to the window
    return regions


def _zoom_window(regions, window_n_samples):
    """Pick a 5-min slice centered on the answer regions (within the window)."""
    if regions:
        all_s = [r[0] for r in regions]
        all_e = [r[1] for r in regions]
        center_ms = (min(all_s) + max(all_e)) // 2
        center_sample = int(center_ms * SOURCE_HZ / 1000)
        span_ms = max(all_e) - min(all_s)
        context_s = max(PLOT_CONTEXT_SECONDS, (span_ms / 1000) * 2)
        half = int(context_s * SOURCE_HZ / 2)
        start = max(0, center_sample - half)
        end = min(window_n_samples, center_sample + half)
        return start, end
    mid = window_n_samples // 2
    half = int(PLOT_CONTEXT_SECONDS * SOURCE_HZ / 2)
    return max(0, mid - half), min(window_n_samples, mid + half)


def verify_dataset(
    label_class: str = "sleep_stages",
    context_length_s = None,        # None == "full"; CLI parses
    tasks: list = None,
    n_samples: int = 3,
    max_subjects: int = None,
    seed: int = 42,
):
    if tasks is None:
        tasks = get_tasks_for_label_class(label_class)

    ctx_dir = context_dir_name(context_length_s)
    output_dir = SLEEP_PSG_DATA_DIR / "ts_haystack" / label_class / "verification" / ctx_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("Sleep PSG Windowed-Natural Benchmark Visual Verification")
    print(f"  Label class: {label_class}")
    print(f"  Context: {ctx_dir}")
    print(f"  Tasks: {tasks}")
    print(f"  Samples per task: {n_samples}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}")

    split = split_participants(seed=seed)
    subject_ids = split["test"]
    if max_subjects:
        subject_ids = subject_ids[:max_subjects]

    # Build / load the matching window index, then a windowed sampler.
    window_index = SleepPSGWindowIndex.get_or_build(
        label_class=label_class,
        context_length_s=context_length_s,
        subject_ids=subject_ids,
    )
    sampler = RecordingSampler(
        subject_ids=subject_ids,
        label_class=label_class,
        window_index=window_index,
    )
    seed_manager = SeedManager(master_seed=seed)
    template_bank = SleepPromptTemplateBank()

    summary = {}
    for task_name in tasks:
        task_cls = PSG_TASK_REGISTRY.get(task_name)
        if task_cls is None:
            print(f"\n  [skip] Unknown task: {task_name}")
            continue
        if not task_cls.supports_context_length(label_class, context_length_s):
            print(f"\n  [skip] {task_name} not applicable at {ctx_dir}")
            continue

        print(f"\n--- Task: {task_name} ---")
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        generator = task_cls(
            recording_sampler=sampler,
            template_bank=template_bank,
            seed_manager=seed_manager,
            label_class=label_class,
        )

        valid = 0
        failures = 0
        for i in range(n_samples * 20):
            if valid >= n_samples:
                break
            rng = np.random.default_rng(seed + i * 1000 + hash(task_name) % 10000)
            recording = sampler.sample_recording(rng)
            sample = generator.generate_sample(recording, rng)
            if not sample.is_valid:
                failures += 1
                continue
            valid += 1
            idx = valid

            # Slice signal for the entire window from memmap
            mmap = load_subject_signals_mmap(sample.subject_id)
            win_start_sample = int(sample.window_start_ms * SOURCE_HZ / 1000)
            win_end_sample = int(sample.window_end_ms * SOURCE_HZ / 1000)
            win_end_sample = min(win_end_sample, mmap.shape[0])
            window_signals = np.array(mmap[win_start_sample:win_end_sample])
            window_n = window_signals.shape[0]

            # Reload the windowed recording matching this sample. For most
            # tasks this equals `recording` from the loop above, but the
            # existence task picks its own window via per-activity balanced
            # sampling, so we reconstruct from sample.window_start_ms.
            windowed_rec = sampler._load_windowed_recording(
                sample.subject_id, sample.window_start_ms,
            )
            regions = _get_all_regions(
                sample.task_type, sample.metadata, windowed_rec,
            )

            # Window-level needles (window-relative timestamps -> fractions)
            window_needles = []
            for s_ms, e_ms, lbl in regions:
                rs = max(0, int(s_ms * SOURCE_HZ / 1000))
                re_ = min(window_n, int(e_ms * SOURCE_HZ / 1000))
                if re_ > rs:
                    window_needles.append({
                        "activity": lbl,
                        "insert_position_frac": rs / window_n,
                        "duration_samples": re_ - rs,
                    })

            win_start_ts = PSGBaseTaskGenerator._ms_to_timestamp(sample.window_start_ms)
            win_end_ts = PSGBaseTaskGenerator._ms_to_timestamp(sample.window_end_ms)

            img_window = create_psg_plot(
                signals=window_signals,
                channel_names=CHANNEL_NAMES,
                time_range=(win_start_ts, win_end_ts),
                needles=window_needles if window_needles else None,
                annotate_needles=bool(window_needles),
                source_hz=SOURCE_HZ,
                title=f"{task_name} — {sample.subject_id} ({ctx_dir} window, {label_class})",
                question=sample.question,
                answer=sample.answer,
            )
            img_window.save(task_dir / f"sample_{idx:02d}_window.png")

            # Zoomed plot inside the window
            zs, ze = _zoom_window(regions, window_n)
            zoom_signals = window_signals[zs:ze]
            zoom_n = zoom_signals.shape[0]
            zoom_needles = []
            for s_ms, e_ms, lbl in regions:
                rs = int(s_ms * SOURCE_HZ / 1000) - zs
                re_ = int(e_ms * SOURCE_HZ / 1000) - zs
                rs = max(0, rs)
                re_ = min(zoom_n, re_)
                if re_ > rs:
                    zoom_needles.append({
                        "activity": lbl,
                        "insert_position_frac": rs / zoom_n,
                        "duration_samples": re_ - rs,
                    })

            zoom_start_ts = PSGBaseTaskGenerator._ms_to_timestamp(
                sample.window_start_ms + int(zs / SOURCE_HZ * 1000)
            )
            zoom_end_ts = PSGBaseTaskGenerator._ms_to_timestamp(
                sample.window_start_ms + int(ze / SOURCE_HZ * 1000)
            )
            img_zoom = create_psg_plot(
                signals=zoom_signals,
                channel_names=CHANNEL_NAMES,
                time_range=(zoom_start_ts, zoom_end_ts),
                needles=zoom_needles if zoom_needles else None,
                annotate_needles=bool(zoom_needles),
                source_hz=SOURCE_HZ,
                title=f"{task_name} — {sample.subject_id} (zoom, {label_class})",
                question=sample.question,
                answer=sample.answer,
            )
            img_zoom.save(task_dir / f"sample_{idx:02d}_zoomed.png")

            with open(task_dir / f"sample_{idx:02d}.json", "w") as f:
                json.dump(sample.to_dict(), f, indent=2)

            print(f"  [{idx}/{n_samples}] {sample.subject_id}  win={win_start_ts}-{win_end_ts}")
            print(f"    Q: {sample.question}")
            print(f"    A: {sample.answer}")

            del window_signals, zoom_signals

        summary[task_name] = {"valid": valid, "failures": failures}
        print(f"  Generated {valid}/{n_samples} valid ({failures} failures)")

    print(f"\n{'=' * 60}")
    for task, stats in summary.items():
        status = "OK" if stats["valid"] >= n_samples else "INCOMPLETE"
        print(f"  {task:25s}: {stats['valid']}/{n_samples} valid, {stats['failures']} failures [{status}]")
    print(f"\nPlots saved to: {output_dir}")


# Import here to avoid circular imports at module level
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual verification of windowed Sleep PSG benchmark"
    )
    parser.add_argument("--label-class", type=str, default="sleep_stages",
                        choices=["sleep_stages", "arousals"])
    parser.add_argument("--context-length-seconds", type=str, default=None,
                        help="Context length in seconds, or 'full'. "
                             "Default: shortest in DEFAULT_CONTEXT_LENGTHS_S")
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.context_length_seconds is None:
        ctx_token = DEFAULT_CONTEXT_LENGTHS_S[args.label_class][0]
    else:
        ctx_token = args.context_length_seconds
    ctx_s = parse_context_token(ctx_token)

    verify_dataset(
        label_class=args.label_class,
        context_length_s=ctx_s,
        tasks=args.tasks,
        n_samples=args.n_samples,
        max_subjects=args.max_subjects,
        seed=args.seed,
    )
