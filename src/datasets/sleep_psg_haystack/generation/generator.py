# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Generation pipeline for the windowed-natural Sleep PSG benchmark.

For each context length:
  1. Build / load a per-(subject, ctx) window index of valid window starts.
  2. Wrap a windowed RecordingSampler over the split's subjects.
  3. For each task, generate samples by drawing (subject, window) pairs and
     constructing window-relative QA from natural annotations.

No needle insertion. Signals are loaded on-demand at evaluation time via the
loader's memmap helpers using (subject_id, window_start_ms, window_end_ms).

Usage:
    .venv/bin/python3 -m src.datasets.sleep_psg_haystack.generation.generator \
        --label-class sleep_stages \
        --context-lengths-seconds 900 3600 7200 full
"""

import argparse
import gc
from typing import Dict, List, Optional

from src.datasets.sleep_psg_haystack.loader import SLEEP_PSG_DATA_DIR
from src.datasets.sleep_psg_haystack.core.timeline_builder import SleepPSGTimelineBuilder
from src.datasets.sleep_psg_haystack.core.recording_sampler import RecordingSampler
from src.datasets.sleep_psg_haystack.core.window_index import SleepPSGWindowIndex
from src.datasets.sleep_psg_haystack.core.participant_splitter import get_or_create_split
from src.datasets.sleep_psg_haystack.core.prompt_templates import SleepPromptTemplateBank
from src.datasets.sleep_psg_haystack.tasks import (
    PSG_TASK_REGISTRY,
    get_psg_task_generator,
    get_tasks_for_label_class,
)

from src.datasets.capture24_haystack.core.seed_manager import SeedManager
from src.datasets.capture24_haystack.utils import format_context_dir


# Default context lengths per label class (in seconds). Aligned with the
# Capture24 ts_haystack grid (2.56s, 10s, 100s, 15min, 1h, 2h) so cross-dataset
# comparisons share scales. Use the special token "full" for whole-recording.
DEFAULT_CONTEXT_LENGTHS_S: Dict[str, List] = {
    "sleep_stages": [900, 3600, 7200, "full"],          # 15min, 1h, 2h, full
    "arousals":     [100, 900, 3600, "full"],           # 100s, 15min, 1h, full
}


def parse_context_token(tok) -> Optional[float]:
    """Convert a CLI context token to seconds (float) or None for 'full'."""
    if isinstance(tok, str) and tok.lower() == "full":
        return None
    return float(tok)


def context_dir_name(context_length_s: Optional[float]) -> str:
    if context_length_s is None:
        return "full"
    return format_context_dir(context_length_s)


class SleepPSGGenerator:
    """
    Orchestrates the windowed-natural Sleep PSG generation pipeline.
    """

    def __init__(
        self,
        label_class: str = "sleep_stages",
        seed: int = 42,
    ):
        self.label_class = label_class
        self.seed = seed
        self.split: Dict[str, List[str]] = {}

    def build_timelines(self, n_jobs: int = 1) -> None:
        """Build per-subject timelines from raw WFDB annotations (idempotent)."""
        print("Building timelines...")
        for lc in ["sleep_stages", "arousals"]:
            builder = SleepPSGTimelineBuilder(label_class=lc)
            builder.build_all_timelines(n_jobs=n_jobs)
        print("Timelines built.")

    def load_split(self) -> Dict[str, List[str]]:
        self.split = get_or_create_split(seed=self.seed)
        for s, ids in self.split.items():
            print(f"  {s}: {len(ids)} subjects")
        return self.split

    def generate(
        self,
        tasks: List[str],
        context_lengths_s: List[Optional[float]],
        samples_per_split: Dict[str, int],
        n_jobs: int = 1,
        overwrite: bool = False,
    ) -> None:
        if not self.split:
            self.load_split()

        seed_manager = SeedManager(master_seed=self.seed)
        template_bank = SleepPromptTemplateBank()
        output_dir = SLEEP_PSG_DATA_DIR / "ts_haystack" / self.label_class / "tasks"

        # All subjects in any split — used to build a global window index that
        # then gets sliced per-split inside RecordingSampler.
        all_subjects = sorted(set(
            sid for ids in self.split.values() for sid in ids
        ))

        for ctx_s in context_lengths_s:
            ctx_label = context_dir_name(ctx_s)
            print(f"\n{'#'*60}")
            print(f"# Context length: {ctx_label}")
            print(f"{'#'*60}")

            # Build (or load) the window index for this context length.
            print(f"Loading/building window index ({self.label_class}, {ctx_label})...")
            window_index = SleepPSGWindowIndex.get_or_build(
                label_class=self.label_class,
                context_length_s=ctx_s,
                subject_ids=all_subjects,
            )
            print(f"  {window_index.n_subjects()} subjects, "
                  f"{window_index.total_windows()} candidate windows")

            for task_name in tasks:
                task_cls = get_psg_task_generator(task_name)
                if not task_cls.supports_context_length(self.label_class, ctx_s):
                    print(f"\n[skip] {task_name} not applicable at {ctx_label}")
                    continue

                print(f"\n{'='*60}")
                print(f"Task: {task_name} ({self.label_class}, {ctx_label})")
                print(f"{'='*60}")

                for split_name, n_samples in samples_per_split.items():
                    expected_path = output_dir / ctx_label / task_name / split_name / "data.parquet"
                    if not overwrite and expected_path.exists():
                        print(f"  [skip] {split_name} already exists at {expected_path}")
                        continue

                    subject_ids = self.split.get(split_name, [])
                    if not subject_ids:
                        print(f"  [skip] No subjects for split '{split_name}'")
                        continue

                    sampler = RecordingSampler(
                        subject_ids=subject_ids,
                        label_class=self.label_class,
                        window_index=window_index,
                    )
                    if not sampler._indexed_pairs:
                        print(f"  [skip] No indexed windows for split '{split_name}' at {ctx_label}")
                        continue

                    generator = task_cls(
                        recording_sampler=sampler,
                        template_bank=template_bank,
                        seed_manager=seed_manager,
                        label_class=self.label_class,
                    )

                    print(f"  Generating {n_samples} {split_name} samples...")
                    samples = generator.generate_dataset(
                        n_samples=n_samples,
                        split=split_name,
                        n_jobs=n_jobs,
                    )
                    path = generator.save_dataset(
                        samples=samples,
                        split=split_name,
                        output_dir=output_dir,
                        context_dir=ctx_label,
                    )
                    print(f"  Saved {len(samples)} samples to {path}")
                    del samples, sampler
                    gc.collect()

            del window_index
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate windowed-natural Sleep PSG benchmark"
    )
    parser.add_argument("--label-class", type=str, default="sleep_stages",
                        choices=["sleep_stages", "arousals"])
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=None,
                        choices=list(PSG_TASK_REGISTRY.keys()),
                        help="Tasks to generate (default: all applicable for label class)")
    parser.add_argument("--context-lengths-seconds", type=str, nargs="+", default=None,
                        help="Context lengths in seconds, e.g. '900 3600 7200 full'. "
                             "Default per label_class.")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Train samples (val/test fixed at 150)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--build-only", action="store_true",
                        help="Only build timelines + window indexes; skip generation")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to generate (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    gen = SleepPSGGenerator(
        label_class=args.label_class,
        seed=args.seed,
    )

    gen.build_timelines(n_jobs=args.n_jobs)
    gen.load_split()

    raw_ctx = args.context_lengths_seconds or DEFAULT_CONTEXT_LENGTHS_S[args.label_class]
    ctx_lengths_s = [parse_context_token(t) for t in raw_ctx]

    if args.build_only:
        all_subjects = sorted(set(
            sid for ids in gen.split.values() for sid in ids
        ))
        for ctx_s in ctx_lengths_s:
            print(f"\nBuilding window index for {context_dir_name(ctx_s)}...")
            idx = SleepPSGWindowIndex.get_or_build(
                label_class=args.label_class,
                context_length_s=ctx_s,
                subject_ids=all_subjects,
            )
            print(f"  {idx.n_subjects()} subjects, {idx.total_windows()} windows")
        print("Build complete.")
    else:
        tasks = args.tasks or get_tasks_for_label_class(args.label_class)
        gen.generate(
            tasks=tasks,
            context_lengths_s=ctx_lengths_s,
            samples_per_split={
                s: (args.n_samples if s == "train" else 150)
                for s in args.splits
            },
            overwrite=args.overwrite,
        )
        print("\nDone!")
