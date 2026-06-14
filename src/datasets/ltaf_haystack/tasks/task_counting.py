# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Counting task: "How many {activity} bouts?" → integer.

Stratified sampling mirrors existence: pick an activity, then pick a target
count bucket (0, 1, 2–4, 5+), and draw a window matching the bucket via the
sampler's presence pools (rejection-sampled for ≥1 buckets). Uniform-window
sampling produces ~80% zero-count samples because 15-min LTAF windows are
almost always dominated by a single rhythm.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator


# Count buckets the task stratifies over. ``None`` upper bound means unbounded.
_COUNT_BUCKETS: Tuple[Tuple[int, Optional[int]], ...] = (
    (0, 0),
    (1, 1),
    (2, 4),
    (5, None),
)
_MAX_REJECTION_TRIES = 50


class LTAFCountingTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "counting"

    @property
    def answer_type(self) -> str:
        return "integer"

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        activities = get_activities_list(self.label_class)
        activity = activities[int(rng.integers(0, len(activities)))]

        bucket_idx = int(rng.integers(0, len(_COUNT_BUCKETS)))
        lo, hi = _COUNT_BUCKETS[bucket_idx]
        sampler = self.recording_sampler

        if lo == 0 and hi == 0:
            win = sampler.sample_recording_for_activity(
                activity=activity, want_present=False, rng=rng
            )
            if win is None:
                # Every window contains this activity (rare). Fall through to
                # presence sampling and take whatever count we get.
                win = sampler.sample_recording_for_activity(
                    activity=activity, want_present=True, rng=rng
                )
        else:
            win = None
            last_present = None
            for _ in range(_MAX_REJECTION_TRIES):
                cand = sampler.sample_recording_for_activity(
                    activity=activity, want_present=True, rng=rng
                )
                if cand is None:
                    break
                last_present = cand
                c = len(cand.activity_index.get(activity, []))
                if c >= lo and (hi is None or c <= hi):
                    win = cand
                    break
            # Fall back to the last presence draw if the bucket was unreachable.
            if win is None:
                win = last_present

        if win is None:
            return self._create_invalid_sample(
                f"No window pool for activity={activity}", recording=recording
            )

        bouts = win.activity_index.get(activity, [])
        count = len(bouts)

        question, answer = self.template_bank.sample(
            task="counting", rng=rng, activity=activity, answer=count, count=count
        )
        meta = {
            "activity": activity,
            "count": count,
            "target_bucket": [lo, hi],
            # Every counted bout — the verifier paints one band per entry so
            # a reviewer can literally count the highlights against `count`.
            "bout_segments": [
                [int(b.start_sample), int(b.end_sample)] for b in bouts
            ],
        }
        return self._build_sample(win, question, answer, metadata=meta)

