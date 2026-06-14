# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Comparison: extremum × polarity reasoning over a single activity.

Four question variants per sample:
  * ``(longest,  with)``    — longest bout of activity X.
  * ``(shortest, with)``    — shortest bout of activity X.
  * ``(longest,  without)`` — longest stretch of the window with no X
                               (edge gaps at window start/end count).
  * ``(shortest, without)`` — shortest such stretch.

The answer is a ``HH:MM:SS-HH:MM:SS`` time range. Selection requires the
chosen extremum to be **strictly unique** among candidate durations so
the answer is unambiguous; otherwise the sample is rejected.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.core.ltaf_prompt_templates import (
    COMPARISON_TEMPLATES,
    pick_vocab,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator


def _unique_extremum_index(values: List[int], extremum: str) -> Optional[int]:
    """Return argmax/argmin only when the extremum is strictly unique."""
    if not values:
        return None
    if extremum == "longest":
        target = max(values)
    else:
        target = min(values)
    hits = [i for i, v in enumerate(values) if v == target]
    return hits[0] if len(hits) == 1 else None


def _without_gaps(
    bouts, ctx_samples: int
) -> List[Tuple[int, int, int]]:
    """Non-zero gaps where the activity is NOT present.

    Edge gaps are included; zero-length gaps (back-to-back bouts or a
    bout flush with the window boundary) are dropped so they cannot win
    the ``shortest`` query trivially.
    Returns a list of ``(start_sample, end_sample, duration)``.
    """
    out: List[Tuple[int, int, int]] = []
    if not bouts:
        return out

    # Left edge gap
    left_end = bouts[0].start_sample
    if left_end > 0:
        out.append((0, int(left_end), int(left_end)))

    # Interior gaps
    for prev, nxt in zip(bouts, bouts[1:]):
        s = int(prev.end_sample)
        e = int(nxt.start_sample)
        if e > s:
            out.append((s, e, e - s))

    # Right edge gap
    right_start = int(bouts[-1].end_sample)
    if ctx_samples > right_start:
        out.append((right_start, int(ctx_samples), int(ctx_samples - right_start)))

    return out


class LTAFComparisonTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "comparison"

    @property
    def answer_type(self) -> str:
        return "time_range"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        extremum = ("longest", "shortest")[int(rng.integers(0, 2))]
        polarity = ("with", "without")[int(rng.integers(0, 2))]

        ctx_samples = int(recording.duration_samples)
        activities = [
            a for a in get_activities_list(self.label_class)
            if recording.activity_index.get(a)
        ]

        candidates: List[Tuple[str, int, int, int]] = []
        for activity in activities:
            bouts = recording.activity_index[activity]

            if polarity == "with":
                # For "with" we pick a single bout as the answer — require
                # the bout to have both its natural onset and offset in the
                # window so the time range is a real ECG event, not a
                # window-edge artifact.
                fully_in = [
                    b for b in bouts
                    if not b.clipped_left and not b.clipped_right
                ]
                if len(fully_in) < 2:
                    continue
                durations = [int(b.duration_samples) for b in fully_in]
                idx = _unique_extremum_index(durations, extremum)
                if idx is None:
                    continue
                b = fully_in[idx]
                candidates.append(
                    (activity, int(b.start_sample), int(b.end_sample), durations[idx])
                )
            else:
                gaps = _without_gaps(bouts, ctx_samples)
                if not gaps:
                    continue
                durs = [d for *_, d in gaps]
                idx = _unique_extremum_index(durs, extremum)
                if idx is None:
                    continue
                s, e, d = gaps[idx]
                candidates.append((activity, s, e, d))

        if not candidates:
            return self._create_invalid_sample(
                f"No activity supports ({extremum}, {polarity}) with a unique extremum",
                recording=recording,
            )

        activity, start_sample, end_sample, duration_samples = candidates[
            int(rng.integers(0, len(candidates)))
        ]

        hz = int(recording.source_hz)
        start_ms = int(round(start_sample * 1000.0 / hz))
        end_ms = int(round(end_sample * 1000.0 / hz))
        answer_str = (
            f"{self._ms_to_timestamp(start_ms)}-{self._ms_to_timestamp(end_ms)}"
        )

        templates = COMPARISON_TEMPLATES[(extremum, polarity)]
        template = templates[int(rng.integers(0, len(templates)))]
        activity_name = pick_vocab(activity, rng)
        question = template.format(activity_name=activity_name)

        return self._build_sample(
            recording,
            question,
            answer_str,
            metadata={
                "activity": activity,
                "extremum": extremum,
                "polarity": polarity,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "duration_samples": int(duration_samples),
            },
        )