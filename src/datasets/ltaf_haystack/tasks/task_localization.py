# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Localization: "When did the Nth {activity} occur?" → time range."""

from __future__ import annotations

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


class LTAFLocalizationTaskGenerator(LTAFBaseTaskGenerator):

    # Answer bout must not cover more than this fraction of the window,
    # otherwise localization degenerates into "point anywhere in the window".
    MAX_ANSWER_FRACTION: float = 0.30

    @property
    def task_name(self) -> str:
        return "localization"

    @property
    def answer_type(self) -> str:
        return "time_range"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        ctx_samples = int(recording.duration_samples)
        max_dur = int(ctx_samples * self.MAX_ANSWER_FRACTION)

        def _legal(bout) -> bool:
            # Bout must have both natural onset and offset inside the
            # window (not clipped at either edge) AND fit under the
            # answer-fraction cap.
            return (
                not bout.clipped_left
                and not bout.clipped_right
                and bout.duration_samples <= max_dur
            )

        activities = [
            a for a in get_activities_list(self.label_class)
            if any(_legal(b) for b in recording.activity_index.get(a, []))
        ]
        if not activities:
            return self._create_invalid_sample(
                "No bout with both boundaries inside window under cap",
                recording=recording,
            )
        activity = activities[int(rng.integers(0, len(activities)))]
        # Keep the natural ordinal index relative to ALL bouts of this
        # activity (so "1st NSR" still means the first NSR episode seen,
        # not the first legal one), but only allow picking an ordinal
        # whose bout passes the legality test.
        all_bouts = recording.activity_index[activity]
        legal_ns = [i + 1 for i, b in enumerate(all_bouts) if _legal(b)]
        n = int(legal_ns[int(rng.integers(0, len(legal_ns)))])
        bout = all_bouts[n - 1]

        hz = recording.source_hz
        start_ms = int(round(bout.start_sample * 1000.0 / hz))
        end_ms = int(round(bout.end_sample * 1000.0 / hz))
        answer_str = (
            f"{self._ms_to_timestamp(start_ms)}-{self._ms_to_timestamp(end_ms)}"
        )

        question, answer = self.template_bank.sample(
            task="localization",
            rng=rng,
            nth=_ordinal(n),
            activity=activity,
            answer=answer_str,
            start_time=self._ms_to_timestamp(start_ms),
            end_time=self._ms_to_timestamp(end_ms),
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "activity": activity,
                "ordinal": n,
                "start_sample": int(bout.start_sample),
                "end_sample": int(bout.end_sample),
            },
        )

