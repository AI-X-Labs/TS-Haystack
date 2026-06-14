# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Ordering: "Did the Nth {A} occur before the Mth {B}?" → boolean.

Picks two activities both present in the recording, picks an ordinal for
each from their natural counts, and evaluates the ordering by comparing
bout start samples.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


class LTAFOrderingTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "ordering"

    @property
    def answer_type(self) -> str:
        return "boolean"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        present: List[str] = [
            a for a in get_activities_list(self.label_class)
            if recording.activity_index.get(a)
        ]
        if len(present) < 2:
            return self._create_invalid_sample(
                f"Need ≥2 rhythms present; have {len(present)}", recording=recording
            )

        a_idx = int(rng.integers(0, len(present)))
        activity_a = present[a_idx]
        b_candidates = [a for a in present if a != activity_a]
        activity_b = b_candidates[int(rng.integers(0, len(b_candidates)))]

        bouts_a = recording.activity_index[activity_a]
        bouts_b = recording.activity_index[activity_b]
        n_a = int(rng.integers(1, len(bouts_a) + 1))
        n_b = int(rng.integers(1, len(bouts_b) + 1))
        bout_a = bouts_a[n_a - 1]
        bout_b = bouts_b[n_b - 1]

        a_before_b = bout_a.start_sample < bout_b.start_sample

        question, answer = self.template_bank.sample(
            task="ordering",
            rng=rng,
            nth=_ordinal(n_a),
            mth=_ordinal(n_b),
            activity_a=activity_a,
            activity_b=activity_b,
            answer=a_before_b,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "activity_a": activity_a,
                "activity_b": activity_b,
                "ordinal_a": n_a,
                "ordinal_b": n_b,
                "a_start": int(bout_a.start_sample),
                "a_end": int(bout_a.end_sample),
                "b_start": int(bout_b.start_sample),
                "b_end": int(bout_b.end_sample),
                # The ordering question references two bouts; paint both
                # via the multi-segment key (both rendered as "answer").
                "bout_segments": [
                    [int(bout_a.start_sample), int(bout_a.end_sample)],
                    [int(bout_b.start_sample), int(bout_b.end_sample)],
                ],
            },
        )

