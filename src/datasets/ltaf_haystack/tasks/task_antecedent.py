# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Antecedent: "Which rhythm directly preceded the Nth {activity}?" → category.

Uses the rhythm timeline to find the bout that ends immediately before
the target bout's start.
"""

from __future__ import annotations

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


class LTAFAntecedentTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "antecedent"

    @property
    def answer_type(self) -> str:
        return "category"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        timeline = sorted(recording.rhythm_timeline, key=lambda b: b.start_sample)
        if len(timeline) < 2:
            return self._create_invalid_sample(
                "Need ≥2 rhythm bouts in window", recording=recording
            )

        regime = set(get_activities_list(self.label_class))
        candidates = [
            (i, b) for i, b in enumerate(timeline)
            if i > 0 and b.activity in regime
        ]
        if not candidates:
            return self._create_invalid_sample(
                "No non-first target bout of regime activity", recording=recording
            )

        idx, target_bout = candidates[int(rng.integers(0, len(candidates)))]
        antecedent_bout = timeline[idx - 1]

        # Count which ordinal of target_bout.activity this is
        same_prior = [
            b for b in timeline[:idx] if b.activity == target_bout.activity
        ]
        n = len(same_prior) + 1  # 1-indexed

        answer = antecedent_bout.activity
        question, _ = self.template_bank.sample(
            task="antecedent",
            rng=rng,
            nth=_ordinal(n),
            target_activity=target_bout.activity,
            activity=target_bout.activity,
            answer=answer,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "target_activity": target_bout.activity,
                "ordinal": n,
                "antecedent_activity": antecedent_bout.activity,
                # Answer region = the antecedent bout (the rhythm we report).
                "start_sample": int(antecedent_bout.start_sample),
                "end_sample": int(antecedent_bout.end_sample),
                # Context region = the target bout the question refers to.
                "context_start_sample": int(target_bout.start_sample),
                "context_end_sample": int(target_bout.end_sample),
            },
        )

