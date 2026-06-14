# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Multi-hop: "When did the Kth {target} occur {direction} the Nth {anchor}?" → time range.

Natural variant: pick an anchor activity present in the window, pick
ordinal N of that activity, then pick a target activity present in the
window and find the Kth bout of target activity that is before/after
the anchor bout.
"""

from __future__ import annotations

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator, _ordinal


class LTAFMultiHopTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "multi_hop"

    @property
    def answer_type(self) -> str:
        return "time_range"

    @classmethod
    def supports_context_length(cls, label_class, context_length_s):
        if context_length_s is None:
            return False
        return context_length_s >= 100

    def _generate(self, recording: LTAFRecordingSample, rng: np.random.Generator):
        present = [
            a for a in get_activities_list(self.label_class)
            if recording.activity_index.get(a)
        ]
        if len(present) < 2:
            return self._create_invalid_sample(
                "Multi-hop needs ≥2 activities present", recording=recording
            )

        anchor = present[int(rng.integers(0, len(present)))]
        target_candidates = [a for a in present if a != anchor]
        target = target_candidates[int(rng.integers(0, len(target_candidates)))]
        direction = "after" if bool(rng.random() < 0.5) else "before"

        anchor_bouts = recording.activity_index[anchor]
        target_bouts = recording.activity_index[target]
        n = int(rng.integers(1, len(anchor_bouts) + 1))
        anchor_bout = anchor_bouts[n - 1]

        if direction == "after":
            viable = [b for b in target_bouts if b.start_sample >= anchor_bout.end_sample]
        else:
            viable = [b for b in target_bouts if b.end_sample <= anchor_bout.start_sample]

        if not viable:
            return self._create_invalid_sample(
                f"No {target} {direction} {_ordinal(n)} {anchor}", recording=recording
            )
        k = int(rng.integers(1, len(viable) + 1))
        result_bout = viable[k - 1]

        hz = recording.source_hz
        start_ms = int(round(result_bout.start_sample * 1000.0 / hz))
        end_ms = int(round(result_bout.end_sample * 1000.0 / hz))
        answer_str = f"{self._ms_to_timestamp(start_ms)}-{self._ms_to_timestamp(end_ms)}"

        question, answer = self.template_bank.sample(
            task="multi_hop",
            rng=rng,
            kth=_ordinal(k),
            nth=_ordinal(n),
            direction=direction,
            target=target,
            anchor=anchor,
            first_activity=anchor,
            second_activity=target,
            answer=answer_str,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "anchor_activity": anchor,
                "anchor_ordinal": n,
                "target_activity": target,
                "target_ordinal": k,
                "direction": direction,
                # Answer bout (painted as "answer" in verify).
                "start_sample": int(result_bout.start_sample),
                "end_sample": int(result_bout.end_sample),
                # Anchor bout is a question-context region, not the answer.
                "context_start_sample": int(anchor_bout.start_sample),
                "context_end_sample": int(anchor_bout.end_sample),
            },
        )

