# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Existence task: "Is there {activity} in this ECG window?" → boolean.

Per-activity balanced sampling: pick a target rhythm, coin-flip whether to
require its presence, then query the sampler's (activity, presence) pool.
Gated off at 2h where the label saturates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.datasets.ltaf_haystack.core.activity_regimes import get_activities_list
from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator


class LTAFExistenceTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "existence"

    @property
    def answer_type(self) -> str:
        return "boolean"

    # Existence at 2h saturates — every 2h LTAF window contains ≥4 rhythms,
    # so the answer is always "yes" for most rhythms.
    _MAX_CTX_S = 3600

    @classmethod
    def supports_context_length(
        cls, label_class: str, context_length_s: Optional[float]
    ) -> bool:
        if context_length_s is None:
            return False
        return context_length_s <= cls._MAX_CTX_S

    def _generate(
        self, recording: LTAFRecordingSample, rng: np.random.Generator
    ) -> LTAFGeneratedSample:
        regime = get_activities_list(self.label_class)
        target = regime[int(rng.integers(0, len(regime)))]
        is_positive = bool(rng.random() < 0.5)

        sampler = self.recording_sampler
        win = sampler.sample_recording_for_activity(
            activity=target, want_present=is_positive, rng=rng
        )
        if win is None:
            # Fallback: try the opposite label if the primary pool is empty.
            win = sampler.sample_recording_for_activity(
                activity=target, want_present=not is_positive, rng=rng
            )
            if win is None:
                return self._create_invalid_sample(
                    f"No existence pool for activity={target}", recording=recording
                )
            is_positive = not is_positive

        question, answer = self.template_bank.sample(
            task="existence", rng=rng, activity=target, answer=is_positive
        )
        return self._build_sample(
            recording=win,
            question=question,
            answer=answer,
            metadata={"target_activity": target, "is_positive": is_positive},
        )
