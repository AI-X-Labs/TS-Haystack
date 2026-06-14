# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Anomaly detection: "Does this window contain any V or A beat?" → boolean.

Gated off at 1h+ where every long LTAF window contains V/A beats.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.datasets.ltaf_haystack.core.data_structures import (
    LTAFGeneratedSample,
    LTAFRecordingSample,
)
from src.datasets.ltaf_haystack.tasks.base_task import LTAFBaseTaskGenerator


class LTAFAnomalyDetectionTaskGenerator(LTAFBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "anomaly_detection"

    @property
    def answer_type(self) -> str:
        return "boolean"

    # Gated off at 1h/2h (saturated — every long window contains V or A beats).
    _MAX_CTX_S = 900  # 15 min

    @classmethod
    def supports_context_length(
        cls, label_class: str, context_length_s: Optional[float]
    ) -> bool:
        if context_length_s is None:
            return False
        return context_length_s <= cls._MAX_CTX_S

    def _generate(
        self, recording: LTAFRecordingSample, rng: np.random.Generator
    ):
        target_symbols = ("V", "A")
        activity = target_symbols[int(rng.integers(0, len(target_symbols)))]
        anomaly_beat_samples = [
            int(b.sample) for b in recording.beats_timeline if b.symbol == activity
        ]
        has_anomaly = bool(anomaly_beat_samples)

        question, answer = self.template_bank.sample(
            task="anomaly_detection",
            rng=rng,
            target_activity=activity,
            symbol=activity,
            activity=activity,
            answer=has_anomaly,
        )
        return self._build_sample(
            recording,
            question,
            answer,
            metadata={
                "target_symbol": activity,
                "has_anomaly": has_anomaly,
                "anomaly_beat_samples": anomaly_beat_samples,
            },
        )

