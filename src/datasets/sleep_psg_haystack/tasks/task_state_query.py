# SPDX-License-Identifier: CC-BY-NC-4.0

"""
State Query Task: "What sleep stage was the subject in when the Nth {arousal} occurred?"

Cross-timeline task (sleep_stages label_class only): uses the arousals
timeline as targets and the sleep stages timeline to find the answer.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.activity_regimes import AROUSAL_ACTIVITIES
from src.datasets.sleep_psg_haystack.core.prompt_templates import ordinal
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGStateQueryTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "state_query"

    @property
    def answer_type(self) -> str:
        return "category"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Build arousal index from the arousals timeline
        arousal_index = {}
        for bout in recording.arousals_timeline:
            if bout.activity in AROUSAL_ACTIVITIES:
                arousal_index.setdefault(bout.activity, []).append(bout)

        # Pick an arousal type with at least 1 bout
        candidates = [a for a, bouts in arousal_index.items() if len(bouts) >= 1]
        if not candidates:
            return self._create_invalid_sample("No viable arousals found", recording)

        arousal_type = candidates[rng.integers(0, len(candidates))]
        arousal_bouts = arousal_index[arousal_type]
        n = rng.integers(1, len(arousal_bouts) + 1)
        arousal_bout = arousal_bouts[n - 1]

        # Find sleep stage at the arousal's midpoint
        midpoint_ms = (arousal_bout.start_ms + arousal_bout.end_ms) // 2
        sleep_stage = self._find_stage_at(recording.sleep_stages_timeline, midpoint_ms)

        if sleep_stage is None:
            return self._create_invalid_sample(
                f"No sleep stage annotation at arousal midpoint {midpoint_ms}ms", recording,
            )

        question, answer = self.template_bank.sample(
            task="state_query", rng=rng,
            nth=ordinal(int(n)), activity=arousal_type,
            sleep_stage=sleep_stage,
        )

        return PSGGeneratedSample(
            subject_id=recording.subject_id,
            recording_duration_ms=recording.recording_duration_ms,
            label_class=self.label_class,
            task_type=self.task_name,
            **self._window_fields(recording),
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            metadata={
                "arousal_type": arousal_type,
                "ordinal": int(n),
                "arousal_start_ms": arousal_bout.start_ms,
                "arousal_end_ms": arousal_bout.end_ms,
                "sleep_stage": sleep_stage,
            },
        )

    @staticmethod
    def _find_stage_at(stages_timeline: list, timestamp_ms: int) -> str | None:
        """Find the sleep stage annotation covering a given timestamp."""
        for bout in stages_timeline:
            if bout.start_ms <= timestamp_ms < bout.end_ms:
                return bout.activity
        return None
