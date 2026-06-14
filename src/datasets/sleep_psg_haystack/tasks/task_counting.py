# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Counting Task: "How many {activity} bouts are there?"

Picks a random activity and asks for the count. Includes absent
activities (answer=0) when possible for variety.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.activity_regimes import get_all_activities
from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGCountingTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "counting"

    @property
    def answer_type(self) -> str:
        return "integer"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        all_activities = list(get_all_activities(self.label_class))
        if not all_activities:
            return self._create_invalid_sample("No activities defined", recording)

        activity = all_activities[rng.integers(0, len(all_activities))]
        bouts = recording.activity_index.get(activity, [])
        count = len(bouts)

        question, answer = self.template_bank.sample(
            task="counting", rng=rng,
            activity=activity, count=count,
        )

        # Include a representative bout position for plotting
        meta = {"activity": activity, "count": count}
        if bouts:
            mid_bout = bouts[len(bouts) // 2]
            meta["start_ms"] = mid_bout.start_ms
            meta["end_ms"] = mid_bout.end_ms

        return PSGGeneratedSample(
            subject_id=recording.subject_id,
            recording_duration_ms=recording.recording_duration_ms,
            label_class=self.label_class,
            task_type=self.task_name,
            **self._window_fields(recording),
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            metadata=meta,
        )
