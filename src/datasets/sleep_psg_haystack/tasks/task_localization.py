# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Localization Task: "When did the Nth {activity} bout occur?"

Picks a random activity present in the recording, selects a random
ordinal, and asks for the time range of that bout.
"""

import numpy as np

from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.core.prompt_templates import ordinal
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGLocalizationTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "localization"

    @property
    def answer_type(self) -> str:
        return "time_range"

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        # Get activities with at least 1 NON-CLIPPED bout. Bouts whose original
        # start was before the window start (or end past the window end) have
        # artifactual timestamps and would let a model memorize "answer = 0".
        candidates = []
        for a, bouts in recording.activity_index.items():
            clipped = recording.clipped_keys.get(a, set())
            usable = [(i, b) for i, b in enumerate(bouts) if i not in clipped]
            if usable:
                candidates.append((a, usable))
        if not candidates:
            return self._create_invalid_sample("No non-clipped activities", recording)

        activity, usable = candidates[rng.integers(0, len(candidates))]
        # Pick a random usable bout, but report its 1-based ordinal *within
        # the activity index* (not within `usable`) so question semantics stay
        # consistent with the window's full bout numbering.
        pick_idx = int(rng.integers(0, len(usable)))
        bout_idx, bout = usable[pick_idx]
        n = bout_idx + 1

        start_ts, end_ts = self._bout_time_range(bout)
        nth = ordinal(int(n))

        question, answer = self.template_bank.sample(
            task="localization", rng=rng,
            nth=nth, activity=activity, start=start_ts, end=end_ts,
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
                "activity": activity,
                "ordinal": int(n),
                "start_ms": bout.start_ms,
                "end_ms": bout.end_ms,
            },
        )
