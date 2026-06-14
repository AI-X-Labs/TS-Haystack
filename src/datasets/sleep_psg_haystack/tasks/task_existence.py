# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Existence Task: "Is there {activity} in this window?"

Window-natural version. At full-recording length almost every regime activity
is present in every subject (e.g. ~99% of subjects show all 5 sleep stages),
which makes the task trivially "yes". At shorter windows (15 min - 2 h for
sleep stages, 100 s - 15 min for arousals), many regime activities are
naturally absent, restoring 50/50 balance without any synthetic insertion.

Algorithm — **per-activity balanced** (target-first):
  1. Pick the target activity uniformly from the regime.
  2. Coin-flip the desired answer (Yes/No) → 50/50 balance per activity.
  3. Ask the sampler for a window where that activity is (present | absent).
     Sampler uses pre-built per-(activity, presence) pools from the window
     index → O(1) draw, no rejection sampling.
  4. Generate Q/A.

This decouples target choice from label choice. With the naive variant
(coin-flip first, sample target second) the conditional distributions
P(target | label) reflect the wildly different base rates of each activity
(e.g. mixed_apnea is in ~5% of windows), so a model that memorizes the
target name → answer mapping can shortcut the task. Per-activity balancing
removes that shortcut: each (target, label) cell holds ~1/(2*|regime|) of
the dataset.

Gated via `supports_context_length` for label classes / context lengths
where some regime activities are present in nearly every window (so the
"absent" pool is empty and per-activity balance becomes impossible).
"""

from typing import Optional

import numpy as np

from src.datasets.sleep_psg_haystack.core.activity_regimes import get_all_activities
from src.datasets.sleep_psg_haystack.core.data_structures import (
    PSGGeneratedSample,
    PSGRecordingSample,
)
from src.datasets.sleep_psg_haystack.tasks.base_task import PSGBaseTaskGenerator


class PSGExistenceTaskGenerator(PSGBaseTaskGenerator):

    @property
    def task_name(self) -> str:
        return "existence"

    @property
    def answer_type(self) -> str:
        return "boolean"

    # Context-length gating per label class.
    #   - sleep_stages: enabled up to 2h. At "full" >90% of subjects have all
    #     5 stages, so answers degenerate to "yes".
    #   - arousals: enabled up to 15min. At ≥1h, rera/hypopnea are present in
    #     ~95% of subjects, so answers degenerate to "yes".
    _MAX_CTX_S = {
        "sleep_stages": 2 * 3600,    # 2 h
        "arousals": 15 * 60,         # 15 min
    }

    @classmethod
    def supports_context_length(
        cls, label_class: str, context_length_s: Optional[float],
    ) -> bool:
        if context_length_s is None:  # full recording
            return False
        cap = cls._MAX_CTX_S.get(label_class)
        if cap is None:
            return False
        return context_length_s <= cap

    def generate_sample(
        self,
        recording: PSGRecordingSample,
        rng: np.random.Generator,
    ) -> PSGGeneratedSample:
        """
        Per-activity balanced existence sampling.

        IMPORTANT: the `recording` argument passed in by the base task
        generator is **ignored**. Existence picks the target activity *first*,
        then asks the sampler for a window matching the desired (target, label)
        combo. This breaks the (target → label) shortcut that arises if you
        sample a window first and then pick a target conditional on its
        present/absent sets.
        """
        regime = sorted(get_all_activities(self.label_class))
        target = regime[int(rng.integers(0, len(regime)))]
        is_positive = bool(rng.random() < 0.5)

        sampler = self.recording_sampler
        win_recording = sampler.sample_recording_for_activity(
            activity=target, want_present=is_positive, rng=rng,
        )
        if win_recording is None:
            # Try the opposite label as a fallback before giving up.
            win_recording = sampler.sample_recording_for_activity(
                activity=target, want_present=not is_positive, rng=rng,
            )
            if win_recording is None:
                return self._create_invalid_sample(
                    f"No window with activity={target} (either presence)", recording,
                )
            is_positive = not is_positive

        question, answer = self.template_bank.sample(
            task="existence", rng=rng,
            activity=target, exists=is_positive,
        )

        return PSGGeneratedSample(
            subject_id=win_recording.subject_id,
            recording_duration_ms=win_recording.recording_duration_ms,
            label_class=self.label_class,
            task_type=self.task_name,
            **self._window_fields(win_recording),
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            metadata={
                "target_activity": target,
                "is_positive": is_positive,
            },
        )
