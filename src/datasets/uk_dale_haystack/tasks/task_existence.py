"""
Existence task. Always inserts >= 1 needle so the model can't shortcut on "any
deviation from baseline". 50/50 yes/no balance: positive inserts the target;
negative inserts a same-regime distractor and asks about the absent target.
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.activity_regimes import (
    same_regime_activities,
    V1_VOCAB,
)
from src.datasets.uk_dale_haystack.core.data_structures import GeneratedSample
from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class ExistenceTaskGenerator(BaseTaskGenerator):
    task_name = "existence"
    answer_type = "boolean"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        is_positive = bool(rng.integers(0, 2))

        candidates = [a for a in self.bg_sampler.appliances_with_bouts
                      if same_regime_activities(a)]  # need at least one regime peer
        if not candidates:
            return self._invalid(ctx_s, "no candidate appliances")
        rng.shuffle(candidates)

        for target in candidates:
            peers = [a for a in same_regime_activities(target)
                     if a in self.bg_sampler.appliances_with_bouts]
            if not peers:
                continue

            distractor = peers[int(rng.integers(0, len(peers)))]

            needle_appliance = target if is_positive else distractor
            asked_about = target

            bg = self.bg_sampler.sample(
                target=asked_about, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needle_house = (None if self.allow_cross_house else bg.house_id)
            needle = self.needle_sampler.sample(
                appliance=needle_appliance,
                max_duration_s=max_dur_s,
                rng=rng,
                require_house_id=needle_house,
                allow_cross_house=self.allow_cross_house,
            )
            if needle is None:
                continue

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [needle.submeter_w.shape[0]],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples,
                rng=rng,
            )
            if positions is None:
                continue

            params_render = {"appliance": asked_about, "exists": is_positive}
            question, answer = self.template_bank.render(
                "existence", params_render, rng,
            )
            return insert_needles(
                bg, [needle], positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": asked_about,
                    "is_positive": is_positive,
                    "needle_appliance": needle_appliance,
                    "distractor_regime_peer": distractor,
                },
            )
        return self._invalid(ctx_s, "exhausted candidates")
