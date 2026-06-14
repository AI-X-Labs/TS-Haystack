"""
Counting. Insert N target needles (N in {0..5}). Answer is the integer count.
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.activity_regimes import (
    same_regime_activities,
)
from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class CountingTaskGenerator(BaseTaskGenerator):
    task_name = "counting"
    answer_type = "integer"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        targets = list(self.bg_sampler.appliances_with_bouts)
        rng.shuffle(targets)

        for target in targets:
            n_target = int(rng.integers(0, 6))  # 0..5 inclusive

            bg = self.bg_sampler.sample(
                target=target, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needles = []
            for _ in range(n_target):
                nd = self.needle_sampler.sample(
                    appliance=target, max_duration_s=max_dur_s, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles.append(nd)

            # When n_target==0 we still need at least one needle in the window so
            # the model can't shortcut on flatness. Insert one same-regime peer.
            if n_target == 0:
                peers = [a for a in same_regime_activities(target)
                         if a in self.bg_sampler.appliances_with_bouts]
                if peers:
                    dapp = peers[int(rng.integers(0, len(peers)))]
                    nd = self.needle_sampler.sample(
                        appliance=dapp, max_duration_s=max_dur_s, rng=rng,
                        require_house_id=(None if self.allow_cross_house else bg.house_id),
                        allow_cross_house=self.allow_cross_house,
                    )
                    if nd is not None:
                        needles.append(nd)

            if not needles and n_target > 0:
                continue

            if needles:
                ctx_samples = bg.mains_w.shape[0]
                positions = sample_positions(
                    ctx_samples, [n.submeter_w.shape[0] for n in needles],
                    margin_samples=margin_samples,
                    min_gap_samples=min_gap_samples, rng=rng,
                )
                if positions is None:
                    continue
            else:
                positions = []

            actual_n = sum(1 for n in needles if n.appliance == target)

            question, answer = self.template_bank.render(
                "counting", {"appliance": target, "n": actual_n}, rng,
            )

            return insert_needles(
                bg, needles, positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target, "n_target_needles": actual_n,
                    "n_distractors": len(needles) - actual_n,
                },
            )
        return self._invalid(ctx_s, "exhausted targets")
