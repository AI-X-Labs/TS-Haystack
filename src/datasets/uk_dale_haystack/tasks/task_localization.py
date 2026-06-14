"""
Localization. Insert N target needles (1..3); ask for the K-th by start time.
Distractors of the same regime are also inserted to prevent power-band
shortcuts.
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


class LocalizationTaskGenerator(BaseTaskGenerator):
    task_name = "localization"
    answer_type = "time_range"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        targets = [a for a in self.bg_sampler.appliances_with_bouts]
        rng.shuffle(targets)

        for target in targets:
            n_target = int(rng.integers(1, 4))   # 1..3 target bouts
            n_distractors = int(rng.integers(0, 2))   # 0..1 distractors

            bg = self.bg_sampler.sample(
                target=target, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needles = []
            # target needles
            for _ in range(n_target):
                nd = self.needle_sampler.sample(
                    appliance=target, max_duration_s=max_dur_s, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles.append(nd)

            if not needles:
                continue

            # distractors from same regime
            peers = [a for a in same_regime_activities(target)
                     if a in self.bg_sampler.appliances_with_bouts]
            for _ in range(n_distractors):
                if not peers:
                    break
                dapp = peers[int(rng.integers(0, len(peers)))]
                nd = self.needle_sampler.sample(
                    appliance=dapp, max_duration_s=max_dur_s, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles.append(nd)

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [n.submeter_w.shape[0] for n in needles],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            # Pick which target ordinal to ask about (count only target needles)
            target_indices = [i for i, n in enumerate(needles) if n.appliance == target]
            target_indices.sort(key=lambda i: positions[i])
            k = int(rng.integers(0, len(target_indices))) + 1
            chosen_idx = target_indices[k - 1]

            t0_s = positions[chosen_idx] * self.dt_s
            t1_s = (positions[chosen_idx] + needles[chosen_idx].submeter_w.shape[0]) * self.dt_s

            question, answer = self.template_bank.render(
                "localization",
                {"appliance": target, "k": k, "t0_s": t0_s, "t1_s": t1_s},
                rng,
            )

            return insert_needles(
                bg, needles, positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target, "k": k,
                    "n_target_needles": len(target_indices),
                    "n_distractors": len(needles) - len(target_indices),
                },
            )
        return self._invalid(ctx_s, "exhausted targets")
