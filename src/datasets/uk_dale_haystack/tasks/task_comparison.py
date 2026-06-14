"""
Comparison. Insert >= 2 needles of the same target appliance with distinctly
different durations or peaks; ask which is longest / shortest / highest peak.
"""
from __future__ import annotations

import numpy as np

from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator


class ComparisonTaskGenerator(BaseTaskGenerator):
    task_name = "comparison"
    answer_type = "time_range"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        targets = [a for a in self.bg_sampler.appliances_with_bouts]
        rng.shuffle(targets)

        for target in targets:
            n_needles = int(rng.integers(2, 4))   # 2..3
            mode = ["longest", "shortest", "highest peak"][int(rng.integers(0, 3))]

            bg = self.bg_sampler.sample(
                target=target, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needles = []
            for _ in range(n_needles):
                nd = self.needle_sampler.sample(
                    appliance=target, max_duration_s=max_dur_s / n_needles, rng=rng,
                    require_house_id=(None if self.allow_cross_house else bg.house_id),
                    allow_cross_house=self.allow_cross_house,
                )
                if nd is not None:
                    needles.append(nd)
            if len(needles) < 2:
                continue

            # Avoid ties: require distinct values on the chosen mode
            def key(n):
                return n.duration_s if mode != "highest peak" else n.peak_w
            keys = [key(n) for n in needles]
            if len(set(round(k, 1) for k in keys)) < 2:
                continue

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [n.submeter_w.shape[0] for n in needles],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            if mode == "longest":
                idx = int(np.argmax([n.duration_s for n in needles]))
            elif mode == "shortest":
                idx = int(np.argmin([n.duration_s for n in needles]))
            else:  # highest peak
                idx = int(np.argmax([n.peak_w for n in needles]))

            t0_s = positions[idx] * self.dt_s
            t1_s = (positions[idx] + needles[idx].submeter_w.shape[0]) * self.dt_s

            question, answer = self.template_bank.render(
                "comparison",
                {"appliance": target, "mode": mode, "t0_s": t0_s, "t1_s": t1_s},
                rng,
            )
            return insert_needles(
                bg, needles, positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target, "mode": mode, "n_needles": len(needles),
                    "winner_index": idx,
                },
            )
        return self._invalid(ctx_s, "exhausted targets")
