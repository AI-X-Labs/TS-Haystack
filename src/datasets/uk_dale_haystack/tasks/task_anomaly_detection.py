"""
Anomaly detection. Boolean: "Is there an anomalous appliance bout in this window?".
Half positive (anomaly inserted) + half negative (only nominal needle inserted).
Anomaly classes:
  - truncated_cycle: long_cycle/cooking only; trim to t in [0.20, 0.50] of duration
  - abnormal_peak:   impulse/cooking only;   scale by s in [1.5, 2.5] (cap at MAX_POWER_W)
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.datasets.uk_dale_haystack.core.activity_regimes import (
    ACTIVITY_TO_REGIME,
    MAX_POWER_W,
)
from src.datasets.uk_dale_haystack.core.data_structures import NeedleSample
from src.datasets.uk_dale_haystack.core.insertion import (
    insert_needles,
    sample_positions,
)
from src.datasets.uk_dale_haystack.tasks.base_task import BaseTaskGenerator

UK_RING_MAIN_CEILING_W = 3500.0
TRUNCATED_FRACTION_RANGE = (0.20, 0.50)
ABNORMAL_PEAK_SCALE_RANGE = (1.5, 2.5)


def _clone_with_signal(needle: NeedleSample, new_w: np.ndarray,
                       anomaly_class: str, params: dict) -> NeedleSample:
    n = deepcopy(needle)
    n.submeter_w = new_w.astype("float32", copy=False)
    n.peak_w = float(new_w.max()) if new_w.size else 0.0
    n.mean_w = float(new_w.mean()) if new_w.size else 0.0
    n.duration_s = float(new_w.size * needle.dt_s)
    n.is_anomalous = True
    n.anomaly_class = anomaly_class
    n.anomaly_params = dict(params)
    n.end_ns = int(needle.start_ns + new_w.size * needle.dt_s * 1e9)
    return n


def _anomaly_classes_for(appliance: str) -> list[str]:
    regime = ACTIVITY_TO_REGIME.get(appliance, "")
    out = []
    if regime in ("long_cycle", "cooking"):
        out.append("truncated_cycle")
    if regime in ("impulse", "cooking"):
        out.append("abnormal_peak")
    return out


def synthesize_truncated_cycle(needle: NeedleSample, rng: np.random.Generator) -> NeedleSample | None:
    if needle.submeter_w.size < 5:
        return None
    frac = float(rng.uniform(*TRUNCATED_FRACTION_RANGE))
    n_keep = max(2, int(needle.submeter_w.size * frac))
    new_w = needle.submeter_w[:n_keep]
    # Reject if we lost the canonical heater/motor burst (heuristic: kept peak
    # should be >= 30% of full peak).
    if float(new_w.max()) < 0.30 * float(needle.submeter_w.max()):
        return None
    return _clone_with_signal(
        needle, new_w, "truncated_cycle",
        {"fraction_kept": frac, "n_kept": n_keep, "n_original": needle.submeter_w.size},
    )


def synthesize_abnormal_peak(needle: NeedleSample, rng: np.random.Generator) -> NeedleSample | None:
    if needle.submeter_w.size < 2:
        return None
    scale = float(rng.uniform(*ABNORMAL_PEAK_SCALE_RANGE))
    ceiling = min(UK_RING_MAIN_CEILING_W, MAX_POWER_W.get(needle.appliance, UK_RING_MAIN_CEILING_W))
    scaled = np.minimum(needle.submeter_w * scale, ceiling)
    if float(scaled.max()) > UK_RING_MAIN_CEILING_W:
        return None
    return _clone_with_signal(
        needle, scaled, "abnormal_peak",
        {"scale": scale, "ceiling_w": ceiling},
    )


def make_anomaly(needle: NeedleSample, rng: np.random.Generator) -> NeedleSample | None:
    classes = _anomaly_classes_for(needle.appliance)
    if not classes:
        return None
    rng.shuffle(classes)
    for cls in classes:
        if cls == "truncated_cycle":
            out = synthesize_truncated_cycle(needle, rng)
        elif cls == "abnormal_peak":
            out = synthesize_abnormal_peak(needle, rng)
        else:
            out = None
        if out is not None:
            return out
    return None


class AnomalyDetectionTaskGenerator(BaseTaskGenerator):
    task_name = "anomaly_detection"
    answer_type = "boolean"

    def generate_sample(self, ctx_s, rng):
        params = self._ctx_params(ctx_s)
        max_dur_s = params["max_needle_duration_s"]
        margin_samples = params["margin_samples"]
        min_gap_samples = params["min_gap_samples"]

        has_anomaly = bool(rng.integers(0, 2))

        targets = [a for a in self.bg_sampler.appliances_with_bouts
                   if _anomaly_classes_for(a)]
        rng.shuffle(targets)

        for target in targets:
            bg = self.bg_sampler.sample(
                target=target, context_length_s=ctx_s, rng=rng,
                margin_s=margin_samples * self.dt_s,
            )
            if bg is None:
                continue

            needle = self.needle_sampler.sample(
                appliance=target, max_duration_s=max_dur_s, rng=rng,
                require_house_id=(None if self.allow_cross_house else bg.house_id),
                allow_cross_house=self.allow_cross_house,
            )
            if needle is None:
                continue

            if has_anomaly:
                anom = make_anomaly(needle, rng)
                if anom is None:
                    continue
                needle = anom

            ctx_samples = bg.mains_w.shape[0]
            positions = sample_positions(
                ctx_samples, [needle.submeter_w.shape[0]],
                margin_samples=margin_samples,
                min_gap_samples=min_gap_samples, rng=rng,
            )
            if positions is None:
                continue

            question, answer = self.template_bank.render(
                "anomaly_detection", {"has_anomaly": has_anomaly}, rng,
            )

            return insert_needles(
                bg, [needle], positions,
                task_type=self.task_name,
                question=question, answer=answer,
                answer_type=self.answer_type,
                difficulty_config={
                    "target": target, "has_anomaly": has_anomaly,
                    "anomaly_class": needle.anomaly_class,
                    "anomaly_params": needle.anomaly_params,
                },
            )
        return self._invalid(ctx_s, "no anomaly-eligible targets fit")
