"""
Base class for UK-DALE-Haystack task generators.

Each subclass implements ``generate_sample(ctx_s, rng) -> GeneratedSample | None``.
Margin / gap defaults follow the plan: margin_s = min(0.05*ctx, 60),
min_gap_s = max(60, ctx/20) capped at 600.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
from src.datasets.uk_dale_haystack.core.data_structures import GeneratedSample
from src.datasets.uk_dale_haystack.core.needle_sampler import NeedleSampler
from src.datasets.uk_dale_haystack.core.prompt_templates import PromptTemplateBank
from src.datasets.uk_dale_haystack.loader import NOMINAL_DT_S


def margin_s_for(ctx_s: float) -> float:
    return min(0.05 * ctx_s, 60.0)


def min_gap_s_for(ctx_s: float) -> float:
    return min(600.0, max(60.0, ctx_s / 20.0))


def s_to_samples(s: float, dt_s: float = NOMINAL_DT_S) -> int:
    return max(0, int(round(s / dt_s)))


class BaseTaskGenerator(ABC):
    task_name: str = "base"
    answer_type: str = "category"

    def __init__(
        self,
        background_sampler: BackgroundSampler,
        needle_sampler: NeedleSampler,
        template_bank: PromptTemplateBank,
        *,
        dt_s: float = NOMINAL_DT_S,
        allow_cross_house: bool = False,
    ):
        self.bg_sampler = background_sampler
        self.needle_sampler = needle_sampler
        self.template_bank = template_bank
        self.dt_s = float(dt_s)
        self.allow_cross_house = bool(allow_cross_house)

    @abstractmethod
    def generate_sample(
        self,
        ctx_s: float,
        rng: np.random.Generator,
    ) -> GeneratedSample | None:
        ...

    # -------- shared helpers ----------------------------------------------

    def _invalid(self, ctx_s: float, reason: str, **extra: Any) -> GeneratedSample:
        ctx_samples = s_to_samples(ctx_s, self.dt_s)
        return GeneratedSample(
            task_type=self.task_name,
            question="",
            answer="",
            answer_type=self.answer_type,
            mains_w=np.zeros(ctx_samples, dtype="float32"),
            context_length_samples=ctx_samples,
            context_length_s=float(ctx_s),
            dt_s=self.dt_s,
            background_house_id=-1,
            background_start_ns=-1,
            background_end_ns=-1,
            is_valid=False,
            validation_notes=reason,
            difficulty_config=dict(extra),
        )

    def _ctx_params(self, ctx_s: float) -> dict[str, int]:
        margin_s = margin_s_for(ctx_s)
        gap_s = min_gap_s_for(ctx_s)
        return {
            "margin_samples": s_to_samples(margin_s, self.dt_s),
            "min_gap_samples": s_to_samples(gap_s, self.dt_s),
            "max_needle_duration_s": float(ctx_s - 2 * margin_s),
        }
