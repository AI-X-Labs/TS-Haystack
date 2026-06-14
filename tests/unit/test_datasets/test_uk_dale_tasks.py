"""Smoke + invariant tests for UK-DALE-Haystack task generators.

Uses the live bout_index.parquet as a fixture (built by Phase 2.2). Skips if
the parquet is absent (so CI without the dataset still works).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

BOUT_INDEX = Path("data/uk_dale/uk_dale_haystack/bout_index.parquet")
pytestmark = pytest.mark.skipif(
    not BOUT_INDEX.exists(),
    reason="bout_index.parquet missing; run scripts/data/uk_dale/build_uk_dale_bout_index.py first",
)


@pytest.fixture(scope="module")
def samplers():
    from src.datasets.uk_dale_haystack.core.background_sampler import BackgroundSampler
    from src.datasets.uk_dale_haystack.core.needle_sampler import NeedleSampler
    from src.datasets.uk_dale_haystack.core.prompt_templates import PromptTemplateBank

    bouts = pl.read_parquet(BOUT_INDEX)
    return (
        BackgroundSampler(bouts, "train"),
        NeedleSampler(bouts, "train"),
        PromptTemplateBank(),
    )


@pytest.mark.parametrize("ctx_s", [900, 3600])
@pytest.mark.parametrize("task_name", [
    "existence", "localization", "counting", "ordering",
    "antecedent", "comparison", "multi_hop", "state_query",
    "anomaly_detection", "anomaly_localization",
])
def test_task_generates_valid_sample(samplers, ctx_s, task_name):
    from src.datasets.uk_dale_haystack.tasks import TASK_REGISTRY
    bg, ns, tb = samplers
    cls = TASK_REGISTRY[task_name]
    gen = cls(bg, ns, tb)
    rng = np.random.default_rng(hash((task_name, ctx_s)) & 0xFFFFFFFF)
    sample = gen.generate_sample(ctx_s, rng)
    assert sample.is_valid, sample.validation_notes
    assert sample.question
    assert sample.answer
    assert sample.mains_w.shape[0] == sample.context_length_samples
    assert sample.context_length_s == pytest.approx(ctx_s, rel=1e-3)


def test_existence_balance(samplers):
    """Over many generations, answer should be ~50/50 yes/no."""
    from src.datasets.uk_dale_haystack.tasks import ExistenceTaskGenerator
    bg, ns, tb = samplers
    gen = ExistenceTaskGenerator(bg, ns, tb)
    answers = []
    for i in range(60):
        rng = np.random.default_rng(i)
        s = gen.generate_sample(900, rng)
        if s.is_valid:
            answers.append(s.answer)
    yes_count = sum(1 for a in answers if a.lower() == "yes")
    assert 0.30 * len(answers) <= yes_count <= 0.70 * len(answers), (
        f"expected ~50/50 yes/no, got {yes_count}/{len(answers)} yes"
    )


def test_insertion_is_additive(samplers):
    """Verify that inserting a needle at position p increases mains[p:p+len]
    by exactly the needle's submeter trace, and leaves the rest untouched."""
    from src.datasets.uk_dale_haystack.core.insertion import (
        insert_needles, sample_positions,
    )
    bg_sampler, ns, _ = samplers
    rng = np.random.default_rng(1)
    bg = bg_sampler.sample("kettle", 900, rng)
    needle = ns.sample("kettle", max_duration_s=300, rng=rng,
                       require_house_id=bg.house_id)
    assert needle is not None
    positions = sample_positions(
        bg.mains_w.shape[0], [needle.submeter_w.shape[0]],
        margin_samples=10, min_gap_samples=10, rng=rng,
    )
    assert positions is not None
    sample = insert_needles(
        bg, [needle], positions,
        task_type="existence", question="?", answer="yes", answer_type="boolean",
    )
    pos = positions[0]
    n = needle.submeter_w.shape[0]
    delta = sample.mains_w - bg.mains_w
    assert np.all(delta[:pos] == 0)
    assert np.all(delta[pos + n:] == 0)
    np.testing.assert_allclose(delta[pos:pos + n], needle.submeter_w, atol=1e-4)
