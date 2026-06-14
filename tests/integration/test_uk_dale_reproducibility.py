"""Reproducibility: two generation passes with the same seed should produce
identical parquet hashes."""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

BOUT_INDEX = Path("data/uk_dale/uk_dale_haystack/bout_index.parquet")
pytestmark = pytest.mark.skipif(
    not BOUT_INDEX.exists(),
    reason="bout_index.parquet missing; run scripts/data/uk_dale/build_uk_dale_bout_index.py first",
)


def _run_generation(out_dir: Path) -> dict[str, str]:
    """Run a tiny generation pass into out_dir; return {shard_path: sha256}."""
    from src.datasets.uk_dale_haystack.generation.config import (
        GenerationConfig, GlobalConfig,
    )
    from src.datasets.uk_dale_haystack.generation.generator import run

    cfg = GenerationConfig(
        global_=GlobalConfig(seed=42, output_dir=str(out_dir), dt_s=6.0),
        context_lengths_seconds=[900],
        samples={"train": 20, "validation": 10, "test": 10},
        allow_cross_house=False,
        tasks={"existence": {"enabled": True},
               "counting":  {"enabled": True}},
    )
    run(cfg, bout_index_path=BOUT_INDEX, overwrite=True)

    hashes: dict[str, str] = {}
    for parquet in sorted(out_dir.rglob("data.parquet")):
        df = pl.read_parquet(parquet).sort(["task_type", "split"])
        # Hash the *content* (question + answer + reconstruction metadata),
        # which is what reproducibility means here.
        digest = hashlib.sha256()
        for row in df.iter_rows():
            digest.update("|".join(str(v) for v in row).encode())
        hashes[parquet.relative_to(out_dir).as_posix()] = digest.hexdigest()
    return hashes


def test_two_passes_match():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        h1 = _run_generation(Path(d1))
        h2 = _run_generation(Path(d2))
        assert sorted(h1.keys()) == sorted(h2.keys()), "different shards produced"
        for k in h1:
            assert h1[k] == h2[k], f"hash mismatch for {k}"
