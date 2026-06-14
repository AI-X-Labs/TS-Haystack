# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Integration tests: LTAF natural-only generation is reproducible + materializes tasks."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

import scripts.data.build_ltaf_haystack as build_ltaf_haystack


# All 10 tasks are applicable at ctx=900s (per coverage table).
TASKS_AT_900S = {
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
    "anomaly_detection",
    "anomaly_localization",
}


def _write_config(path: Path, samples: dict[str, int], seed: int = 123) -> None:
    path.write_text(
        f"""
global:
  seed: {seed}
  source_hz: 128
  n_jobs: 1
  label_class: rhythms
  output_dir: data/ltafdb/ltaf_haystack/rhythms/tasks

context_lengths_seconds: [900]

samples:
  train: {samples['train']}
  validation: {samples['validation']}
  test: {samples['test']}

tasks:
  existence:           {{enabled: true}}
  localization:        {{enabled: true}}
  counting:            {{enabled: true}}
  ordering:            {{enabled: true}}
  state_query:         {{enabled: true}}
  antecedent:          {{enabled: true}}
  comparison:          {{enabled: true}}
  multi_hop:           {{enabled: true}}
  anomaly_detection:   {{enabled: true}}
  anomaly_localization: {{enabled: true}}
""".lstrip(),
        encoding="utf-8",
    )


def _run_generator(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    output_root: Path,
    extra_args: list[str] | None = None,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_ltaf_haystack.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--overwrite",
            *(extra_args or []),
        ],
    )
    build_ltaf_haystack.main()


def _load_parquet_payload(root: Path) -> dict[str, list[dict]]:
    payload: dict[str, list[dict]] = {}
    for parquet_path in sorted(root.rglob("data.parquet")):
        rel = parquet_path.relative_to(root).as_posix()
        df = pl.read_parquet(parquet_path).drop("signals", strict=False)
        payload[rel] = df.to_dicts()
    return payload


@pytest.mark.integration
def test_ltaf_generation_reproducible_from_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ltaf_repro.yaml"
    output_a = tmp_path / "run_a"
    output_b = tmp_path / "run_b"

    _write_config(config_path, samples={"train": 3, "validation": 2, "test": 2})
    _run_generator(monkeypatch=monkeypatch, config_path=config_path, output_root=output_a)
    _run_generator(monkeypatch=monkeypatch, config_path=config_path, output_root=output_b)

    payload_a = _load_parquet_payload(output_a)
    payload_b = _load_parquet_payload(output_b)

    assert payload_a, "Expected at least one generated parquet shard"
    assert set(payload_a.keys()) == set(payload_b.keys())
    for path in sorted(payload_a.keys()):
        assert payload_a[path] == payload_b[path], f"Reproducibility diff in {path}"


@pytest.mark.integration
def test_ltaf_generation_materializes_enabled_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "ltaf_all_tasks.yaml"
    output_root = tmp_path / "all_tasks"

    _write_config(config_path, samples={"train": 2, "validation": 1, "test": 1}, seed=777)
    _run_generator(
        monkeypatch=monkeypatch,
        config_path=config_path,
        output_root=output_root,
        extra_args=["--max-samples-per-split", "1"],
    )

    context_root = output_root / "900_0s"
    assert context_root.exists(), f"Missing ctx dir: {context_root}"

    observed_tasks = {p.name for p in context_root.iterdir() if p.is_dir()}
    assert observed_tasks == TASKS_AT_900S, (
        f"Task dirs: expected {sorted(TASKS_AT_900S)}, got {sorted(observed_tasks)}"
    )

    # Easy tasks (single-rhythm windows suffice) must produce ≥1 train row.
    easy_tasks = {
        "existence", "counting", "state_query",
        "anomaly_detection", "anomaly_localization", "localization",
    }
    for task_name in sorted(easy_tasks):
        train_path = context_root / task_name / "train" / "data.parquet"
        assert train_path.exists(), f"Missing shard: {train_path}"
        train_rows = pl.read_parquet(train_path).height
        assert train_rows >= 1, (
            f"Expected ≥1 train row for {task_name}, got {train_rows}"
        )
