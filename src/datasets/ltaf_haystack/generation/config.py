# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Generation configuration for LTAF-Haystack (natural-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_generation_config.yaml"


@dataclass
class LTAFTaskConfig:
    enabled: bool = True
    task_specific: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LTAFTaskConfig":
        return cls(
            enabled=bool(raw.get("enabled", True)),
            task_specific={k: v for k, v in raw.items() if k != "enabled"},
        )


@dataclass
class LTAFGenerationConfig:
    seed: int = 42
    source_hz: int = 128
    n_jobs: int = 1
    label_class: str = "rhythms"
    output_dir: Path = field(
        default_factory=lambda: Path("data/ltafdb/ltaf_haystack/rhythms/tasks")
    )
    context_lengths_seconds: List[float] = field(
        default_factory=lambda: [100.0, 900.0, 3600.0, 7200.0]
    )
    samples_per_split: Dict[str, int] = field(
        default_factory=lambda: {"train": 1000, "validation": 150, "test": 150}
    )
    tasks: Dict[str, LTAFTaskConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "LTAFGenerationConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        g = raw.get("global", {})
        tasks = {
            name: LTAFTaskConfig.from_dict(cfg or {})
            for name, cfg in (raw.get("tasks", {}) or {}).items()
        }
        return cls(
            seed=int(g.get("seed", 42)),
            source_hz=int(g.get("source_hz", 128)),
            n_jobs=int(g.get("n_jobs", 1)),
            label_class=str(g.get("label_class", "rhythms")),
            output_dir=Path(
                g.get("output_dir", "data/ltafdb/ltaf_haystack/rhythms/tasks")
            ),
            context_lengths_seconds=[
                float(x)
                for x in raw.get(
                    "context_lengths_seconds", [100.0, 900.0, 3600.0, 7200.0]
                )
            ],
            samples_per_split={
                k: int(v)
                for k, v in (
                    raw.get("samples", {"train": 1000, "validation": 150, "test": 150})
                    or {}
                ).items()
            },
            tasks=tasks,
        )

    @classmethod
    def load_default(cls) -> "LTAFGenerationConfig":
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    def get_context_lengths_samples(self) -> List[int]:
        return [int(s * self.source_hz) for s in self.context_lengths_seconds]

    def get_enabled_tasks(self) -> List[str]:
        return sorted([n for n, c in self.tasks.items() if c.enabled])

    def get_task_config(self, task_name: str) -> LTAFTaskConfig:
        return self.tasks.get(task_name, LTAFTaskConfig())


__all__ = [
    "LTAFGenerationConfig",
    "LTAFTaskConfig",
    "DEFAULT_CONFIG_PATH",
]
