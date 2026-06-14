"""YAML config loader for UK-DALE-Haystack generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("defaults.yaml")


@dataclass
class GlobalConfig:
    seed: int = 42
    n_jobs: int = 1
    output_dir: str = "data/uk_dale/uk_dale_haystack/tasks"
    source_hz: float = 1.0 / 6.0
    dt_s: float = 6.0


@dataclass
class GenerationConfig:
    global_: GlobalConfig
    context_lengths_seconds: list[int]
    samples: dict[str, int]            # {train, validation, test}
    allow_cross_house: bool = False
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "GenerationConfig":
        raw = yaml.safe_load(Path(path).read_text())
        g = raw.get("global", {})
        return cls(
            global_=GlobalConfig(
                seed=int(g.get("seed", 42)),
                n_jobs=int(g.get("n_jobs", 1)),
                output_dir=str(g.get("output_dir", "data/uk_dale/uk_dale_haystack/tasks")),
                source_hz=float(g.get("source_hz", 1.0 / 6.0)),
                dt_s=float(g.get("dt_s", 6.0)),
            ),
            context_lengths_seconds=list(raw.get("context_lengths_seconds", [900])),
            samples=dict(raw.get("samples", {"train": 1000, "validation": 150, "test": 150})),
            allow_cross_house=bool(raw.get("allow_cross_house", False)),
            tasks=dict(raw.get("tasks", {})),
        )

    def enabled_tasks(self) -> list[str]:
        return [name for name, cfg in self.tasks.items() if cfg.get("enabled", True)]
