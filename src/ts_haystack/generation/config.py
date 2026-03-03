# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Configuration classes for TS-Haystack dataset generation.

Loads YAML configuration and builds DifficultyConfig objects for task generators.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ts_haystack.core.data_structures import DifficultyConfig


# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_generation_config.yaml"


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer during needle insertion."""

    transfer_mode: str = "mean_only"  # "mean_only" or "full"
    blend_mode: str = "cosine"  # "cosine" or "linear"
    blend_window_samples: int = 50


@dataclass
class TaskDifficultyConfig:
    """
    Difficulty configuration for a single task from YAML.

    Standard fields are parsed separately from task_specific parameters.
    """

    enabled: bool = True
    needle_position: str = "random"
    needle_length_ratio_range: Tuple[float, float] = (0.02, 0.10)
    background_purity: str = "pure"
    task_specific: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskDifficultyConfig":
        """
        Create from YAML task dictionary.

        Standard fields are extracted; everything else goes to task_specific.
        """
        standard_fields = {
            "enabled",
            "needle_position",
            "needle_length_ratio_range",
            "background_purity",
        }
        task_specific = {k: v for k, v in d.items() if k not in standard_fields}

        ratio_range = d.get("needle_length_ratio_range", [0.02, 0.10])
        if isinstance(ratio_range, list):
            ratio_range = tuple(ratio_range)

        return cls(
            enabled=d.get("enabled", True),
            needle_position=d.get("needle_position", "random"),
            needle_length_ratio_range=ratio_range,
            background_purity=d.get("background_purity", "pure"),
            task_specific=task_specific,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "needle_position": self.needle_position,
            "needle_length_ratio_range": list(self.needle_length_ratio_range),
            "background_purity": self.background_purity,
            **self.task_specific,
        }


@dataclass
class GenerationConfig:
    """
    Complete configuration for TS-Haystack dataset generation.

    Loaded from YAML configuration file. Provides methods to:
    - Get list of enabled tasks
    - Convert context lengths from seconds to samples
    - Build DifficultyConfig for specific task/context combinations
    """

    # Global settings
    seed: int = 42
    n_jobs: int = 4
    output_dir: Path = field(
        default_factory=lambda: Path("data/capture24/ts_haystack/tasks")
    )
    overwrite: bool = False
    source_hz: int = 100

    # What to generate
    context_lengths_seconds: List[float] = field(default_factory=lambda: [100])
    samples_per_split: Dict[str, int] = field(
        default_factory=lambda: {"train": 10000, "val": 1000, "test": 1000}
    )

    # Style transfer
    style_transfer: StyleTransferConfig = field(default_factory=StyleTransferConfig)

    # Per-task configs
    tasks: Dict[str, TaskDifficultyConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Populated GenerationConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        global_cfg = data.get("global", {})
        samples_cfg = data.get("samples", {"train": 10000, "val": 1000, "test": 1000})
        style_cfg = data.get("style_transfer", {})
        tasks_cfg = data.get("tasks", {})

        # Parse tasks
        parsed_tasks = {}
        for task_name, task_dict in tasks_cfg.items():
            parsed_tasks[task_name] = TaskDifficultyConfig.from_dict(task_dict)

        return cls(
            seed=global_cfg.get("seed", 42),
            n_jobs=global_cfg.get("n_jobs", 4),
            output_dir=Path(
                global_cfg.get("output_dir", "data/capture24/ts_haystack/tasks")
            ),
            overwrite=global_cfg.get("overwrite", False),
            source_hz=global_cfg.get("source_hz", 100),
            context_lengths_seconds=data.get("context_lengths_seconds", [100]),
            samples_per_split=samples_cfg,
            style_transfer=StyleTransferConfig(
                transfer_mode=style_cfg.get("transfer_mode", "mean_only"),
                blend_mode=style_cfg.get("blend_mode", "cosine"),
                blend_window_samples=style_cfg.get("blend_window_samples", 50),
            ),
            tasks=parsed_tasks,
        )

    @classmethod
    def load_default(cls) -> "GenerationConfig":
        """Load default configuration from bundled YAML file."""
        if not DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Default config not found at {DEFAULT_CONFIG_PATH}. "
                "Please specify a config file with --config."
            )
        return cls.from_yaml(DEFAULT_CONFIG_PATH)

    def get_enabled_tasks(self) -> List[str]:
        """Get list of enabled task names."""
        return [name for name, cfg in self.tasks.items() if cfg.enabled]

    def get_context_lengths_samples(self) -> List[int]:
        """Convert context lengths from seconds to samples."""
        return [int(s * self.source_hz) for s in self.context_lengths_seconds]

    def get_difficulty_config(
        self,
        task_name: str,
        context_length_samples: int,
    ) -> DifficultyConfig:
        """
        Build DifficultyConfig for a specific task and context length.

        Args:
            task_name: Name of the task
            context_length_samples: Context length in samples

        Returns:
            DifficultyConfig ready for task generator
        """
        task_cfg = self.tasks.get(task_name)
        if task_cfg is None:
            # Return defaults if task not in config
            return DifficultyConfig(
                context_length_samples=context_length_samples,
            )

        return DifficultyConfig(
            context_length_samples=context_length_samples,
            needle_position=task_cfg.needle_position,
            needle_length_ratio_range=task_cfg.needle_length_ratio_range,
            background_purity=task_cfg.background_purity,
            task_specific=task_cfg.task_specific,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata serialization."""
        return {
            "seed": self.seed,
            "n_jobs": self.n_jobs,
            "output_dir": str(self.output_dir),
            "overwrite": self.overwrite,
            "source_hz": self.source_hz,
            "context_lengths_seconds": self.context_lengths_seconds,
            "samples_per_split": self.samples_per_split,
            "style_transfer": {
                "transfer_mode": self.style_transfer.transfer_mode,
                "blend_mode": self.style_transfer.blend_mode,
                "blend_window_samples": self.style_transfer.blend_window_samples,
            },
            "tasks": {name: cfg.to_dict() for name, cfg in self.tasks.items()},
        }


def print_default_config() -> str:
    """
    Return default configuration as YAML string.

    Used by CLI to print default config for user customization.
    """
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH.read_text()
    else:
        return "# Default config file not found at expected location"
