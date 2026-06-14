# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Configuration loading and validation utilities."""

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dictionary to YAML.

    Args:
        config: Configuration dictionary to save.
        path: Output path for the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Values in `override` take precedence over values in `base`.
    Nested dictionaries are merged recursively.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config_with_defaults(
    config_path: str | Path,
    defaults_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a configuration file and merge with defaults.

    Looks for default configs in the defaults directory and merges them
    with the experiment config. Default configs are loaded based on keys
    in the experiment config (e.g., 'model' -> 'model_defaults.yaml').

    Args:
        config_path: Path to the experiment configuration file.
        defaults_dir: Directory containing default config files.
            If None, looks for 'configs/defaults' relative to config_path.

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_path)
    config = load_yaml(config_path)

    if defaults_dir is None:
        defaults_dir = config_path.parent.parent / "defaults"

    defaults_dir = Path(defaults_dir)

    if not defaults_dir.exists():
        return config

    # Load and merge defaults for each top-level key
    for key in ["model", "training", "dataset"]:
        default_file = defaults_dir / f"{key}_defaults.yaml"
        if default_file.exists() and key in config:
            defaults = load_yaml(default_file)
            config[key] = merge_configs(defaults, config[key])

    return config


def resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve environment variables in config values.

    Replaces ${VAR_NAME} patterns with environment variable values.

    Args:
        config: Configuration dictionary.

    Returns:
        Configuration with environment variables resolved.
    """
    def resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            # Replace ${VAR_NAME} patterns
            import re
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                value = value.replace(f"${{{var_name}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]
        return value

    return resolve_value(config)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Dataset-specific parameters (e.g. ``tasks``, ``context_lengths_seconds``
    for TS-Haystack) belong in ``extra_kwargs`` and are forwarded to the
    dataset constructor by ``create_dataloaders()``.
    """

    name: str = "capture24_haystack_cot"
    data_dir: str = "data"
    split: str = "train"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    architecture: str = "flamingo"
    encoder: dict[str, Any] = field(default_factory=dict)
    projector: dict[str, Any] = field(default_factory=dict)
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BackboneConfig:
    """Configuration for LLM backbone."""

    name: str = "llama"
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_checkpoint_repo: str | None = None  # Optional pretrained checkpoint
    hf_checkpoint_file: str = "model_checkpoint.pt"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BackboneConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 2
    num_epochs: int = 30
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    backbone_training: str = "freeze"  # "freeze" | "lora" | "full"
    lora: dict[str, Any] = field(default_factory=lambda: {
        "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": None,
    })
    learning_rate: dict[str, float] = field(default_factory=lambda: {"default": 2e-4})
    early_stopping: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "patience": 5, "metric": "val_loss", "mode": "min"
    })
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    checkpointing: dict[str, Any] = field(default_factory=dict)
    dataloader: dict[str, Any] = field(default_factory=dict)
    eval_max_new_tokens: int = 500

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RuntimeConfig:
    """Operational settings that are not experiment hyperparameters."""

    output_dir: Path = field(default_factory=lambda: Path("results"))
    run_name: str | None = None
    resume_from: str | None = None
    resume_weights_only: bool = False  # Load only model weights, not optimizer/scheduler/epoch
    max_samples: int | None = None
    eval_samples_ratio: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        if "output_dir" in filtered:
            filtered["output_dir"] = Path(filtered["output_dir"])
        return cls(**filtered)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "experiment"
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # -- convenience properties ------------------------------------------------

    @property
    def early_stop_patience(self) -> int:
        return self.training.early_stopping.get("patience", 5)

    @property
    def checkpoint_interval(self) -> int:
        return self.training.checkpointing.get("save_every_n_steps", 3000)

    @property
    def use_wandb(self) -> bool:
        wandb = self.training.logging.get("wandb", {})
        if isinstance(wandb, dict):
            return wandb.get("enabled", True)
        return True

    @property
    def wandb_project(self) -> str:
        wandb = self.training.logging.get("wandb", {})
        if isinstance(wandb, dict):
            return wandb.get("project", "ts-haystack")
        return "ts-haystack"

    @property
    def wandb_entity(self) -> str | None:
        wandb = self.training.logging.get("wandb", {})
        if isinstance(wandb, dict):
            return wandb.get("entity")
        return None

    @property
    def wandb_group(self) -> str | None:
        wandb = self.training.logging.get("wandb", {})
        if isinstance(wandb, dict):
            return wandb.get("group")
        return None

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full config to a plain dict (Path -> str)."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        return _convert(asdict(self))

    # -- construction helpers --------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "experiment"),
            seed=d.get("seed", 42),
            dataset=DatasetConfig.from_dict(d.get("dataset", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            backbone=BackboneConfig.from_dict(d.get("backbone", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            runtime=RuntimeConfig.from_dict(d.get("runtime", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment config from YAML file with defaults."""
        config_dict = load_config_with_defaults(path)
        config_dict = resolve_env_vars(config_dict)
        return cls.from_dict(config_dict)


def compute_patch_sizes(
    max_context_seconds: float,
    sample_rate: int = 100,
    patch_size: int = 4,
    buffer: int = 1000,
) -> tuple[int, int]:
    """Compute trained_patches and max_patches based on context length.

    This follows the formula from train_ts_haystack_cot.py:
        trained_patches = (max_context_seconds * sample_rate) // patch_size
        max_patches = trained_patches + buffer

    Args:
        max_context_seconds: Maximum context length in seconds.
        sample_rate: Sampling rate in Hz (default: 100 for Capture24).
        patch_size: Samples per patch (default: 4 from PATCH_SIZE).
        buffer: Extra patches for slightly longer sequences (default: 1000).

    Returns:
        Tuple of (trained_patches, max_patches).

    Example:
        >>> compute_patch_sizes(7200)  # 2 hours
        (180000, 181000)
        >>> compute_patch_sizes(100)   # 100 seconds
        (2500, 3500)
    """
    trained_patches = int((max_context_seconds * sample_rate) // patch_size)
    max_patches = trained_patches + buffer
    return trained_patches, max_patches
