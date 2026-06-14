# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""I/O utilities for saving and loading models, checkpoints, and data."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    path: str | Path,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state (optional).
        epoch: Current epoch number.
        step: Current global step.
        metrics: Dictionary of current metrics.
        path: Path to save the checkpoint.
        scheduler: Learning rate scheduler (optional).
        extra: Additional data to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into (optional).
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        map_location: Device to map tensors to.

    Returns:
        Checkpoint dictionary with epoch, step, metrics, and extra data.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model state")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Loaded optimizer state")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Loaded scheduler state")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "extra": checkpoint.get("extra", {}),
    }


def save_model(
    model: nn.Module,
    path: str | Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Save model weights and optional config.

    Args:
        model: The model to save.
        path: Directory path to save the model.
        config: Optional configuration dictionary.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pt")

    if config is not None:
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    logger.info(f"Saved model to {path}")


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    """Load model weights from a saved model.

    Args:
        model: Model to load weights into.
        path: Path to the model directory or weights file.
        strict: Whether to strictly enforce state dict matching.
        map_location: Device to map tensors to.

    Returns:
        Model with loaded weights.
    """
    path = Path(path)

    if path.is_dir():
        weights_path = path / "model.pt"
    else:
        weights_path = path

    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=strict)

    logger.info(f"Loaded model weights from {weights_path}")
    return model


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a directory.

    Looks for files matching 'checkpoint_*.pt' or 'checkpoint-*.pt'.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint*.pt"))
    checkpoints.extend(checkpoint_dir.glob("checkpoint-*.pt"))

    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return checkpoints[-1]


def get_best_checkpoint(
    checkpoint_dir: str | Path,
    metric_name: str = "val_accuracy",
    mode: str = "max",
) -> Path | None:
    """Find the best checkpoint based on a metric.

    Looks for a 'best_checkpoint.json' file that tracks the best checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        metric_name: Name of the metric to optimize.
        mode: 'max' or 'min'.

    Returns:
        Path to the best checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_file = checkpoint_dir / "best_checkpoint.json"

    if best_file.exists():
        with open(best_file) as f:
            best_info = json.load(f)
        checkpoint_path = checkpoint_dir / best_info.get("checkpoint", "")
        if checkpoint_path.exists():
            return checkpoint_path

    return None


def save_best_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_name: str,
    metric_name: str,
    metric_value: float,
    epoch: int,
    step: int,
) -> None:
    """Save information about the best checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        checkpoint_name: Name of the checkpoint file.
        metric_name: Name of the metric.
        metric_value: Value of the metric.
        epoch: Epoch number.
        step: Step number.
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_file = checkpoint_dir / "best_checkpoint.json"

    best_info = {
        "checkpoint": checkpoint_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "epoch": epoch,
        "step": step,
    }

    with open(best_file, "w") as f:
        json.dump(best_info, f, indent=2)
