# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Logging infrastructure for training experiments."""

import atexit
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Unified logging for training experiments.

    Supports:
        - Weights & Biases (wandb) integration
        - Local JSON lines logging
        - Console logging
        - Metric tracking and aggregation
        - Checkpoint artifact logging

    Usage:
        logger = ExperimentLogger(
            experiment_name="my_experiment",
            config=config_dict,
            use_wandb=True,
            log_dir="outputs/logs",
        )

        for step, metrics in enumerate(training_loop):
            logger.log_metrics(metrics, step=step)

        logger.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        config: dict[str, Any] | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
        wandb_group: str | None = None,
        log_dir: str | Path | None = None,
        console_level: int = logging.INFO,
    ):
        """Initialize the experiment logger.

        Args:
            experiment_name: Name of the experiment.
            config: Configuration dictionary to log.
            use_wandb: Whether to use Weights & Biases.
            wandb_project: W&B project name.
            wandb_entity: W&B entity (team/user).
            wandb_tags: Tags for the W&B run.
            wandb_group: W&B run group (groups related runs together).
            log_dir: Directory for local logs. If None, uses 'outputs/{experiment_name}'.
            console_level: Logging level for console output.
        """
        self.experiment_name = experiment_name
        self.config = config or {}
        self.use_wandb = use_wandb
        self._wandb_run = None

        # Set up log directory
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("outputs") / experiment_name / timestamp
        else:
            log_dir = Path(log_dir)

        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging
        self._metrics_file = open(self.log_dir / "metrics.jsonl", "w")
        self._setup_console_logging(console_level)

        # Save config
        if config:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

        # Initialize W&B if requested
        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity, wandb_tags, wandb_group)

        # Track metrics history for aggregation
        self._metric_history: dict[str, list[float]] = {}
        self._step = 0

        # Register cleanup on exit
        atexit.register(self.finish)

        logger.info(f"Experiment '{experiment_name}' logging to {self.log_dir}")

    def _setup_console_logging(self, level: int) -> None:
        """Set up console logging."""
        root_logger = logging.getLogger()

        # Only add handler if not already configured
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            root_logger.setLevel(level)

        # Also log to file
        file_handler = logging.FileHandler(self.log_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        root_logger.addHandler(file_handler)

    def _init_wandb(
        self,
        project: str | None,
        entity: str | None,
        tags: list[str] | None,
        group: str | None = None,
    ) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            # Surface key hyperparameters as top-level W&B config keys
            # so they appear in the runs table without expanding nested groups.
            flat_config = dict(self.config)
            training = self.config.get("training", {})
            model = self.config.get("model", {})
            training = self.config.get("training", {})
            backbone = self.config.get("backbone", {})
            dataset_kw = self.config.get("dataset", {}).get("extra_kwargs", {})

            flat_config["architecture"] = model.get("architecture")
            flat_config["encoder"] = model.get("encoder", {}).get("type")
            flat_config["projector"] = model.get("projector", {}).get("type")
            flat_config["llm_id"] = backbone.get("model_id")
            flat_config["batch_size"] = training.get("batch_size")
            flat_config["epochs"] = training.get("num_epochs")
            flat_config["lr"] = training.get("learning_rate", {}).get("default")
            flat_config["backbone_training"] = training.get("backbone_training")
            flat_config["tasks"] = dataset_kw.get("tasks")
            flat_config["context_lengths"] = dataset_kw.get("context_lengths_seconds")

            self._wandb_run = wandb.init(
                project=project or "ts-haystack",
                entity=entity,
                name=self.experiment_name,
                config=flat_config,
                tags=tags,
                group=group,
                dir=str(self.log_dir),
            )
            logger.info(f"W&B run initialized: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Disabling W&B logging.")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Disabling W&B logging.")
            self.use_wandb = False

    def log_metrics(
        self,
        metrics: dict[str, float | int],
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log metrics for a training step.

        Args:
            metrics: Dictionary of metric names to values.
            step: Step number. If None, uses internal counter.
            prefix: Prefix to add to metric names (e.g., 'train/', 'val/').
        """
        if step is None:
            step = self._step
            self._step += 1

        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Add step and timestamp
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        # Write to JSON lines file
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()

        # Track history for aggregation
        for name, value in metrics.items():
            if name not in self._metric_history:
                self._metric_history[name] = []
            self._metric_history[name].append(float(value))

        # Log to W&B
        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)

    def log_epoch_metrics(
        self,
        metrics: dict[str, float | int],
        epoch: int,
        step: int | None = None,
        prefix: str = "",
    ) -> None:
        """Log metrics at the end of an epoch.

        Args:
            metrics: Dictionary of metric names to values.
            epoch: Epoch number.
            step: Global step number. Must be provided to stay in sync
                  with per-step training logs sent to W&B.
            prefix: Prefix to add to metric names.
        """
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        metrics["epoch"] = epoch
        self.log_metrics(metrics, step=step)
        logger.info(f"Epoch {epoch}: {metrics}")

    def log_artifact(
        self,
        path: str | Path,
        artifact_type: str = "model",
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an artifact (model checkpoint, dataset, etc.).

        Args:
            path: Path to the artifact file or directory.
            artifact_type: Type of artifact ('model', 'dataset', 'config').
            name: Name for the artifact. If None, uses filename.
            metadata: Additional metadata to attach.
        """
        path = Path(path)
        name = name or path.name

        # Log to local registry
        artifact_record = {
            "name": name,
            "type": artifact_type,
            "path": str(path),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        artifacts_file = self.log_dir / "artifacts.jsonl"
        with open(artifacts_file, "a") as f:
            f.write(json.dumps(artifact_record) + "\n")

        # Log to W&B
        if self.use_wandb and self._wandb_run:
            import wandb

            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)

        logger.info(f"Logged artifact: {name} ({artifact_type})")

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary metrics (final results).

        Args:
            summary: Dictionary of summary metrics.
        """
        # Write to summary file
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Log to W&B
        if self.use_wandb and self._wandb_run:
            import wandb
            for key, value in summary.items():
                wandb.run.summary[key] = value

        logger.info(f"Summary: {summary}")

    def get_metric_history(self, metric_name: str) -> list[float]:
        """Get the history of a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            List of metric values.
        """
        return self._metric_history.get(metric_name, [])

    def get_best_metric(
        self,
        metric_name: str,
        mode: str = "max",
    ) -> tuple[float, int] | None:
        """Get the best value of a metric and its step.

        Args:
            metric_name: Name of the metric.
            mode: 'max' or 'min'.

        Returns:
            Tuple of (best_value, step) or None if no history.
        """
        history = self._metric_history.get(metric_name, [])
        if not history:
            return None

        if mode == "max":
            best_idx = max(range(len(history)), key=lambda i: history[i])
        else:
            best_idx = min(range(len(history)), key=lambda i: history[i])

        return history[best_idx], best_idx

    def finish(self) -> None:
        """Finish logging and clean up resources."""
        if hasattr(self, "_metrics_file") and not self._metrics_file.closed:
            self._metrics_file.close()

        if self.use_wandb and self._wandb_run:
            import wandb
            wandb.finish()

        logger.info(f"Experiment finished. Logs saved to {self.log_dir}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
