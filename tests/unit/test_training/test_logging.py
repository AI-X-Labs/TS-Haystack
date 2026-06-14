# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for experiment logging."""

import json

import pytest

from src.training.logging import ExperimentLogger


class TestExperimentLogger:
    def test_init_creates_log_dir(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )
        assert logger.log_dir.exists()
        logger.finish()

    def test_log_metrics_writes_to_file(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )

        logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=0)
        logger.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=1)
        logger.finish()

        metrics_file = logger.log_dir / "metrics.jsonl"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["loss"] == 0.5
        assert record["step"] == 0

    def test_log_metrics_with_prefix(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )

        logger.log_metrics({"loss": 0.5}, step=0, prefix="train/")
        logger.finish()

        metrics_file = logger.log_dir / "metrics.jsonl"
        with open(metrics_file) as f:
            record = json.loads(f.readline())

        assert "train/loss" in record
        assert record["train/loss"] == 0.5

    def test_saves_config(self, temp_dir, sample_config):
        logger = ExperimentLogger(
            experiment_name="test",
            config=sample_config,
            log_dir=temp_dir / "logs",
        )
        logger.finish()

        config_file = logger.log_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["name"] == "test_experiment"

    def test_get_metric_history(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )

        logger.log_metrics({"loss": 0.5}, step=0)
        logger.log_metrics({"loss": 0.3}, step=1)
        logger.log_metrics({"loss": 0.1}, step=2)

        history = logger.get_metric_history("loss")
        assert history == [0.5, 0.3, 0.1]
        logger.finish()

    def test_get_best_metric(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )

        logger.log_metrics({"accuracy": 0.5}, step=0)
        logger.log_metrics({"accuracy": 0.9}, step=1)
        logger.log_metrics({"accuracy": 0.7}, step=2)

        best_val, best_step = logger.get_best_metric("accuracy", mode="max")
        assert best_val == 0.9
        assert best_step == 1
        logger.finish()

    def test_log_summary(self, temp_dir):
        logger = ExperimentLogger(
            experiment_name="test",
            log_dir=temp_dir / "logs",
        )

        logger.log_summary({"final_accuracy": 0.95, "total_steps": 1000})
        logger.finish()

        summary_file = logger.log_dir / "summary.json"
        assert summary_file.exists()

        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["final_accuracy"] == 0.95
