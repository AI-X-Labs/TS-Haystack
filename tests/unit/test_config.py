# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Tests for configuration utilities."""

from pathlib import Path

import pytest

from src.utils.config import (
    DatasetConfig,
    ExperimentConfig,
    RuntimeConfig,
    merge_configs,
    resolve_env_vars,
    save_yaml,
    load_yaml,
    compute_patch_sizes,
)


class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 10, "z": 20}}
        result = merge_configs(base, override)
        assert result == {"a": {"x": 1, "y": 10, "z": 20}, "b": 3}

    def test_override_nested_with_scalar(self):
        base = {"a": {"x": 1}}
        override = {"a": 5}
        result = merge_configs(base, override)
        assert result == {"a": 5}


class TestResolveEnvVars:
    def test_resolve_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "test_value")
        config = {"path": "${TEST_VAR}/data"}
        result = resolve_env_vars(config)
        assert result["path"] == "test_value/data"

    def test_missing_env_var(self):
        config = {"path": "${NONEXISTENT_VAR}/data"}
        result = resolve_env_vars(config)
        assert result["path"] == "/data"

    def test_nested_env_vars(self, monkeypatch):
        monkeypatch.setenv("BASE", "/home")
        config = {"nested": {"path": "${BASE}/user"}}
        result = resolve_env_vars(config)
        assert result["nested"]["path"] == "/home/user"


class TestYamlIO:
    def test_save_and_load(self, temp_dir):
        config = {"name": "test", "value": 42, "nested": {"a": 1}}
        path = temp_dir / "config.yaml"

        save_yaml(config, path)
        loaded = load_yaml(path)

        assert loaded == config

    def test_load_nonexistent(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            load_yaml(temp_dir / "nonexistent.yaml")


class TestDatasetConfig:
    def test_from_dict(self):
        d = {
            "name": "ts_haystack",
            "extra_kwargs": {
                "tasks": ["existence", "localization"],
                "context_lengths_seconds": [100, 500],
            },
        }
        config = DatasetConfig.from_dict(d)
        assert config.name == "ts_haystack"
        assert config.extra_kwargs["tasks"] == ["existence", "localization"]
        assert config.extra_kwargs["context_lengths_seconds"] == [100, 500]

    def test_defaults(self):
        config = DatasetConfig.from_dict({"name": "test"})
        assert config.split == "train"
        assert config.extra_kwargs == {}

    def test_extra_kwargs_forwarded(self):
        config = DatasetConfig.from_dict({
            "name": "test",
            "extra_kwargs": {"tasks": "all", "custom_param": 42},
        })
        assert config.extra_kwargs["tasks"] == "all"
        assert config.extra_kwargs["custom_param"] == 42

    def test_unknown_top_level_keys_ignored(self):
        config = DatasetConfig.from_dict({
            "name": "test",
            "unknown_field": "should be dropped",
        })
        assert config.name == "test"
        assert config.extra_kwargs == {}


class TestComputePatchSizes:
    def test_short_context(self):
        trained, max_p = compute_patch_sizes(100)  # 100 seconds
        assert trained == 2500  # 100 * 100 / 4
        assert max_p == 3500  # 2500 + 1000

    def test_long_context(self):
        trained, max_p = compute_patch_sizes(7200)  # 2 hours
        assert trained == 180000  # 7200 * 100 / 4
        assert max_p == 181000

    def test_custom_params(self):
        trained, max_p = compute_patch_sizes(
            max_context_seconds=10,
            sample_rate=50,
            patch_size=2,
            buffer=100,
        )
        assert trained == 250  # 10 * 50 / 2
        assert max_p == 350


class TestRuntimeConfig:
    def test_defaults(self):
        rc = RuntimeConfig()
        assert rc.output_dir == Path("results")
        assert rc.run_name is None
        assert rc.resume_from is None
        assert rc.max_samples is None

    def test_from_dict(self):
        rc = RuntimeConfig.from_dict({
            "output_dir": "/tmp/out",
            "run_name": "test",
            "max_samples": 50,
        })
        assert rc.output_dir == Path("/tmp/out")
        assert rc.run_name == "test"
        assert rc.max_samples == 50

    def test_from_dict_ignores_unknown_keys(self):
        rc = RuntimeConfig.from_dict({"output_dir": "x", "unknown_key": 123})
        assert rc.output_dir == Path("x")


class TestExperimentConfig:
    def test_from_dict(self, sample_config):
        config = ExperimentConfig.from_dict(sample_config)
        assert config.name == "test_experiment"
        assert config.seed == 42
        assert config.dataset.name == "ts_haystack"
        assert config.model.architecture == "flamingo"

    def test_from_dict_with_runtime(self, sample_config):
        config = ExperimentConfig.from_dict(sample_config)
        assert config.runtime.run_name == "test_run"
        assert config.runtime.max_samples == 100

    def test_defaults_no_args(self):
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert config.seed == 42
        assert config.dataset.name == "capture24_haystack_cot"
        assert config.model.architecture == "flamingo"
        assert config.runtime.output_dir == Path("results")

    def test_convenience_properties_defaults(self):
        config = ExperimentConfig()
        assert config.early_stop_patience == 5
        assert config.checkpoint_interval == 3000
        assert config.use_wandb is True
        assert config.wandb_project == "ts-haystack"
        assert config.wandb_entity is None

    def test_convenience_properties_from_training(self):
        config = ExperimentConfig.from_dict({
            "training": {
                "early_stopping": {"patience": 10},
                "checkpointing": {"save_every_n_steps": 500},
                "logging": {
                    "wandb": {
                        "enabled": False,
                        "project": "my-project",
                        "entity": "my-team",
                    }
                },
            },
        })
        assert config.early_stop_patience == 10
        assert config.checkpoint_interval == 500
        assert config.use_wandb is False
        assert config.wandb_project == "my-project"
        assert config.wandb_entity == "my-team"

    def test_to_dict_round_trip(self, sample_config):
        config = ExperimentConfig.from_dict(sample_config)
        d = config.to_dict()
        # Paths should be strings
        assert isinstance(d["runtime"]["output_dir"], str)
        # Round-trip: reconstruct from dict
        config2 = ExperimentConfig.from_dict(d)
        assert config2.name == config.name
        assert config2.seed == config.seed
        assert config2.dataset.name == config.dataset.name
        assert config2.runtime.run_name == config.runtime.run_name

    def test_to_dict_serialisable(self, sample_config):
        """to_dict() output must be JSON-serialisable."""
        import json
        config = ExperimentConfig.from_dict(sample_config)
        json.dumps(config.to_dict())  # should not raise

    def test_from_yaml(self, temp_dir):
        save_yaml(
            {"name": "yaml_test", "seed": 7, "dataset": {"name": "test_ds"}},
            temp_dir / "exp.yaml",
        )
        config = ExperimentConfig.from_yaml(temp_dir / "exp.yaml")
        assert config.name == "yaml_test"
        assert config.seed == 7
        assert config.dataset.name == "test_ds"
