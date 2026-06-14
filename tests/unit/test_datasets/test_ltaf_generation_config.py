# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Unit tests for the natural-only LTAF generation config."""

from src.datasets.ltaf_haystack.generation.config import LTAFGenerationConfig


def test_default_config_is_natural_only():
    cfg = LTAFGenerationConfig.load_default()

    assert cfg.context_lengths_seconds == [100.0, 900.0, 3600.0, 7200.0]
    assert cfg.label_class == "rhythms"
    assert cfg.seed == 42
    assert cfg.source_hz == 128

    # No insertion-era fields should survive.
    assert not hasattr(cfg, "splicer")
    assert not hasattr(cfg, "natural_pool_min")

    # All 10 tasks enabled.
    assert len(cfg.tasks) == 10
    assert set(cfg.tasks.keys()) == {
        "existence", "localization", "counting", "ordering", "state_query",
        "antecedent", "comparison", "multi_hop", "anomaly_detection",
        "anomaly_localization",
    }
    for name, tcfg in cfg.tasks.items():
        assert tcfg.enabled, name
        # No insertion mode flags on the task config either.
        assert not hasattr(tcfg, "natural_only")
        assert not hasattr(tcfg, "insertion_only")


def test_task_config_overrides_parse(temp_dir):
    cfg_path = temp_dir / "ltaf_generation.yaml"
    cfg_path.write_text(
        """
global:
  seed: 7
  source_hz: 128
  label_class: rhythms
context_lengths_seconds: [100, 900]
tasks:
  existence:
    enabled: true
  counting:
    enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    cfg = LTAFGenerationConfig.from_yaml(cfg_path)
    assert cfg.seed == 7
    assert cfg.context_lengths_seconds == [100.0, 900.0]
    assert cfg.get_enabled_tasks() == ["existence"]
