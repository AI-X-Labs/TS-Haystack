# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Integration smoke test: train/evaluate on generated LTAF parquet data."""

from __future__ import annotations

import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

import scripts.data.build_ltaf_haystack as build_ltaf_haystack
from src.datasets.ltaf_haystack.qa_dataset import LTAFHaystackQADataset
from src.utils.config import ExperimentConfig


def _import_train_script_with_minimal_stubs():
    """Import ``scripts.train`` without pulling optional full-registry deps.

    This smoke test monkeypatches ``load_model`` and only needs the
    ``ltaf_haystack`` dataset class, so we can safely stub model/dataset
    registries during import to avoid unrelated optional dependencies.
    """

    previous_modules = {
        name: sys.modules.get(name)
        for name in [
            "src.models.base",
            "src.models.registry",
            "src.datasets.registry",
            "scripts.train",
        ]
    }

    model_base_module = types.ModuleType("src.models.base")

    class _BaseModel(torch.nn.Module):
        pass

    model_base_module.BaseModel = _BaseModel

    model_registry_module = types.ModuleType("src.models.registry")

    def _get_model_class(_architecture: str):
        raise RuntimeError("get_model_class should not be called in this test (load_model is monkeypatched)")

    model_registry_module.get_model_class = _get_model_class

    dataset_registry_module = types.ModuleType("src.datasets.registry")

    def _get_dataset_class(name: str):
        if name == "ltaf_haystack":
            return LTAFHaystackQADataset
        raise KeyError(f"Unexpected dataset in smoke test: {name}")

    dataset_registry_module.get_dataset_class = _get_dataset_class

    sys.modules["src.models.base"] = model_base_module
    sys.modules["src.models.registry"] = model_registry_module
    sys.modules["src.datasets.registry"] = dataset_registry_module
    sys.modules.pop("scripts.train", None)

    train_script = importlib.import_module("scripts.train")

    # Restore global module state after import; ``train_script`` keeps direct
    # references to imported callables from stubs for this test invocation.
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    return train_script


def _clear_qadataset_cache(dataset_cls: type) -> None:
    attrs = [
        "_raw_loaded",
        "_raw_train",
        "_raw_val",
        "_raw_test",
        "loaded",
        "_train_dataset",
        "_validation_dataset",
        "_test_dataset",
    ]
    for attr in attrs:
        if attr in dataset_cls.__dict__:
            delattr(dataset_cls, attr)


def _generate_ltaf_parquet(monkeypatch, tmp_path: Path, output_root: Path) -> None:
    """Generate a tiny LTAF-Haystack parquet tree via the natural-only build script."""
    config_path = tmp_path / "ltaf_smoke_gen.yaml"
    config_path.write_text(
        """
global:
  seed: 42
  source_hz: 128
  n_jobs: 1
  label_class: rhythms
  output_dir: data/ltafdb/ltaf_haystack/rhythms/tasks

context_lengths_seconds:
  - 900

samples:
  train: 2
  validation: 1
  test: 1

tasks:
  existence:
    enabled: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
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
            "--max-samples-per-split",
            "2",
            "--tasks",
            "existence",
        ],
    )
    build_ltaf_haystack.main()


class _DummyTokenizer:
    def batch_decode(self, token_ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            batch_size = int(token_ids.shape[0])
        else:
            batch_size = len(token_ids)
        return ["Answer: yes"] * batch_size


class _DummyTrainModel(torch.nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.tokenizer = _DummyTokenizer()

    def prepare_batch(self, batch: list[dict], training: bool = True, normalize: bool = False):
        del normalize
        bsz = len(batch)
        time_series = torch.zeros((bsz, 2, 8), dtype=torch.float32, device=self.device)
        input_ids = torch.zeros((bsz, 4), dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        out = {
            "time_series": time_series,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if training:
            out["labels"] = torch.zeros_like(input_ids)
        return out

    def forward(self, time_series, input_ids, attention_mask, labels=None, **kwargs):
        del time_series, input_ids, attention_mask, labels, kwargs
        # Keep dependency on a trainable parameter so backward/optimizer run.
        loss = (self.weight * 0.0 + 1.0).sum()
        return {"loss": loss}

    def generate(self, time_series, input_ids, attention_mask=None, **generate_kwargs):
        del time_series, attention_mask, generate_kwargs
        return torch.ones((input_ids.shape[0], 3), dtype=torch.long, device=self.device)

    def get_trainable_parameters(self):
        return {"dummy": [self.weight]}

    def get_eos_token(self) -> str:
        return "<eos>"


@pytest.mark.integration
def test_ltaf_train_eval_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    train_script = _import_train_script_with_minimal_stubs()

    data_root = tmp_path / "ltaf_tasks"
    _generate_ltaf_parquet(monkeypatch, tmp_path, data_root)

    # Ensure this run doesn't reuse stale class-level dataset cache from prior tests.
    _clear_qadataset_cache(LTAFHaystackQADataset)

    config = ExperimentConfig.from_dict(
        {
            "name": "ltaf_train_eval_smoke",
            "seed": 42,
            "dataset": {
                "name": "ltaf_haystack",
                "extra_kwargs": {
                    "tasks": ["existence"],
                    "context_lengths_seconds": [900.0],
                    "base_dir": str(data_root),
                    "lazy_loading": True,
                },
            },
            "model": {
                "architecture": "flamingo",
                "encoder": {
                    "type": "cnn_tokenizer",
                    "patch_size": 4,
                    "embed_dim": 32,
                    "trained_patches": 64,
                    "max_patches": 128,
                },
                "projector": {"type": "perceiver_resampler", "num_latents": 8},
            },
            "backbone": {
                "name": "llama",
                "model_id": "meta-llama/Llama-3.2-1B",
            },
            "training": {
                "backbone_training": "freeze",
                "batch_size": 2,
                "num_epochs": 1,
                "learning_rate": {"default": 1.0e-3},
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "dataloader": {"num_workers": 0},
                "early_stopping": {"enabled": False},
                "logging": {"wandb": {"enabled": False}},
                "checkpointing": {"save_best": True, "metric": "val_loss", "mode": "min"},
                "eval_max_new_tokens": 8,
            },
            "runtime": {
                "output_dir": str(tmp_path / "results"),
                "run_name": "ltaf_smoke",
                "max_samples": 4,
                "eval_samples_ratio": 0.5,
            },
        }
    )

    monkeypatch.setattr(train_script, "get_device", lambda: "cpu")
    monkeypatch.setattr(train_script, "load_model", lambda cfg, device: _DummyTrainModel(device).to(device))

    history = train_script.train(config)

    run_dir = tmp_path / "results" / "ltaf_haystack" / "ltaf_smoke"
    assert run_dir.exists()

    history_path = run_dir / "history.json"
    assert history_path.exists()

    with open(history_path, encoding="utf-8") as f:
        on_disk_history = json.load(f)

    assert len(on_disk_history["train_loss"]) == 1
    assert len(on_disk_history["val_loss"]) == 1
    assert math.isfinite(on_disk_history["test_loss"])

    assert (run_dir / "checkpoints" / "best_model.pt").exists()
    assert (run_dir / "output_logs" / "val_epoch_1.json").exists()
    assert (run_dir / "output_logs" / "test_epoch_0.json").exists()

    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert math.isfinite(history["test_loss"])
