# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_time_series():
    """Provide a sample time series tensor."""
    # Shape: (batch, channels, seq_len)
    return torch.randn(2, 3, 1000)


@pytest.fixture
def sample_config():
    """Provide a sample experiment configuration dictionary."""
    return {
        "name": "test_experiment",
        "seed": 42,
        "dataset": {
            "name": "ts_haystack",
            "data_dir": "data",
            "extra_kwargs": {
                "tasks": ["existence"],
                "context_lengths_seconds": [100],
            },
        },
        "model": {
            "architecture": "flamingo",
            "encoder": {
                "type": "patch",
                "patch_size": 16,
                "hidden_dim": 256,
            },
            "projector": {
                "type": "mlp",
                "hidden_dim": 512,
            },
        },
        "backbone": {
            "name": "llama",
            "model_id": "meta-llama/Llama-3.2-1B",
            "peft": {
                "enabled": True,
                "method": "lora",
                "r": 8,
            },
        },
        "training": {
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": {"default": 1e-4},
        },
        "runtime": {
            "output_dir": "results",
            "run_name": "test_run",
            "max_samples": 100,
        },
    }


@pytest.fixture
def device():
    """Provide the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
