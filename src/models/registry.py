# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Model registry mapping architecture names to model classes."""

from typing import Any

from src.models.base import BaseModel
from src.models.ts_llm.opentslm_flamingo import OpenTSLMFlamingo
from src.models.ts_llm.itformer import ITFormerModel
from src.models.ts_llm.chatTS import ChatTSModel
from src.models.ts_llm.chattime import ChatTimeModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "flamingo": OpenTSLMFlamingo,
    "itformer": ITFormerModel,
    "chatts": ChatTSModel,
    "chattime": ChatTimeModel,
}


def get_model_class(architecture: str) -> type[BaseModel]:
    """Look up a model class by architecture name.

    Args:
        architecture: Registry key (e.g. 'flamingo').

    Returns:
        The model class.

    Raises:
        KeyError: If the architecture is not registered.
    """
    if architecture not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture '{architecture}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]


def create_model(config: dict[str, Any]) -> BaseModel:
    """Instantiate a model from a configuration dictionary.

    The config must contain an 'architecture' key that maps to a registered
    model class. All other keys are forwarded to the model constructor.

    Args:
        config: Configuration dict. Must include 'architecture'.

    Returns:
        An instantiated model.
    """
    config = dict(config)  # shallow copy to avoid mutating caller's dict
    architecture = config.pop("architecture")
    cls = get_model_class(architecture)
    return cls(**config)


def list_architectures() -> list[str]:
    """Return all registered architecture names."""
    return list(MODEL_REGISTRY.keys())
