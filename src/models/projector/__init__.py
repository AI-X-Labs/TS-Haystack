# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Encoder-to-LLM projection layers."""

import torch.nn as nn

from open_flamingo.src.helpers import PerceiverResampler

from src.models.projector.linear import LinearProjector
from src.models.projector.mlp import MLPProjector

PROJECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearProjector,
    "mlp": MLPProjector,
    "perceiver_resampler": PerceiverResampler,
}


def get_projector(name: str) -> type[nn.Module]:
    """Look up a projector class by name.

    Args:
        name: Registry key (e.g. 'linear', 'mlp', 'perceiver_resampler').

    Returns:
        The projector class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in PROJECTOR_REGISTRY:
        raise KeyError(
            f"Unknown projector '{name}'. Available: {list(PROJECTOR_REGISTRY.keys())}"
        )
    return PROJECTOR_REGISTRY[name]


def get_projector_output_dim(projector: nn.Module, input_dim: int) -> int:
    """Return the output dimension of an instantiated projector.

    Projectors that change dimensionality (Linear, MLP) store ``output_dim``.
    Dimension-preserving projectors (PerceiverResampler) have no such attribute,
    so we fall back to *input_dim*.

    Args:
        projector: An instantiated projector module.
        input_dim: The dimension fed into the projector (used as fallback).

    Returns:
        The output feature dimension.
    """
    return getattr(projector, "output_dim", input_dim)


__all__ = [
    "LinearProjector",
    "MLPProjector",
    "PerceiverResampler",
    "PROJECTOR_REGISTRY",
    "get_projector",
    "get_projector_output_dim",
]
