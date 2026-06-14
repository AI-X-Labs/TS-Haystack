# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""LLM backbone implementations for TS-Haystack."""

from src.backbones.base import BaseBackbone
from src.backbones.llama import LlamaBackbone

BACKBONE_REGISTRY: dict[str, type[BaseBackbone]] = {
    "llama": LlamaBackbone,
}


def get_backbone(name: str) -> type[BaseBackbone]:
    """Look up a backbone class by name.

    Args:
        name: Registry key (e.g. 'llama').

    Returns:
        The backbone class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name]


__all__ = [
    "BaseBackbone",
    "LlamaBackbone",
    "BACKBONE_REGISTRY",
    "get_backbone",
]
