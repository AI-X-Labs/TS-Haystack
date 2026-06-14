# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Gated feedforward primitive with ``tanh(gate)`` scaling."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFeedForward(nn.Module):
    """Gated feedforward with ``tanh(gate)`` scaling, matching Flamingo's pattern.

    Gate is initialised to **0** so the model starts as the vanilla LLM.
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * self.net(x)
