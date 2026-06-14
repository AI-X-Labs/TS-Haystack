# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Alignment projector: maps encoder dimension to LLM dimension."""

from __future__ import annotations

import torch
import torch.nn as nn


class AlignmentProjector(nn.Module):
    """Two-layer MLP that maps encoder dim to LLM dim with LayerNorm + GELU."""

    def __init__(self, enc_dim: int, llm_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
