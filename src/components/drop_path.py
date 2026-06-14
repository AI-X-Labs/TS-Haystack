# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Stochastic depth (drop path) regularisation — lightweight alternative to timm."""

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularisation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
        return x.div(keep_prob) * mask
