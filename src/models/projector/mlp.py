# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, *, dim: int, output_dim: int | None = None, dropout: float = 0.0, **kwargs):
        super().__init__()
        if output_dim is None:
            output_dim = dim
        self.output_dim = output_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.projector(x)
