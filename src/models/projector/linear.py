# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, *, dim: int, output_dim: int | None = None, **kwargs):
        super().__init__()
        if output_dim is None:
            output_dim = dim
        self.output_dim = output_dim
        self.projector = nn.Linear(dim, output_dim)

    def forward(self, x):
        return self.projector(x)
