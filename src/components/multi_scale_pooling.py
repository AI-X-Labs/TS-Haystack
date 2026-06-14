# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Multi-scale temporal pooling primitive."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePooling(nn.Module):
    """Pool encoder output at multiple temporal scales.

    Given ``(B, N, D)`` input, produces pooled features at each scale via
    average-pooling + LayerNorm.

    Args:
        dim: Feature dimension.
        scales: List of pooling scales (default ``[1, 4, 16]``).
        return_list: If ``False`` (default), returns ``torch.cat(parts, dim=1)``
            — a single tensor of shape ``(B, sum(ceil(N/s) for s in scales), D)``.
            If ``True``, returns a ``list[Tensor]`` with one ``(B, ceil(N/s), D)``
            tensor per scale.
    """

    def __init__(self, dim: int, scales: list[int] | None = None, return_list: bool = False):
        super().__init__()
        self.scales = scales or [1, 4, 16]
        self.return_list = return_list
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in self.scales])

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """``(B, N, D)`` -> concatenated tensor or list of per-scale tensors."""
        parts = []
        for scale, norm in zip(self.scales, self.norms):
            if scale == 1:
                parts.append(norm(x))
            else:
                # avg_pool1d expects (B, D, N)
                pooled = F.avg_pool1d(
                    x.transpose(1, 2), kernel_size=scale, stride=scale, ceil_mode=True
                )  # (B, D, ceil(N/scale))
                parts.append(norm(pooled.transpose(1, 2)))
        if self.return_list:
            return parts
        return torch.cat(parts, dim=1)
