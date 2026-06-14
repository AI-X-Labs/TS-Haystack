# SPDX-License-Identifier: CC-BY-NC-4.0
"""From-scratch 1D-ResNet rhythm classifier for LTAF.

`RhythmResNet1D` — accepts variable input length via final adaptive avg
pool, so the same module trains on mixed window sizes. Production model
configured with base_channels=64, blocks_per_stage=2 (8.79 M params).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


RHYTHM_CLASS_NAMES_6 = ["NSR", "AFIB", "SBR", "AB", "SVTA", "B"]


class _BasicBlock1D(nn.Module):
    """Two-conv residual block with optional stride-2 downsample."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 7, stride: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernel, stride=1,
                               padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_c != out_c:
            self.proj = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + identity, inplace=True)


class RhythmResNet1D(nn.Module):
    """1D ResNet — stem + 4 stages, each stage halves time and doubles channels."""

    def __init__(
        self,
        num_classes: int = 6,
        class_names: List[str] = RHYTHM_CLASS_NAMES_6,
        n_channels: int = 2,
        base_channels: int = 64,
        blocks_per_stage: int = 2,
        stem_kernel: int = 15,
        block_kernel: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(class_names) == num_classes
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.n_channels = int(n_channels)
        self.base_channels = int(base_channels)
        self.blocks_per_stage = int(blocks_per_stage)

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_channels, kernel_size=stem_kernel,
                      stride=2, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        stages = []
        in_c = base_channels
        out_c = base_channels
        for s in range(4):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and s > 0) else 1
                stages.append(_BasicBlock1D(in_c, out_c, kernel=block_kernel,
                                            stride=stride, dropout=dropout))
                in_c = out_c
            out_c = min(out_c * 2, 512)
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_c, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stages(h)
        feat = self.pool(h).squeeze(-1)
        return self.head(feat)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": "RhythmResNet1D",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "n_channels": self.n_channels,
            "base_channels": self.base_channels,
            "blocks_per_stage": self.blocks_per_stage,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RhythmResNet1D":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_classes=ckpt["num_classes"], class_names=ckpt["class_names"],
            n_channels=ckpt["n_channels"],
            base_channels=ckpt.get("base_channels", 64),
            blocks_per_stage=ckpt.get("blocks_per_stage", 2),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model
