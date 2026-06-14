# SPDX-License-Identifier: CC-BY-NC-4.0
"""
UK-DALE appliance classifier — 1-D dilated TCN over single-channel mains
active-power, producing per-sample multi-label predictions over the V1
10-appliance vocabulary.

Input:  (B, 1, L)  log1p(power_w) on the 6 s UK-DALE grid
Output: (B, 10, L) per-sample logits (sigmoid -> probability of each
                   appliance being active in that 6 s sample)

The receptive field of the default config is ~133 samples (~13 min @ 6 s),
enough to span both kettle impulses (~30 s) and the start/end edges of
washer-dryer cycles (~90 min, observed as a sustained level whose change
points lie within the receptive field).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.uk_dale_haystack.core.activity_regimes import V1_VOCAB


UK_DALE_CLASS_NAMES: List[str] = list(V1_VOCAB)
NUM_CLASSES = len(UK_DALE_CLASS_NAMES)


def normalize_power(power_w: torch.Tensor | float) -> torch.Tensor | float:
    """log1p(W). Picked over /1000 because mains spans 0-5kW with a heavy
    tail; log compresses while preserving fridge-vs-kettle ratios."""
    if isinstance(power_w, torch.Tensor):
        return torch.log1p(power_w.clamp(min=0.0))
    import numpy as np
    return np.log1p(np.clip(power_w, 0.0, None))


def featurize_power(power_w):
    """Build the 2-channel input the v2 classifier consumes from raw watts.

    Channel 0: log1p(power) — absolute level.
    Channel 1: signed log1p of first-difference — edge sign + magnitude.
        Kettle / microwave / hair-dryer all peak around 1-3 kW so the level
        channel alone confuses them; their turn-on / turn-off transients
        differ markedly (kettle = sharp rise + slow fall, microwave = fast
        on/off duty cycle, hair-dryer = sustained plateau).

    Accepts torch.Tensor (B, 1, L) or (1, L) or (L,), or numpy (L,).
    Returns the same backend with a leading channel dim of size 2.
    """
    if isinstance(power_w, torch.Tensor):
        x = power_w
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, L)
        if x.dim() == 3:
            assert x.shape[1] == 1, "expected single power channel before featurization"
            x = x.squeeze(1)  # (B, L)
        # x is now (..., L)
        level = torch.log1p(x.clamp(min=0.0))
        diff = torch.diff(x, dim=-1, prepend=x[..., :1])
        edge = torch.sign(diff) * torch.log1p(diff.abs())
        out = torch.stack([level, edge], dim=-2)  # (..., 2, L)
        return out
    import numpy as np
    x = np.asarray(power_w, dtype=np.float32)
    flat = x.ndim == 1
    if flat:
        x = x[None, :]
    level = np.log1p(np.clip(x, 0.0, None))
    diff = np.diff(x, axis=-1, prepend=x[..., :1])
    edge = np.sign(diff) * np.log1p(np.abs(diff))
    out = np.stack([level, edge], axis=-2)  # (B, 2, L) or (2, L) if flat
    return out[0] if flat else out


class _ResidualDilatedBlock(nn.Module):
    """Two dilated convs + residual. Same-padding via padding=dilation*(k-1)//2."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.proj(x), inplace=True)


class UKDaleClassifier(nn.Module):
    """Dilated 1-D TCN producing per-sample multi-label appliance logits."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_names: List[str] = UK_DALE_CLASS_NAMES,
        n_channels: int = 2,
        base_channels: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        block_channels: tuple[int, ...] = (96, 96, 128, 128, 192, 192, 192),
        kernel: int = 3,
        stem_kernel: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(class_names) == num_classes
        assert len(dilations) == len(block_channels)
        self.num_classes = int(num_classes)
        self.class_names = list(class_names)
        self.n_channels = int(n_channels)
        self.base_channels = int(base_channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.block_channels = tuple(int(c) for c in block_channels)
        self.kernel = int(kernel)
        self.stem_kernel = int(stem_kernel)
        self.dropout = float(dropout)

        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, base_channels, stem_kernel,
                      padding=stem_kernel // 2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_c = base_channels
        for d, c in zip(self.dilations, self.block_channels):
            blocks.append(_ResidualDilatedBlock(
                in_c, c, kernel=kernel, dilation=d, dropout=dropout,
            ))
            in_c = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(in_c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) log1p-power. Returns (B, num_classes, L) logits."""
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, L) -> (B, num_classes, L) sigmoid probabilities."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": "UKDaleClassifier",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "n_channels": self.n_channels,
            "base_channels": self.base_channels,
            "dilations": list(self.dilations),
            "block_channels": list(self.block_channels),
            "kernel": self.kernel,
            "stem_kernel": self.stem_kernel,
            "dropout": self.dropout,
            "state_dict": self.state_dict(),
        }, path)
        print(f"UKDaleClassifier saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "UKDaleClassifier":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_classes=ckpt["num_classes"],
            class_names=ckpt["class_names"],
            n_channels=ckpt.get("n_channels", 1),
            base_channels=ckpt.get("base_channels", 64),
            dilations=tuple(ckpt.get("dilations", (1, 2, 4, 8, 16, 32))),
            block_channels=tuple(ckpt.get("block_channels", (64, 64, 128, 128, 128, 128))),
            kernel=ckpt.get("kernel", 3),
            stem_kernel=ckpt.get("stem_kernel", 7),
            dropout=ckpt.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
