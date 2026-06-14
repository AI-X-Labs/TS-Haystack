# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Sleep PSG classifier — 13-channel input → Chronos-2 (multivariate) → MLP head.

Two configurations:
  - sleep_stages: L = 3000 (30 s @ 100 Hz), 5 classes
  - arousals:     L = 2000 (20 s @ 100 Hz), 6 classes (incl. ``none``)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifiers.encoders import Chronos2ClassifierEncoder


SLEEP_STAGE_CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
AROUSAL_CLASS_NAMES = [
    "rera",
    "hypopnea",
    "obstructive_apnea",
    "central_apnea",
    "mixed_apnea",
    "none",
]


def default_class_names(label_class: str) -> List[str]:
    if label_class == "sleep_stages":
        return list(SLEEP_STAGE_CLASS_NAMES)
    if label_class == "arousals":
        return list(AROUSAL_CLASS_NAMES)
    raise ValueError(f"Unknown label_class: {label_class}")


def default_window_samples(label_class: str, effective_hz: int = 100) -> int:
    if label_class == "sleep_stages":
        return 30 * effective_hz
    if label_class == "arousals":
        return 20 * effective_hz
    raise ValueError(f"Unknown label_class: {label_class}")


class SleepClassifier(nn.Module):
    """Chronos-2 multivariate encoder + classification head for 13-channel PSG.

    Input:  (B, 13, L) at 100 Hz
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        window_samples: int,
        n_channels: int = 13,
        chronos_model_id: str = "amazon/chronos-2",
        freeze_encoder: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        assert len(class_names) == num_classes
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.window_samples = int(window_samples)
        self.n_channels = int(n_channels)
        self.chronos_model_id = chronos_model_id
        self.freeze_encoder = freeze_encoder

        self.chronos_encoder = Chronos2ClassifierEncoder(
            model_id=chronos_model_id,
            freeze=freeze_encoder,
            device=device,
            multivariate=True,
        )
        self.feature_dim = self.chronos_encoder.FEATURE_DIM

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.chronos_encoder(x)  # (B, D)
        return self.head(feats)

    @torch.no_grad()
    def classify_window(self, x: torch.Tensor) -> Tuple[str, float]:
        """Classify a single (C, L) window. Pads/crops to ``window_samples``."""
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, C, L = x.shape
        if L < self.window_samples:
            x = F.pad(x, (0, self.window_samples - L))
        elif L > self.window_samples:
            x = x[:, :, : self.window_samples]
        device = next(self.parameters()).device
        logits = self.forward(x.to(device))
        probs = F.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return self.class_names[int(idx[0].item())], float(conf[0].item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "window_samples": self.window_samples,
            "n_channels": self.n_channels,
            "chronos_model_id": self.chronos_model_id,
            "freeze_encoder": self.freeze_encoder,
            "head_state_dict": self.head.state_dict(),
        }
        if not self.freeze_encoder:
            state["chronos_encoder_state_dict"] = self.chronos_encoder.state_dict()
        torch.save(state, path)
        print(f"SleepClassifier saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        freeze_encoder: bool = True,
    ) -> "SleepClassifier":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_classes=ckpt["num_classes"],
            class_names=ckpt["class_names"],
            window_samples=ckpt["window_samples"],
            n_channels=ckpt.get("n_channels", 13),
            chronos_model_id=ckpt.get("chronos_model_id", "amazon/chronos-2"),
            freeze_encoder=freeze_encoder,
            device=device,
        )
        if "chronos_encoder_state_dict" in ckpt:
            model.chronos_encoder.load_state_dict(ckpt["chronos_encoder_state_dict"])
        model.head.load_state_dict(ckpt["head_state_dict"])
        model.to(device)
        model.eval()
        return model
