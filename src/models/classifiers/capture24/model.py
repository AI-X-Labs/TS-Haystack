# SPDX-License-Identifier: CC-BY-NC-4.0
"""
HAR Classifier with pluggable encoder backends.

Supports three encoder modes:
  - "oxwearables": OxWearables SSL encoder (1024-dim) — domain-specific
  - "chronos2": Chronos-2 foundation model (d_model-dim) — general-purpose
  - "dual": Both encoders fused via concatenation — best of both worlds

All modes support variable-length inputs. Bouts longer than 10 s are split
into independent 10 s chunks, each classified separately.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifiers.encoders import (
    OxWearablesEncoder,
    Chronos2ClassifierEncoder,
    OXWEARABLES_FEATURE_DIM,
    FEATURE_DIM,  # backward compat alias
)


# WillettsSpecific2018 — 10 classes (alphabetically sorted to match classification.py get_class_names)
WILLETTS_SPECIFIC_2018_CLASSES = [
    "bicycling",
    "household-chores",
    "manual-work",
    "mixed-activity",
    "sitting",
    "sleep",
    "sports",
    "standing",
    "vehicle",
    "walking",
]


class HARClassifier(nn.Module):
    """Pluggable encoder + trainable classification head.

    Input:  (B, 3, L) at 30 Hz — variable length
    Output: (B, num_classes) logits

    Encoder modes:
        - "oxwearables": OxWearables SSL (1024-dim features)
        - "chronos2": Chronos-2 foundation model (d_model-dim features)
        - "dual": Concatenated features from both encoders

    Variable-length handling:
        - L < 300 (< 10 s): zero-pad to 300, classify as one window
        - L == 300 (10 s): classify directly
        - L > 300 (> 10 s): split into non-overlapping 10 s chunks,
          classify each independently (remainder < 10 s is padded)
    """

    WINDOW_SAMPLES = 300  # 10 s @ 30 Hz

    def __init__(
        self,
        num_classes: int = 10,
        class_names: list[str] | None = None,
        encoder_type: str = "oxwearables",
        pretrained_encoder: bool = True,
        freeze_encoder: bool = True,
        chronos_model_id: str = "amazon/chronos-2",
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or WILLETTS_SPECIFIC_2018_CLASSES
        self.encoder_type = encoder_type
        assert len(self.class_names) == num_classes

        # Build encoder(s)
        self.encoder = None
        self.chronos_encoder = None
        feature_dim = 0

        if encoder_type in ("oxwearables", "dual"):
            self.encoder = OxWearablesEncoder(pretrained=pretrained_encoder, device=device)
            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            feature_dim += OXWEARABLES_FEATURE_DIM

        if encoder_type in ("chronos2", "dual"):
            self.chronos_encoder = Chronos2ClassifierEncoder(
                model_id=chronos_model_id, freeze=freeze_encoder, device=device,
            )
            if freeze_encoder:
                for p in self.chronos_encoder.parameters():
                    p.requires_grad = False
            feature_dim += self.chronos_encoder.FEATURE_DIM

        self.feature_dim = feature_dim

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder(s) and return feature vector.

        Args:
            x: (B, 3, 300) accelerometer data at 30 Hz.

        Returns:
            (B, feature_dim) features.
        """
        features = []

        if self.encoder is not None:
            features.append(self.encoder(x))

        if self.chronos_encoder is not None:
            features.append(self.chronos_encoder(x))

        return torch.cat(features, dim=-1) if len(features) > 1 else features[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a single fixed-size window (used for training).

        Args:
            x: (B, 3, 300) accelerometer data at 30 Hz.

        Returns:
            (B, num_classes) logits.
        """
        features = self._encode(x)
        return self.head(features)

    @torch.no_grad()
    def classify_bout(self, bout_data: torch.Tensor) -> list[tuple[str, float]]:
        """Classify a variable-length bout, returning per-chunk results.

        Bouts are split into non-overlapping 10 s chunks. Each chunk
        gets its own (class_name, confidence) prediction. Short bouts
        (< 10 s) are zero-padded to 300 samples.

        Args:
            bout_data: (3, L) or (1, 3, L) accelerometer data at 30 Hz.

        Returns:
            List of (class_name, confidence) tuples, one per 10 s chunk.
        """
        self.eval()
        if bout_data.dim() == 2:
            bout_data = bout_data.unsqueeze(0)  # (1, 3, L)

        _, C, L = bout_data.shape

        # Split into 10 s chunks
        chunks = []
        for start in range(0, L, self.WINDOW_SAMPLES):
            chunk = bout_data[:, :, start : start + self.WINDOW_SAMPLES]
            if chunk.shape[2] < self.WINDOW_SAMPLES:
                chunk = F.pad(chunk, (0, self.WINDOW_SAMPLES - chunk.shape[2]))
            chunks.append(chunk)

        if not chunks:
            return []

        batch = torch.cat(chunks, dim=0)  # (n_chunks, 3, 300)
        logits = self.forward(batch)  # (n_chunks, num_classes)
        probs = F.softmax(logits, dim=-1)
        confidences, pred_indices = probs.max(dim=-1)

        results = []
        for conf, idx in zip(confidences, pred_indices):
            results.append((self.class_names[idx.item()], conf.item()))
        return results

    def save(self, path: str | Path) -> None:
        """Save full classifier state (encoder(s) + head + metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "head_state_dict": self.head.state_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "encoder_type": self.encoder_type,
            "feature_dim": self.feature_dim,
        }

        if self.encoder is not None:
            state["encoder_state_dict"] = self.encoder.state_dict()
        if self.chronos_encoder is not None:
            state["chronos_encoder_state_dict"] = self.chronos_encoder.state_dict()

        torch.save(state, path)
        print(f"HARClassifier ({self.encoder_type}) saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
        freeze_encoder: bool = True,
    ) -> "HARClassifier":
        """Load a trained classifier from checkpoint.

        Args:
            path: Path to the checkpoint file.
            device: Device to load the model on.
            freeze_encoder: If False, encoder parameters will be unfrozen
                (useful for resuming training with fine-tuning).

        Backward compatible: checkpoints without encoder_type default to "oxwearables".
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        encoder_type = checkpoint.get("encoder_type", "oxwearables")

        model = cls(
            num_classes=checkpoint["num_classes"],
            class_names=checkpoint["class_names"],
            encoder_type=encoder_type,
            pretrained_encoder=True,
            freeze_encoder=freeze_encoder,
            device=device,
        )

        if model.encoder is not None and "encoder_state_dict" in checkpoint:
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        if model.chronos_encoder is not None and "chronos_encoder_state_dict" in checkpoint:
            model.chronos_encoder.load_state_dict(checkpoint["chronos_encoder_state_dict"])

        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.to(device)
        model.eval()
        return model
