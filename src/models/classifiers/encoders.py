# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Shared frozen encoders for the ARTS per-domain classifiers.

Provides three encoder backends:
  - OxWearablesEncoder: ResNet-V2 1D pretrained on 700k person-days (1024-dim)
  - Chronos2ClassifierEncoder: Chronos-2 foundation model with pooled patches
  - Both can be used independently or fused in dual-encoder mode via HARClassifier

OxWearables reference:
    Yuan et al. "Self-supervised learning for Human Activity Recognition
    Using 700,000 Person-days of Wearable Data" (NPJ Digital Medicine, 2024)
    https://github.com/OxWearables/ssl-wearables
"""

from __future__ import annotations

import torch
import torch.nn as nn
from chronos import Chronos2Model


# ---------------------------------------------------------------------------
# OxWearables encoder
# ---------------------------------------------------------------------------

OXWEARABLES_REPO = "OxWearables/ssl-wearables"
OXWEARABLES_FEATURE_DIM = 1024

# Keep backward-compatible alias
FEATURE_DIM = OXWEARABLES_FEATURE_DIM


def load_pretrained_encoder(device: str = "cpu") -> nn.Module:
    """Load the OxWearables SSL feature extractor via torch.hub.

    Returns only the ``feature_extractor`` part of the harnet10 model
    (the ResNet backbone). The ``classifier`` head is discarded.

    Args:
        device: Target device.

    Returns:
        nn.Module whose forward takes (B, 3, 300) and returns (B, 1024).
    """
    harnet = torch.hub.load(OXWEARABLES_REPO, "harnet10", class_num=5, pretrained=True)
    encoder = harnet.feature_extractor
    encoder.eval()
    return encoder.to(device)


class OxWearablesEncoder(nn.Module):
    """Wrapper that loads harnet10 feature_extractor and flattens output.

    Input:  (B, 3, 300) — 3 accel axes, 10 s @ 30 Hz
    Output: (B, 1024) feature vector
    """

    FEATURE_DIM = OXWEARABLES_FEATURE_DIM

    def __init__(self, pretrained: bool = True, device: str = "cpu"):
        super().__init__()
        if pretrained:
            self.feature_extractor = load_pretrained_encoder(device)
        else:
            harnet = torch.hub.load(OXWEARABLES_REPO, "harnet10", class_num=5, pretrained=False)
            self.feature_extractor = harnet.feature_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, L) — accelerometer data (3 axes).

        Returns:
            (B, 1024) feature vector.
        """
        feats = self.feature_extractor(x)  # (B, 1024, 1) for 300-sample input
        return feats.view(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Chronos-2 classifier encoder
# ---------------------------------------------------------------------------


class Chronos2ClassifierEncoder(nn.Module):
    """Chronos-2 foundation model adapted for classification.

    Uses Chronos2Model (not ChronosPipeline) to match the pattern in
    ``src/models/ts_encoder/chronos2_encoder.py``.

    Pipeline:
        (B, 3, L) → flatten to (B*3, L) → Chronos-2 encode → (B*3, N, D)
        → avg pool patches → (B*3, D) → reshape (B, 3, D) → avg pool axes → (B, D)

    Input:  (B, 3, L) at 30 Hz
    Output: (B, D) where D = Chronos-2 d_model
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        freeze: bool = True,
        device: str = "cpu",
        multivariate: bool = False,
    ):
        super().__init__()
        self.chronos_model = Chronos2Model.from_pretrained(model_id).to(device)
        self._feature_dim = self.chronos_model.config.d_model
        self.model_id = model_id
        self.multivariate = multivariate

        if freeze:
            self.chronos_model.requires_grad_(False)

    @property
    def FEATURE_DIM(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, L) — triaxial accelerometer data.

        Returns:
            (B, D) feature vector where D = Chronos-2 d_model.
        """
        B, C, L = x.shape

        # Flatten axes into batch: (B*C, L)
        x_flat = x.reshape(B * C, L)

        # In multivariate mode, group all C channels of each sample together so
        # Chronos-2's GroupSelfAttention can mix across channels.
        group_ids = None
        if self.multivariate:
            group_ids = torch.arange(B, device=x.device).repeat_interleave(C)

        # Encode through Chronos-2
        ctx = torch.no_grad() if not any(p.requires_grad for p in self.chronos_model.parameters()) else torch.enable_grad()
        with ctx:
            encoder_outputs, _loc_scale, _fut_mask, num_context_patches = self.chronos_model.encode(
                context=x_flat,
                group_ids=group_ids,
                num_output_patches=1,
            )
            # Keep only context patches: (B*C, num_context_patches, D)
            hidden = encoder_outputs.last_hidden_state[:, :num_context_patches, :]

        # Average pool across patches: (B*C, N, D) → (B*C, D)
        pooled = hidden.mean(dim=1)

        # Reshape and average across channels: (B, C, D) → (B, D).
        # In multivariate mode, cross-channel info is already mixed in by the
        # encoder, so the head input stays D rather than C*D.
        pooled = pooled.reshape(B, C, -1).mean(dim=1)

        return pooled
