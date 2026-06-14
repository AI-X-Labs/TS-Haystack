# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""TimesFM 2.5 pretrained time series encoder.

Wraps Google's TimesFM 2.5 (200M) transformer encoder as a composable
:class:`TimeSeriesEncoderBase` for use with Flamingo (univariate) and
ITFormer (multivariate via batch-flattening).

Requires ``uv pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'``.
Registered in ENCODER_REGISTRY as ``timesfm``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.ts_encoder.base import TimeSeriesEncoderBase

_TOLERANCE = 1e-6


class TimesFMTSEncoder(TimeSeriesEncoderBase):
    """Pretrained TimesFM 2.5 encoder usable as a drop-in TS encoder.

    Univariate (Flamingo): ``(B, L) -> (B, N, D)``
    Multivariate (ITFormer): ``(B, L, V) -> (B, V, N, D)``

    TimesFM has no native multi-variable support — multi-variable mode
    simply flattens V into the batch dimension.
    """

    # No patch_size attribute — Flamingo skips external patching,
    # TimesFM handles its own patching internally.

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        output_dim: int = 1280,
        dropout: float = 0.0,
        freeze: bool = True,
        multi_variable: bool = False,
        # Accept and ignore Flamingo's default kwargs:
        max_patches: int | None = None,
        trained_patches: int | None = None,
        **kwargs,
    ):
        try:
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
        except ImportError as exc:
            raise ImportError(
                "TimesFM encoder requires the timesfm package. "
                "Install it with: uv pip install 'timesfm[torch] @ "
                "git+https://github.com/google-research/timesfm.git'"
            ) from exc

        super().__init__(output_dim=output_dim, dropout=dropout)

        # Build the raw module (no auto-compile, no forced re-download)
        self.timesfm_module = TimesFM_2p5_200M_torch_module()

        # Load pretrained weights via hf_hub_download + safetensors
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        weights_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
        )
        tensors = load_file(weights_path)
        self.timesfm_module.load_state_dict(tensors, strict=True)

        # Auto-detect dims from model
        self.output_dim = self.timesfm_module.md  # 1280
        self.patch_len = self.timesfm_module.p  # 32
        self.multi_variable = multi_variable

        self.freeze = freeze
        if freeze:
            self.timesfm_module.requires_grad_(False)

    @staticmethod
    def _running_revin(
        patched: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RevIN normalization using running stats (Welford's algorithm).

        Reimplements TimesFM's normalization from ``timesfm.torch.util`` to
        avoid deep internal imports.

        Args:
            patched: ``(B, N, P)`` — patched time series data.
            masks: ``(B, N, P)`` — boolean mask (True = padding/masked).

        Returns:
            ``(B, N, P)`` — normalized patches.
        """
        B, N, P = patched.shape
        device = patched.device

        n = torch.zeros(B, device=device)
        mu = torch.zeros(B, device=device)
        sigma = torch.zeros(B, device=device)

        patch_mu = []
        patch_sigma = []

        for i in range(N):
            x_patch = patched[:, i]  # (B, P)
            mask_patch = masks[:, i]  # (B, P)
            is_legit = ~mask_patch

            inc_n = is_legit.to(x_patch.dtype).sum(dim=-1)  # (B,)
            inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)

            inc_mu = (x_patch * is_legit).sum(dim=-1) / inc_n_safe
            inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

            inc_var = (((x_patch - inc_mu.unsqueeze(-1)) ** 2) * is_legit).sum(dim=-1) / inc_n_safe
            inc_var = torch.where(inc_n == 0, 0.0, inc_var)
            inc_sigma = torch.sqrt(inc_var)

            new_n = n + inc_n
            new_n_safe = torch.where(new_n == 0, 1.0, new_n)

            new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
            new_mu = torch.where(new_n == 0, 0.0, new_mu)

            term1 = n * sigma.pow(2)
            term2 = inc_n * inc_sigma.pow(2)
            term3 = n * (mu - new_mu).pow(2)
            term4 = inc_n * (inc_mu - new_mu).pow(2)

            new_var = (term1 + term2 + term3 + term4) / new_n_safe
            new_var = torch.where(new_n == 0, 0.0, new_var)
            new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

            n, mu, sigma = new_n, new_mu, new_sigma
            patch_mu.append(mu)
            patch_sigma.append(sigma)

        # Stack per-patch running stats: (B, N)
        all_mu = torch.stack(patch_mu, dim=1)
        all_sigma = torch.stack(patch_sigma, dim=1)

        # Apply RevIN: (x - mu) / sigma, with sigma clamped
        normed = (patched - all_mu.unsqueeze(-1)) / torch.where(
            all_sigma.unsqueeze(-1) < _TOLERANCE, 1.0, all_sigma.unsqueeze(-1)
        )
        # Zero out masked positions
        normed = torch.where(masks, 0.0, normed)

        return normed

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """Encode univariate batch through TimesFM.

        Args:
            x: ``(B, L)`` — univariate time series batch.

        Returns:
            ``(B, N, D)`` — post-transformer patch embeddings.
        """
        B, L = x.shape
        P = self.patch_len

        # Left-pad to multiple of patch_len
        pad_len = (-L) % P  # 0 if already aligned
        if pad_len > 0:
            x = torch.cat([torch.zeros(B, pad_len, device=x.device, dtype=x.dtype), x], dim=1)

        padded_L = x.shape[1]
        N = padded_L // P

        # Reshape to patches: (B, N, P)
        patched = x.reshape(B, N, P)

        # Build mask: True = padding (left-padded positions)
        masks = torch.zeros_like(patched, dtype=torch.bool)
        if pad_len > 0:
            masks[:, 0, :pad_len] = True

        # RevIN normalization
        normed = self._running_revin(patched, masks)

        # Forward through TimesFM transformer
        (_, output_embeddings, _, _), _ = self.timesfm_module(normed, masks)

        return output_embeddings  # (B, N, D=1280)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            # x: (B, L, V) — multi-variable input
            B, L, V = x.shape
            # Stack all variables into batch dim: (B*V, L)
            x_flat = x.permute(0, 2, 1).reshape(B * V, L)
            encoded = self._encode_single(x_flat)  # (B*V, N, D)
            _, N, D = encoded.shape
            return encoded.reshape(B, V, N, D)
        else:
            # x: (B, L) — single-channel input
            return self._encode_single(x)
