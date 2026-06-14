# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
from torch import nn
from open_flamingo import Flamingo
from open_flamingo.src.helpers import PerceiverResampler
from einops import rearrange


class TimeSeriesFlamingoWithTrainableEncoder(Flamingo):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        num_latents: int = 64,
        projector_instance: nn.Module | None = None,
    ):
        """
        TimeSeriesFlamingoWithTrainableEncoder with configurable projector.

        Args:
            num_latents: Number of latent tokens (M) in the Perceiver Resampler.
                Controls the compression ratio of time series features.
                Default is 64 for backwards compatibility.
            projector_instance: A pre-built projector module. When provided,
                replaces the default PerceiverResampler. Use this with the
                projector registry for full control over the projection layer.
        """
        super().__init__(vision_encoder, lang_encoder, eoc_token_id, media_token_id, vis_dim, cross_attn_every_n_layers, gradient_checkpointing)

        if projector_instance is not None:
            self.perceiver = projector_instance
        elif num_latents != 64:
            # Override perceiver with configurable num_latents for ablation studies
            # The base Flamingo class hardcodes num_latents=64, so we replace it here
            self.perceiver = PerceiverResampler(dim=vis_dim, num_latents=num_latents)

        # PerceiverResampler expects 5D input (b, T, F, p, d) and reduces p → M.
        # Linear/MLP projectors work on the last dim only and expect 4D (b, T, tokens, d).
        self._projector_handles_5d = isinstance(self.perceiver, PerceiverResampler)

    # Override the _encode_vision_x method to handle time series data
    # In the original Flamingo, the vision_encoder is a CLIPModel, which is not trainable (with torch.no_grad())
    # Here, we use a TimeSeriesCNNEncoder, which is trainable
    def _encode_vision_x(self, vision_x):
        # Handle time series data while still using the TimeSeriesCNNEncoder
        if vision_x.ndim == 4:  # For shape (b, T_img, F, features)
            b, T, F, features = vision_x.shape

            if getattr(self.vision_encoder, "multi_variable", False):
                # Multi-variable encoder: pass all channels together
                vision_x = rearrange(vision_x, "b T F c -> (b T) c F")  # (b*T, L, V)
                vision_x = self.vision_encoder(vision_x)  # (b*T, V, N, D)
                vision_x = rearrange(vision_x, "(b T) V N D -> b T V N D", b=b, T=T)
                # Flatten V and N for the projector: treat like (b, T, V*N, D)
                V, N = vision_x.shape[2], vision_x.shape[3]
                vision_x = rearrange(vision_x, "b T V N D -> b T (V N) D")
                # Expand F dim for compatibility: (b, T, 1, V*N, D)
                vision_x = vision_x.unsqueeze(2)
                F = 1  # override F for downstream reshape
            else:
                # Single-channel encoder: process channels independently (existing path)
                vision_x = rearrange(vision_x, "b T F c -> (b T F) c")
                vision_x = self.vision_encoder(vision_x)  # [(b*T*F), patches, features]
                vision_x = rearrange(vision_x, "(b T F) p d -> b T F p d", b=b, T=T, F=F)

            # Process through projector (perceiver or linear/mlp)
            if self._projector_handles_5d:
                vision_x = self.perceiver(vision_x)
            else:
                vision_x = rearrange(vision_x, "b T F p d -> b T (F p) d")
                vision_x = self.perceiver(vision_x)

        else:
            # Original image processing path
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            assert F == 1, "Only single frame supported"

            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")

            vision_x = self.vision_encoder(vision_x)[1]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

            if self._projector_handles_5d:
                vision_x = self.perceiver(vision_x)
            else:
                vision_x = rearrange(vision_x, "b T F p d -> b T (F p) d")
                vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
