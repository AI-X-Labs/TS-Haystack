# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""ChatTSModel — LLaVA-style time series LLM implementing BaseModel.

Data flow (SP encoding — mlp_encoder):
    Per-channel time_series  →  SP-encode  →  (flat_len,)
    All channels batched     →  MLPEncoder →  (total_patches, hidden_size)
        →  replace <ts_pad> token positions in LLM input embeddings
            →  LLM forward  →  loss

Data flow (raw encoding — any encoder):
    Per-channel time_series  →  pad to patch boundary  →  (L,)
    All channels batched     →  Encoder  →  (total_channels, N, encoder_dim)
        →  flatten  →  (total_patches, encoder_dim)
            →  optional Linear projector  →  (total_patches, llm_dim)
                →  replace <ts_pad> token positions
                    →  LLM forward  →  loss

Reference: ChatTS (ByteDance Research) — extended for TSLM-Bench composability.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.models.base import BaseModel
from src.models.ts_encoder import get_encoder
from src.models.projector import get_projector

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class ChatTSModel(BaseModel):
    """ChatTS: LLaVA-style TSLM with placeholder-token embedding replacement.

    Supports two encoding modes controlled by ``encoding_method``:

    * ``"sp"`` (default) — SP-encodes each channel, uses ``mlp_encoder``.
      Encoder ``hidden_size`` must equal the LLM hidden size.
    * ``"raw"`` — passes raw time series to any encoder from the registry.
      A linear projector bridges ``encoder.output_dim → llm_hidden_size``
      when dimensions differ.
    """

    TS_PAD_TOKEN = "<ts_pad>"

    def __init__(
        self,
        device: str,
        llm_model: nn.Module,
        text_tokenizer: Any,
        ts_encoder: nn.Module,
        backbone: Any,
        patch_size: int = 20,
        num_features: int = 2,
        freeze_ts_encoder: bool = False,
        ts_projector: nn.Module | None = None,
        encoding_method: str = "sp",
        freeze_lm_embeddings: bool = False,
    ):
        super().__init__(device)
        self.llm_model = llm_model
        self.text_tokenizer = text_tokenizer
        self.ts_encoder = ts_encoder
        self.backbone = backbone
        self.patch_size = patch_size
        self.num_features = num_features
        self.freeze_ts_encoder = freeze_ts_encoder
        self.ts_projector = ts_projector
        self.encoding_method = encoding_method
        self.freeze_lm_embeddings = freeze_lm_embeddings

        if freeze_ts_encoder:
            for p in self.ts_encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "ChatTSModel":
        from src.backbones import get_backbone

        # --- Backbone ---
        backbone_cls = get_backbone(config.backbone.name)
        backbone = backbone_cls(model_id=config.backbone.model_id)

        text_tokenizer = backbone.load_tokenizer()
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        # Add special token for TS placeholder
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": [cls.TS_PAD_TOKEN]}
        )

        llm_model = backbone.load_model(device_map={"": device})
        llm_model.resize_token_embeddings(len(text_tokenizer))
        llm_dim = llm_model.config.hidden_size

        # --- Encoder from registry ---
        enc_cfg = dict(config.model.encoder)
        encoder_type = enc_cfg.pop("type", "mlp_encoder")

        # Map embed_dim → output_dim for consistency with Flamingo configs
        if "embed_dim" in enc_cfg:
            enc_cfg["output_dim"] = enc_cfg.pop("embed_dim")

        encoder_cls = get_encoder(encoder_type)
        ts_encoder = encoder_cls(**enc_cfg).to(device)

        extra = config.model.extra_kwargs
        freeze_ts_encoder = extra.get("freeze_ts_encoder", False)
        encoding_method = extra.get("encoding_method", "sp")
        freeze_lm_embeddings = extra.get("freeze_lm_embeddings", False)

        # --- Optional projector when encoder dim != LLM dim ---
        # Uses PROJECTOR_REGISTRY (linear, mlp, perceiver_resampler).
        # Reads from config.model.projector if present, defaults to linear.
        ts_projector = None
        if ts_encoder.output_dim != llm_dim:
            proj_cfg = dict(config.model.projector) if config.model.projector else {}
            proj_type = proj_cfg.pop("type", "linear")
            proj_cls = get_projector(proj_type)
            ts_projector = proj_cls(
                dim=ts_encoder.output_dim, output_dim=llm_dim, **proj_cfg
            ).to(device)

        # Align to LLM dtype
        llm_dtype = next(llm_model.parameters()).dtype
        ts_encoder.to(llm_dtype)
        if ts_projector is not None:
            ts_projector.to(llm_dtype)

        return cls(
            device=device,
            llm_model=llm_model,
            text_tokenizer=text_tokenizer,
            ts_encoder=ts_encoder,
            backbone=backbone,
            patch_size=enc_cfg.get("patch_size", 20),
            num_features=enc_cfg.get("num_features", 2),
            freeze_ts_encoder=freeze_ts_encoder,
            ts_projector=ts_projector,
            encoding_method=encoding_method,
            freeze_lm_embeddings=freeze_lm_embeddings,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.text_tokenizer

    # ------------------------------------------------------------------
    # SP encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _sp_encode(ts: np.ndarray) -> tuple[np.ndarray, str]:
        """SP-encode a single 1-D time series channel.

        Normalizes (mean-subtract, scale to [-3,3]), then interleaves values
        with 1.0 validity markers.

        Args:
            ts: 1-D numpy array of time series values.

        Returns:
            ``(encoded, metadata_str)`` where encoded is a 1-D array of length
            ``len(ts) * 2`` (interleaved [v0, 1, v1, 1, ...]) and
            metadata_str is e.g. ``"[Value Offset: 0.1234|Value Scaling: 1.5678]"``.
        """
        mean = float(np.mean(ts))
        scaled = ts - mean
        scale_factor = 1.0
        if np.any(np.abs(scaled) >= 3.0):
            scale_factor = float(np.max(np.abs(scaled))) / 3.0
            scaled = scaled / scale_factor

        # Interleave values with 1.0 markers: shape (L,2) → flatten to (L*2,)
        encoded = np.stack([scaled, np.ones_like(scaled)], axis=-1).reshape(-1)
        metadata = f"[Value Offset: {-mean:.4f}|Value Scaling: {scale_factor:.4f}]"
        return encoded, metadata

    # ------------------------------------------------------------------
    # Token merging
    # ------------------------------------------------------------------

    def _merge_features(
        self,
        ts_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Replace ``<ts_pad>`` positions in the input embeddings with
        encoder output features.

        Args:
            ts_features: ``(total_patches, dim)`` — all patches
                concatenated across the batch.  ``dim`` must equal the
                LLM embedding dimension (projection is applied before
                calling this method).
            input_ids: ``(B, seq_len)``

        Returns:
            inputs_embeds: ``(B, seq_len, hidden_size)``
        """
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        pad_id = self.text_tokenizer(
            self.TS_PAD_TOKEN, add_special_tokens=False
        )["input_ids"][0]
        batch_idx, seq_idx = torch.where(input_ids == pad_id)

        total_patches = ts_features.size(0)
        n_positions = len(batch_idx)
        n = min(total_patches, n_positions)

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_idx[:n], seq_idx[:n]] = ts_features[:n].to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Encoder dispatch
    # ------------------------------------------------------------------

    def _encode_and_project(self, time_series: torch.Tensor) -> torch.Tensor:
        """Run encoder and optional projector, returning flat features.

        Returns:
            ``(total_patches, llm_dim)``
        """
        enc_out = self.ts_encoder(time_series)

        if isinstance(enc_out, tuple):
            # mlp_encoder returns (features, patch_cnt)
            features = enc_out[0]  # (total_patches, hidden_size)
        else:
            # Standard encoders return (total_channels, N, d)
            if enc_out.dim() == 3:
                C, N, D = enc_out.shape
                features = enc_out.reshape(C * N, D)
            else:
                features = enc_out

        if self.ts_projector is not None:
            features = self.ts_projector(features)

        return features

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.encoding_method == "sp":
            return self._prepare_batch_sp(batch, training)
        else:
            return self._prepare_batch_raw(batch, training)

    def _prepare_batch_sp(
        self,
        batch: list[dict[str, Any]],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        """Original SP-encoding path (backward compatible with mlp_encoder)."""
        tokenizer = self.text_tokenizer
        ts_pad = self.TS_PAD_TOKEN

        all_encoded_channels: list[np.ndarray] = []
        text_inputs: list[str] = []
        prompt_lengths: list[int] = []

        for item in batch:
            channels = item["time_series"]  # list of 1-D arrays/lists
            channel_texts = item["time_series_text"]  # label per channel

            prompt = item["pre_prompt"]

            for ch_data, ch_text in zip(channels, channel_texts):
                ts_array = np.asarray(ch_data, dtype=np.float64)
                encoded, metadata = self._sp_encode(ts_array)

                # After encoder reshape with num_features, seq_len = len(ts_array)
                # patch_cnt = ceil(seq_len / patch_size)
                seq_len = len(ts_array)
                num_patches = (seq_len + self.patch_size - 1) // self.patch_size

                all_encoded_channels.append(encoded)

                prompt += f"\n{ch_text} {metadata} " + (ts_pad * num_patches)

            if item["post_prompt"]:
                prompt += f"\n{item['post_prompt']}"

            if not training:
                text_inputs.append(prompt)
                continue

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))
            full_text = prompt + " " + item["answer"]
            text_inputs.append(full_text)

        # Pad encoded channels to same flat length and stack
        max_flat_len = max(len(e) for e in all_encoded_channels)
        padded_channels = []
        for e in all_encoded_channels:
            pad_len = max_flat_len - len(e)
            if pad_len > 0:
                e = np.concatenate([e, np.zeros(pad_len)])
            padded_channels.append(e)

        llm_dtype = next(self.llm_model.parameters()).dtype
        time_series = torch.tensor(
            np.stack(padded_channels), dtype=llm_dtype, device=self.device
        )
        # shape: (total_channels_across_batch, max_flat_len)

        return self._finalize_batch(time_series, text_inputs, prompt_lengths, training)

    def _prepare_batch_raw(
        self,
        batch: list[dict[str, Any]],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        """Raw encoding path — pass raw time series to any encoder."""
        tokenizer = self.text_tokenizer
        ts_pad = self.TS_PAD_TOKEN

        all_raw_channels: list[np.ndarray] = []
        text_inputs: list[str] = []
        prompt_lengths: list[int] = []

        for item in batch:
            channels = item["time_series"]  # list of 1-D arrays/lists
            channel_texts = item["time_series_text"]  # label per channel

            prompt = item["pre_prompt"]

            for ch_data, ch_text in zip(channels, channel_texts):
                ts_array = np.asarray(ch_data, dtype=np.float32)

                # Pad to encoder's patch boundary if needed
                if hasattr(self.ts_encoder, "patch_size"):
                    ps = self.ts_encoder.patch_size
                    remainder = len(ts_array) % ps
                    if remainder != 0:
                        pad_len = ps - remainder
                        ts_array = np.concatenate([ts_array, np.zeros(pad_len, dtype=ts_array.dtype)])

                num_patches = self.ts_encoder.get_num_output_patches(len(ts_array))
                all_raw_channels.append(ts_array)

                prompt += f"\n{ch_text} " + (ts_pad * num_patches)

            if item["post_prompt"]:
                prompt += f"\n{item['post_prompt']}"

            if not training:
                text_inputs.append(prompt)
                continue

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))
            full_text = prompt + " " + item["answer"]
            text_inputs.append(full_text)

        # Pad all channels to same length and stack
        max_len = max(len(c) for c in all_raw_channels)
        # Ensure max_len is also a multiple of patch boundary
        if hasattr(self.ts_encoder, "patch_size"):
            ps = self.ts_encoder.patch_size
            remainder = max_len % ps
            if remainder != 0:
                max_len += ps - remainder

        padded_channels = []
        for c in all_raw_channels:
            pad_len = max_len - len(c)
            if pad_len > 0:
                c = np.concatenate([c, np.zeros(pad_len, dtype=c.dtype)])
            padded_channels.append(c)

        llm_dtype = next(self.llm_model.parameters()).dtype
        time_series = torch.tensor(
            np.stack(padded_channels), dtype=llm_dtype, device=self.device
        )
        # shape: (total_channels_across_batch, max_len)

        return self._finalize_batch(time_series, text_inputs, prompt_lengths, training)

    def _finalize_batch(
        self,
        time_series: torch.Tensor,
        text_inputs: list[str],
        prompt_lengths: list[int],
        training: bool,
    ) -> dict[str, torch.Tensor]:
        """Shared tokenization and label creation for both encoding paths."""
        tokenizer = self.text_tokenizer

        # Tokenize — left-pad for generation, right-pad for training
        original_padding = tokenizer.padding_side
        if not training:
            tokenizer.padding_side = "left"

        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        tokenizer.padding_side = original_padding

        if not training:
            return {
                "time_series": time_series,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None,
            }

        # Create labels: mask prompt tokens with -100
        labels = torch.full_like(input_ids, -100)
        for i, prompt_length in enumerate(prompt_lengths):
            non_pad = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_pos = non_pad[non_pad >= prompt_length]
            if len(answer_pos) > 0:
                labels[i, answer_pos] = input_ids[i, answer_pos]

        return {
            "time_series": time_series,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        ts_features = self._encode_and_project(time_series)
        inputs_embeds = self._merge_features(ts_features, input_ids)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        ts_features = self._encode_and_project(time_series)
        inputs_embeds = self._merge_features(ts_features, input_ids)

        with torch.inference_mode():
            gen_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Strip input prefix — return answer tokens only
        input_len = input_ids.shape[1]
        if gen_ids.shape[1] > input_len:
            return gen_ids[:, input_len:]
        return gen_ids

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {}
        if not self.freeze_ts_encoder:
            groups["encoder"] = list(self.ts_encoder.parameters())
        if self.ts_projector is not None:
            groups["projector"] = list(self.ts_projector.parameters())
        if not self.freeze_lm_embeddings:
            groups["lm_embeddings"] = list(self.llm_model.get_input_embeddings().parameters())
        return groups

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    def get_backbone_module(self) -> nn.Module:
        return self.llm_model
