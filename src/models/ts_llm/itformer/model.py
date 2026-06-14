# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""ITFormerModel — instruction-aware time series LLM implementing BaseModel.

Data flow:
    time_series (B,L,V)  →  TSEncoder  →  (B,V,P,d_model)
        →  ts_project  →  (B,V,P,it_d_model)
    query_ids  →  LLM.embed  →  query_project  →  (B,Q,it_d_model)
        →  ITFormerFusion(query, ts, stage)  →  (B,prefix_num,it_d_model)
            →  fusion_project  →  (B,prefix_num,llm_d_model)
                →  merge into input_ids embeddings at <|image_pad|> positions
                    →  LLM forward  →  loss
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import BaseModel
from src.models.ts_encoder import get_encoder
from src.models.ts_llm.itformer.fusion import ITFormerFusion

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class ITFormerModel(BaseModel):
    """Instruction-aware Time Series Transformer for time series QA.

    Uses a multi-variable encoder, an instruction-time fusion module
    with learnable prefix tokens, and token merging into a frozen LLM.
    """

    IMAGE_PAD_TOKEN = "<|image_pad|>"

    def __init__(
        self,
        device: str,
        llm_model: nn.Module,
        text_tokenizer: Any,
        ts_encoder: nn.Module,
        itformer_fusion: ITFormerFusion,
        ts_project: nn.Linear,
        query_project: nn.Linear,
        fusion_project: nn.Linear,
        freeze_ts_encoder: bool = True,
    ):
        super().__init__(device)
        self.llm_model = llm_model
        self.text_tokenizer = text_tokenizer
        self.ts_encoder = ts_encoder
        self.itformer_fusion = itformer_fusion
        self.ts_project = ts_project
        self.query_project = query_project
        self.fusion_project = fusion_project
        self.prefix_num = itformer_fusion.prefix_num
        self.freeze_ts_encoder = freeze_ts_encoder

        if freeze_ts_encoder:
            for p in self.ts_encoder.parameters():
                p.requires_grad = False

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "ITFormerModel":
        from src.backbones import get_backbone

        # --- Backbone ---
        backbone_cls = get_backbone(config.backbone.name)
        backbone = backbone_cls(model_id=config.backbone.model_id)

        text_tokenizer = backbone.load_tokenizer()
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        # Add placeholder token (Qwen has it natively; Llama etc. do not)
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": [cls.IMAGE_PAD_TOKEN]}
        )

        llm_model = backbone.load_model(device_map={"": device})
        llm_model.resize_token_embeddings(len(text_tokenizer))
        llm_d_model = llm_model.config.hidden_size

        # --- Encoder from registry ---
        enc_cfg = dict(config.model.encoder)
        encoder_type = enc_cfg.pop("type", "itformer_ts_encoder")
        encoder_cls = get_encoder(encoder_type)
        ts_encoder = encoder_cls(**enc_cfg).to(device)

        # Load pretrained encoder weights if specified
        extra = config.model.extra_kwargs
        ts_encoder_ckpt = extra.get("ts_encoder_checkpoint")
        if ts_encoder_ckpt and os.path.exists(ts_encoder_ckpt):
            state = torch.load(ts_encoder_ckpt, map_location="cpu")
            # Strip 'model.' prefix if present
            cleaned = {}
            for k, v in state.items():
                cleaned[k.removeprefix("model.")] = v
            msg = ts_encoder.load_state_dict(cleaned, strict=False)
            print(f"Loaded TS encoder weights: missing={len(msg.missing_keys)}, "
                  f"unexpected={len(msg.unexpected_keys)}")

        # --- ITFormer fusion params ---
        it_d_model = extra.get("it_d_model", 896)
        it_n_heads = extra.get("it_n_heads", 16)
        it_layers = extra.get("it_layers", 2)
        it_dropout = extra.get("it_dropout", 0.1)
        prefix_num = extra.get("prefix_num", 25)
        freeze_ts_encoder = extra.get("freeze_ts_encoder", True)

        fusion = ITFormerFusion(
            it_d_model=it_d_model,
            it_n_heads=it_n_heads,
            it_layers=it_layers,
            it_dropout=it_dropout,
            prefix_num=prefix_num,
        ).to(device)

        ts_project = nn.Linear(ts_encoder.output_dim, it_d_model).to(device)
        query_project = nn.Linear(llm_d_model, it_d_model).to(device)
        fusion_project = nn.Linear(it_d_model, llm_d_model).to(device)

        # Align all trainable components to the LLM's dtype (e.g. bfloat16)
        llm_dtype = next(llm_model.parameters()).dtype
        ts_encoder.to(llm_dtype)
        fusion.to(llm_dtype)
        ts_project.to(llm_dtype)
        query_project.to(llm_dtype)
        fusion_project.to(llm_dtype)

        return cls(
            device=device,
            llm_model=llm_model,
            text_tokenizer=text_tokenizer,
            ts_encoder=ts_encoder,
            itformer_fusion=fusion,
            ts_project=ts_project,
            query_project=query_project,
            fusion_project=fusion_project,
            freeze_ts_encoder=freeze_ts_encoder,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self.text_tokenizer

    # ------------------------------------------------------------------
    # Token merging
    # ------------------------------------------------------------------

    def _merge_features(
        self,
        fused_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Replace ``<|image_pad|>`` positions in the input embeddings with
        fused time-series features.

        Args:
            fused_features: (B, prefix_num, llm_d_model)
            input_ids: (B, seq_len)

        Returns:
            inputs_embeds: (B, seq_len, llm_d_model)
        """
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
        pad_id = self.text_tokenizer(self.IMAGE_PAD_TOKEN, add_special_tokens=False)["input_ids"][0]
        batch_idx, seq_idx = torch.where(input_ids == pad_id)

        B, P, D = fused_features.shape
        expected = B * P
        actual = len(batch_idx)
        if actual != expected:
            # Truncate or expand to match
            if actual > expected:
                batch_idx = batch_idx[:expected]
                seq_idx = seq_idx[:expected]
            else:
                # Pad with zeros if fewer positions than expected — shouldn't happen
                pass

        flat_features = fused_features.reshape(-1, D)[:len(batch_idx)]
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_idx, seq_idx] = flat_features.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------

    def _encode_and_fuse(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        query_ids: torch.Tensor,
        stage: list[int] | torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the full encoding pipeline and return inputs_embeds.

        Returns:
            inputs_embeds: (B, seq_len, llm_d_model)
        """
        # Encode time series
        if self.ts_encoder.multi_variable:
            ts_out = self.ts_encoder(time_series)  # (B, V, P, d_model)
        else:
            # Single-variable encoder: process each variable independently
            B, L, V = time_series.shape
            outs = []
            for v in range(V):
                out = self.ts_encoder(time_series[:, :, v])  # (B, P, d_model)
                outs.append(out)
            ts_out = torch.stack(outs, dim=1)  # (B, V, P, d_model)
        ts_proj = self.ts_project(ts_out)  # (B, V, P, it_d_model)

        # Encode query
        query_embeds = self.llm_model.get_input_embeddings()(query_ids)
        query_proj = self.query_project(query_embeds)  # (B, Q, it_d_model)

        # Fuse
        fused = self.itformer_fusion(query_proj, ts_proj, stage)  # (B, prefix_num, it_d_model)
        fused = self.fusion_project(fused)  # (B, prefix_num, llm_d_model)

        # Merge into LLM input
        return self._merge_features(fused, input_ids)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self.text_tokenizer
        pad_token = self.IMAGE_PAD_TOKEN

        # --- Stack time series: list of (L,) tensors per sample × V channels ---
        ts_list = []
        for item in batch:
            channels = item["time_series"]  # list of (L,) arrays or tensors
            tensors = [torch.as_tensor(ch, dtype=torch.float32) for ch in channels]
            ts_list.append(torch.stack(tensors, dim=-1))  # (L, V)
        # Pad to same length
        max_len = max(t.size(0) for t in ts_list)
        padded = []
        for t in ts_list:
            if t.size(0) < max_len:
                pad = torch.zeros(max_len - t.size(0), t.size(1), dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        llm_dtype = next(self.llm_model.parameters()).dtype
        time_series = torch.stack(padded).to(device=self.device, dtype=llm_dtype)  # (B, L, V)

        # --- Build prompts with <|image_pad|> placeholders ---
        text_inputs = []
        prompt_lengths = []
        query_texts = []
        stages = []

        for item in batch:
            # Build prompt text
            prompt = item["pre_prompt"]
            # Insert pad tokens for the fused features
            prompt += " " + (pad_token * self.prefix_num)
            if item["post_prompt"]:
                prompt += " " + item["post_prompt"]

            # Extract query (the question text for ITFormer's query_ids)
            question = item.get("question", item.get("post_prompt", ""))
            query_texts.append(question)

            # Extract stage (default to 1)
            stages.append(item.get("stage", 1))

            if not training:
                text_inputs.append(prompt)
                continue

            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))
            full_text = prompt + " " + item["answer"]
            text_inputs.append(full_text)

        # --- Tokenize query_ids ---
        query_tokenized = tokenizer(
            query_texts, padding="longest", return_tensors="pt", add_special_tokens=False
        )
        query_ids = query_tokenized.input_ids.to(self.device)

        # --- Tokenize main text ---
        original_padding = tokenizer.padding_side
        if not training:
            tokenizer.padding_side = "left"

        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        tokenizer.padding_side = original_padding

        stage_tensor = torch.tensor(stages, dtype=torch.long, device=self.device)

        if not training:
            return {
                "time_series": time_series,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None,
                "query_ids": query_ids,
                "stage": stage_tensor,
            }

        # --- Create labels ---
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
            "query_ids": query_ids,
            "stage": stage_tensor,
        }

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        query_ids = kwargs.get("query_ids")
        stage = kwargs.get("stage")

        if query_ids is None:
            raise ValueError("ITFormerModel.forward() requires 'query_ids' in kwargs.")

        inputs_embeds = self._encode_and_fuse(time_series, input_ids, query_ids, stage)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

        result: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Shift logits and labels for next-token prediction
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
        query_ids = generate_kwargs.pop("query_ids", None)
        stage = generate_kwargs.pop("stage", None)

        if query_ids is None:
            raise ValueError("ITFormerModel.generate() requires 'query_ids' in kwargs.")

        inputs_embeds = self._encode_and_fuse(time_series, input_ids, query_ids, stage)

        with torch.inference_mode():
            gen_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pad_token_id=self.text_tokenizer.pad_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                **generate_kwargs,
            )

        # Strip input prefix — return answer tokens only.
        # With inputs_embeds, some transformers versions return only new
        # tokens (shape < input length); others prepend the full input.
        input_len = input_ids.shape[1]
        if gen_ids.shape[1] > input_len:
            return gen_ids[:, input_len:]
        return gen_ids

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "itformer_fusion": list(self.itformer_fusion.parameters()),
            "ts_project": list(self.ts_project.parameters()),
            "query_project": list(self.query_project.parameters()),
            "fusion_project": list(self.fusion_project.parameters()),
        }
        # Include encoder if not frozen
        if not self.freeze_ts_encoder:
            groups["encoder"] = list(self.ts_encoder.parameters())
        return groups

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    def get_backbone_module(self) -> nn.Module:
        return self.llm_model
