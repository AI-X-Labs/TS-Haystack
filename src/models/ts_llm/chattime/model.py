"""ChatTime model: time series as text via discretization + serialization.

Implements the ChatTime architecture (Wang et al., AAAI 2025) within the
TSLM-Bench framework. Unlike encoder-based models (Flamingo, ITFormer),
ChatTime has **no neural encoder or projector**. Instead it:

1. Discretizes continuous time series values into ~10K bins
2. Serializes bin centers as delimited text tokens (``###0.1234###``)
3. Feeds the serialized text directly into a standard causal LLM

The LLM vocabulary is extended with the discretized bin tokens during
model construction, so the tokenizer can represent each bin value as a
single token.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from src.backbones import get_backbone
from src.backbones.base import BaseBackbone
from src.models.base import BaseModel
from src.models.ts_llm.chattime.discretizer import Discretizer
from src.models.ts_llm.chattime.serializer import Serializer

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


# -- Prompt templates (Alpaca-style, matching original ChatTime) -----------

_SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

_TEMPLATE = """{system}

### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""


class ChatTimeModel(BaseModel):
    """ChatTime: a pure-LLM TSLM that encodes time series as text.

    The model discretizes each time series channel, serializes the discrete
    values into ``###value###`` tokens, builds an Alpaca-style prompt, and
    forwards everything through a standard causal LLM.

    Config keys (under ``model.extra_kwargs``):

    - ``n_tokens`` (int): Number of discretization bins. Default 10002.
    - ``precision`` (int): Decimal precision for serialized values. Default 4.
    - ``train_embeddings`` (bool): If True, ``embed_tokens`` and ``lm_head``
      are included in trainable parameters (for the pretrain phase where
      the extended vocabulary needs to be learned). Default False.
    - ``max_seq_length`` (int): Maximum sequence length for tokenization.
      Default 2048.
    """

    def __init__(
        self,
        device: str,
        backbone: BaseBackbone,
        llm: nn.Module,
        tokenizer: Any,
        discretizer: Discretizer,
        serializer: Serializer,
        *,
        train_embeddings: bool = False,
        max_seq_length: int = 2048,
    ):
        super().__init__(device)
        self.backbone = backbone
        self.llm = llm
        self.tokenizer = tokenizer
        self.discretizer = discretizer
        self.serializer = serializer
        self.train_embeddings = train_embeddings
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "ChatTimeModel":
        extra = config.model.extra_kwargs or {}
        n_tokens = extra.get("n_tokens", 10002)
        precision = extra.get("precision", 4)
        train_embeddings = extra.get("train_embeddings", False)
        max_seq_length = extra.get("max_seq_length", 2048)

        # --- backbone --------------------------------------------------
        backbone = get_backbone(config.backbone.name)(
            model_id=config.backbone.model_id,
        )

        # --- tokenizer -------------------------------------------------
        tokenizer = backbone.load_tokenizer()

        # --- extend vocabulary with discretized bin tokens -------------
        discretizer = Discretizer(n_tokens=n_tokens)
        serializer = Serializer(prec=precision)

        # Build vocabulary: one token per bin center + NaN token
        bin_centers = discretizer.get_centers()[1:-1]  # strip padding entries
        vocab_tokens = [serializer.serialize(np.array([c])) for c in bin_centers]
        vocab_tokens.append(f"{serializer.time_flag}{serializer.nan_flag}{serializer.time_flag}")

        num_added = tokenizer.add_tokens(vocab_tokens)
        print(f"  ChatTime: extended tokenizer by {num_added} discretized tokens "
              f"(total vocab: {len(tokenizer)})")

        # --- load LLM and resize embeddings ----------------------------
        llm = backbone.load_model(device_map={"": device})
        llm.resize_token_embeddings(len(tokenizer))

        return cls(
            device=device,
            backbone=backbone,
            llm=llm,
            tokenizer=tokenizer,
            discretizer=discretizer,
            serializer=serializer,
            train_embeddings=train_embeddings,
            max_seq_length=max_seq_length,
        )

    # ------------------------------------------------------------------
    # prepare_batch — discretize + serialize time series into text
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        batch: list[dict[str, Any]],
        training: bool = True,
        normalize: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokenizer = self.tokenizer

        text_inputs: list[str] = []
        prompt_lengths: list[int] = []

        for sample in batch:
            # -- serialize each time series channel to text --
            ts_list: list = sample["time_series"]        # list[Tensor], one per channel
            ts_labels: list[str] = sample["time_series_text"]  # label per channel

            serialized_channels: list[str] = []
            for ts_tensor, label in zip(ts_list, ts_labels):
                ts_np = ts_tensor.numpy() if isinstance(ts_tensor, torch.Tensor) else np.asarray(ts_tensor)
                discretized = self.discretizer.discretize(ts_np.astype(np.float64))
                text = self.serializer.serialize(discretized)
                serialized_channels.append(f"{label}: {text}")

            ts_text_block = "\n".join(serialized_channels)

            # -- build prompt --
            instruction = sample.get("post_prompt", "")
            input_block = ts_text_block
            if sample.get("pre_prompt"):
                input_block = sample["pre_prompt"] + "\n" + ts_text_block

            prompt_text = _TEMPLATE.format(
                system=_SYSTEM_PROMPT,
                instruction=instruction,
                input=input_block,
                response="",
            )

            if not training:
                text_inputs.append(prompt_text)
                continue

            # Training: measure prompt length, then append answer
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))

            full_text = prompt_text + sample["answer"]
            text_inputs.append(full_text)

        # -- tokenize ---------------------------------------------------
        original_padding_side = tokenizer.padding_side
        if not training:
            tokenizer.padding_side = "left"

        tokenized = tokenizer(
            text_inputs,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        tokenizer.padding_side = original_padding_side

        # Dummy time_series tensor (required by interface, unused in forward)
        dummy_ts = torch.zeros(len(batch), 1, 1, device=self.device)

        if not training:
            return {
                "time_series": dummy_ts,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": None,
            }

        # -- create labels with -100 masking for prompt tokens ----------
        labels = torch.full_like(input_ids, -100)
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_positions = non_padding[non_padding >= prompt_length]
            if len(answer_positions) > 0:
                labels[i, answer_positions] = input_ids[i, answer_positions]

        return {
            "time_series": dummy_ts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ------------------------------------------------------------------
    # forward / generate
    # ------------------------------------------------------------------

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Standard causal LM forward. ``time_series`` is unused (already in text)."""
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate answer tokens. ``time_series`` is unused."""
        generate_kwargs.setdefault("do_sample", False)

        output_ids = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        # Strip the input prompt, return only generated tokens
        return output_ids[:, input_ids.shape[1]:]

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def get_trainable_parameters(self) -> dict[str, list[nn.Parameter]]:
        """Return parameter groups for the optimizer.

        When ``train_embeddings`` is True (pretrain phase), ``embed_tokens``
        and ``lm_head`` are included so the extended vocabulary embeddings
        can be learned. Otherwise returns empty (only LoRA / full mode
        controls what's trainable).
        """
        if self.train_embeddings:
            embed_params = list(self.llm.model.embed_tokens.parameters())
            lm_head_params = list(self.llm.lm_head.parameters())
            return {"embeddings": embed_params + lm_head_params}
        return {}

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def get_backbone_module(self) -> nn.Module:
        """Return the LLM module for framework-managed LoRA injection."""
        return self.llm
