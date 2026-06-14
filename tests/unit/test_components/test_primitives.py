# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shape contract tests for all architectural primitives in src.components."""

import torch
import torch.nn as nn

from src.components.alignment import AlignmentProjector
from src.components.cross_attention import SteerableCrossAttention
from src.components.drop_path import DropPath
from src.components.fusion_layer import FusionLayerWrapper
from src.components.gated_ff import GatedFeedForward
from src.components.multi_scale_pooling import MultiScalePooling
from src.components.positional_encoding import (
    LearnablePositionalEmbedding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from src.components.ssm import AttentionPooling, MemoryBank, SelectiveSSMBlock


class TestSteerableCrossAttention:
    def test_output_shape(self):
        attn = SteerableCrossAttention(dim=64, num_heads=4, dim_head=16)
        x = torch.randn(2, 10, 64)
        context = torch.randn(2, 5, 64)
        out = attn(x, context)
        assert out.shape == (2, 10, 64)

    def test_gate_starts_at_zero(self):
        attn = SteerableCrossAttention(dim=64)
        assert attn.gate.item() == 0.0

    def test_zero_gate_passthrough(self):
        """With gate=0, output should equal input (tanh(0)=0)."""
        attn = SteerableCrossAttention(dim=64, num_heads=4, dim_head=16)
        x = torch.randn(2, 10, 64)
        context = torch.randn(2, 5, 64)
        with torch.no_grad():
            out = attn(x, context)
        torch.testing.assert_close(out, x)


class TestAlignmentProjector:
    def test_output_shape(self):
        proj = AlignmentProjector(enc_dim=128, llm_dim=256)
        x = torch.randn(2, 10, 128)
        out = proj(x)
        assert out.shape == (2, 10, 256)


class TestMultiScalePooling:
    def test_concat_mode(self):
        pool = MultiScalePooling(dim=64, scales=[1, 4])
        x = torch.randn(2, 16, 64)
        out = pool(x)
        assert isinstance(out, torch.Tensor)
        # scale=1: 16 tokens, scale=4: ceil(16/4)=4 tokens -> 20 total
        assert out.shape == (2, 20, 64)

    def test_list_mode(self):
        pool = MultiScalePooling(dim=64, scales=[1, 4], return_list=True)
        x = torch.randn(2, 16, 64)
        out = pool(x)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape == (2, 16, 64)
        assert out[1].shape == (2, 4, 64)


class TestFusionLayerWrapper:
    def _make_dummy_decoder_layer(self, dim):
        """Simple linear layer that returns a tuple like HF decoder layers."""

        class DummyDecoder(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear = nn.Linear(d, d)

            def forward(self, hidden_states, **kwargs):
                return (self.linear(hidden_states),)

        return DummyDecoder(dim)

    def test_without_ff(self):
        dim = 64
        decoder = self._make_dummy_decoder_layer(dim)
        cross_attn = SteerableCrossAttention(dim=dim, num_heads=4, dim_head=16)
        wrapper = FusionLayerWrapper(decoder, cross_attn)

        x = torch.randn(2, 10, dim)
        wrapper.ts_features = torch.randn(2, 5, dim)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (2, 10, dim)

    def test_with_ff(self):
        dim = 64
        decoder = self._make_dummy_decoder_layer(dim)
        cross_attn = SteerableCrossAttention(dim=dim, num_heads=4, dim_head=16)
        ff = GatedFeedForward(dim=dim, mult=4)
        wrapper = FusionLayerWrapper(decoder, cross_attn, ff=ff)

        x = torch.randn(2, 10, dim)
        wrapper.ts_features = torch.randn(2, 5, dim)
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (2, 10, dim)

    def test_no_ts_features(self):
        """Without conditioning, output should be decoder-only."""
        dim = 64
        decoder = self._make_dummy_decoder_layer(dim)
        cross_attn = SteerableCrossAttention(dim=dim, num_heads=4, dim_head=16)
        wrapper = FusionLayerWrapper(decoder, cross_attn)

        x = torch.randn(2, 10, dim)
        wrapper.ts_features = None
        out = wrapper(x)
        assert isinstance(out, tuple)
        assert out[0].shape == (2, 10, dim)


class TestSelectiveSSMBlock:
    def test_output_shape(self):
        ssm = SelectiveSSMBlock(dim=64, d_state=16, d_conv=4, expand=2)
        x = torch.randn(2, 16, 64)
        out = ssm(x)
        assert out.shape == (2, 16, 64)


class TestAttentionPooling:
    def test_output_shape(self):
        pool = AttentionPooling(dim=64, num_queries=8, num_heads=4, dim_head=16)
        x = torch.randn(2, 16, 64)
        out = pool(x)
        assert out.shape == (2, 8, 64)


class TestMemoryBank:
    def test_output_shape(self):
        bank = MemoryBank(
            enc_dim=64,
            num_tokens_per_scale=4,
            scales=[1, 4],
            d_state=8,
            d_conv=4,
            expand=2,
            pool_heads=4,
            pool_dim_head=16,
        )
        # Input: list of per-scale tensors
        scale_features = [
            torch.randn(2, 16, 64),  # scale=1
            torch.randn(2, 4, 64),  # scale=4
        ]
        out = bank(scale_features)
        # 2 scales * 4 tokens per scale = 8 tokens
        assert out.shape == (2, 8, 64)


class TestGatedFeedForward:
    def test_output_shape(self):
        ff = GatedFeedForward(dim=64, mult=4)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == (2, 10, 64)

    def test_gate_starts_at_zero(self):
        ff = GatedFeedForward(dim=64)
        assert ff.gate.item() == 0.0


class TestDropPath:
    def test_passthrough_zero_prob(self):
        dp = DropPath(drop_prob=0.0)
        x = torch.randn(2, 10, 64)
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_passthrough_eval_mode(self):
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(2, 10, 64)
        out = dp(x)
        torch.testing.assert_close(out, x)

    def test_training_mode_shape(self):
        dp = DropPath(drop_prob=0.5)
        dp.train()
        x = torch.randn(2, 10, 64)
        out = dp(x)
        assert out.shape == x.shape


class TestPositionalEncodings:
    def test_sinusoidal_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape[-1] == 64
        assert out.shape[-2] == 10

    def test_rotary_shape(self):
        pe = RotaryPositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_learnable_shape(self):
        pe = LearnablePositionalEmbedding(d_model=64)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape[-1] == 64
        assert out.shape[-2] == 10
