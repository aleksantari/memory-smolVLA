"""Tests for MemoryCompressor (learned write head)."""

import pytest
import torch

from memory_smolvla.memory.compressor import MemoryCompressor


class TestMemoryCompressor:
    """Tests for Perceiver-style memory compression."""

    @pytest.fixture
    def compressor(self):
        return MemoryCompressor(d_model=64, n_slots=4, n_heads=4)

    def test_output_shape(self, compressor):
        prefix = torch.randn(1, 100, 64)
        out = compressor(prefix)
        assert out.shape == (1, 4, 64)

    def test_batched(self, compressor):
        prefix = torch.randn(2, 100, 64)
        out = compressor(prefix)
        assert out.shape == (2, 4, 64)

    def test_gradients_flow_to_params(self, compressor):
        """At least some compressor parameters receive gradients."""
        prefix = torch.randn(1, 50, 64)
        out = compressor(prefix)
        out.sum().backward()
        grads = [
            p.grad for p in compressor.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(grads) > 0, "No compressor parameters received gradients"

    def test_slot_queries_are_learnable(self, compressor):
        assert compressor.slot_queries.requires_grad

    def test_different_input_lengths(self, compressor):
        """Output slots are always n_slots regardless of input length."""
        for L in [10, 50, 200]:
            prefix = torch.randn(1, L, 64)
            out = compressor(prefix)
            assert out.shape == (1, 4, 64)

    def test_output_is_finite(self, compressor):
        prefix = torch.randn(1, 50, 64)
        out = compressor(prefix)
        assert torch.isfinite(out).all()
