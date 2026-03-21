"""Tests for CrossAttentionRetrieval and TemporalPositionalEncoding."""

import torch

from memory_smolvla.memory.retrieval import CrossAttentionRetrieval
from memory_smolvla.memory.temporal_pe import TemporalPositionalEncoding


# ------------------------------------------------------------------
# TemporalPositionalEncoding
# ------------------------------------------------------------------

class TestTemporalPE:
    def setup_method(self):
        self.pe = TemporalPositionalEncoding(d_model=16)

    def test_output_shape_1d(self):
        time_deltas = torch.tensor([0.0, 1.0, 5.0, 10.0])
        out = self.pe(time_deltas)
        assert out.shape == (4, 16)

    def test_output_shape_2d(self):
        time_deltas = torch.zeros(2, 8)
        out = self.pe(time_deltas)
        assert out.shape == (2, 8, 16)

    def test_zero_delta_gives_finite(self):
        time_deltas = torch.zeros(3)
        out = self.pe(time_deltas)
        assert torch.isfinite(out).all()

    def test_different_deltas_give_different_encodings(self):
        pe_0 = self.pe(torch.tensor([0.0]))
        pe_10 = self.pe(torch.tensor([10.0]))
        assert not torch.allclose(pe_0, pe_10)

    def test_no_trainable_params(self):
        trainable = [p for p in self.pe.parameters() if p.requires_grad]
        assert len(trainable) == 0


# ------------------------------------------------------------------
# CrossAttentionRetrieval
# ------------------------------------------------------------------

class TestCrossAttentionRetrieval:
    def setup_method(self):
        self.B, self.L, self.D = 1, 10, 32
        self.M = 24  # K * N_tok
        self.retrieval = CrossAttentionRetrieval(d_model=self.D, n_heads=2)

    def test_output_shape(self):
        current = torch.randn(self.B, self.L, self.D)
        memory = torch.randn(self.B, self.M, self.D)
        time_deltas = torch.zeros(self.B, self.M)
        out = self.retrieval(current, memory, time_deltas)
        assert out.shape == (self.B, self.L, self.D)

    def test_gradient_flows(self):
        current = torch.randn(self.B, self.L, self.D, requires_grad=True)
        memory = torch.randn(self.B, self.M, self.D)
        time_deltas = torch.zeros(self.B, self.M)
        out = self.retrieval(current, memory, time_deltas)
        loss = out.sum()
        loss.backward()
        assert current.grad is not None
        assert torch.isfinite(current.grad).all()

    def test_gradient_flows_through_retrieval_params(self):
        current = torch.randn(self.B, self.L, self.D)
        memory = torch.randn(self.B, self.M, self.D)
        time_deltas = torch.zeros(self.B, self.M)
        out = self.retrieval(current, memory, time_deltas)
        out.sum().backward()
        for name, param in self.retrieval.named_parameters():
            assert param.grad is not None, f"No grad for {name}"

    def test_output_is_finite(self):
        current = torch.randn(self.B, self.L, self.D)
        memory = torch.randn(self.B, self.M, self.D)
        time_deltas = torch.arange(self.M, dtype=torch.float).unsqueeze(0)
        out = self.retrieval(current, memory, time_deltas)
        assert torch.isfinite(out).all()

    def test_single_memory_token(self):
        current = torch.randn(self.B, self.L, self.D)
        memory = torch.randn(self.B, 1, self.D)
        time_deltas = torch.zeros(self.B, 1)
        out = self.retrieval(current, memory, time_deltas)
        assert out.shape == (self.B, self.L, self.D)
