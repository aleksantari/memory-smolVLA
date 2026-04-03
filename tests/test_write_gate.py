"""Tests for WriteGate (selective memory writing)."""

import pytest
import torch

from memory_smolvla.memory.write_gate import WriteGate


class TestWriteGate:
    """Tests for the learned write gate."""

    @pytest.fixture
    def gate(self):
        return WriteGate(d_model=64, threshold=0.5)

    def test_output_shape(self, gate):
        current = torch.randn(1, 10, 64)
        recent = torch.randn(1, 64)
        prob = gate(current, recent)
        assert prob.shape == (1, 1)

    def test_output_range(self, gate):
        current = torch.randn(1, 10, 64)
        recent = torch.randn(1, 64)
        prob = gate(current, recent)
        assert (prob >= 0.0).all() and (prob <= 1.0).all()

    def test_none_recent_gives_one(self, gate):
        current = torch.randn(1, 10, 64)
        prob = gate(current, None)
        assert prob.item() == 1.0

    def test_should_write_returns_bool(self, gate):
        current = torch.randn(1, 10, 64)
        should, prob = gate.should_write(current, None)
        assert isinstance(should, bool)
        assert should is True  # None recent → always write

    def test_gradients_flow(self, gate):
        current = torch.randn(1, 10, 64, requires_grad=True)
        recent = torch.randn(1, 64)
        prob = gate(current, recent)
        prob.sum().backward()
        assert current.grad is not None

    def test_batched(self, gate):
        current = torch.randn(3, 10, 64)
        recent = torch.randn(3, 64)
        prob = gate(current, recent)
        assert prob.shape == (3, 1)
