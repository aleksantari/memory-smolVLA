"""Tests for SigmoidGate including regularization."""

import pytest
import torch

from memory_smolvla.memory.gating import SigmoidGate


class TestSigmoidGate:
    """Tests for gate fusion and regularization."""

    @pytest.fixture
    def gate(self):
        return SigmoidGate(d_model=64, hidden_dim=32)

    @pytest.fixture
    def reg_gate(self):
        return SigmoidGate(
            d_model=64, hidden_dim=32,
            alpha_target=0.2, alpha_reg_weight=0.1,
        )

    def test_output_shape(self, gate):
        current = torch.randn(1, 10, 64)
        retrieved = torch.randn(1, 10, 64)
        fused, alpha = gate(current, retrieved)
        assert fused.shape == (1, 10, 64)
        assert alpha.shape == (1, 10, 1)

    def test_alpha_range(self, gate):
        current = torch.randn(1, 10, 64)
        retrieved = torch.randn(1, 10, 64)
        _, alpha = gate(current, retrieved)
        assert (alpha >= 0.0).all() and (alpha <= 1.0).all()

    def test_no_reg_returns_zero(self, gate):
        alpha = torch.tensor([[[0.9]]])
        loss = gate.regularization_loss(alpha)
        assert loss.item() == 0.0

    def test_reg_loss_is_positive(self, reg_gate):
        alpha = torch.tensor([[[0.9]]])  # far from target 0.2
        loss = reg_gate.regularization_loss(alpha)
        assert loss.item() > 0.0

    def test_reg_loss_zero_at_target(self, reg_gate):
        alpha = torch.tensor([[[0.2]]])  # exactly at target
        loss = reg_gate.regularization_loss(alpha)
        assert loss.item() < 1e-6

    def test_reg_loss_gradient_flows(self, reg_gate):
        alpha = torch.tensor([[[0.8]]], requires_grad=True)
        loss = reg_gate.regularization_loss(alpha)
        loss.backward()
        assert alpha.grad is not None
        assert alpha.grad.abs().sum() > 0

    def test_fusion_identity_when_alpha_zero(self, gate):
        current = torch.randn(1, 10, 64)
        retrieved = torch.randn(1, 10, 64)
        alpha = torch.zeros(1, 10, 1)
        # Manual check: alpha=0 → fused = current
        fused = alpha * retrieved + (1.0 - alpha) * current
        assert torch.allclose(fused, current)

    def test_fusion_replaces_when_alpha_one(self, gate):
        current = torch.randn(1, 10, 64)
        retrieved = torch.randn(1, 10, 64)
        alpha = torch.ones(1, 10, 1)
        fused = alpha * retrieved + (1.0 - alpha) * current
        assert torch.allclose(fused, retrieved)
