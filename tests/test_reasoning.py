"""Tests for the V8 reasoning heads."""

from __future__ import annotations

import pytest
import torch

from memory_smolvla.memory.reasoning import FutureStatePredictor, ReasoningSummaryHead


@pytest.mark.parametrize(
    ("B", "L", "D", "n_slots", "n_heads"),
    [
        (1, 5, 16, 8, 4),
        (2, 12, 32, 4, 4),
        (3, 7, 24, 2, 8),
    ],
)
def test_summary_head_output_shape(B: int, L: int, D: int, n_slots: int, n_heads: int) -> None:
    head = ReasoningSummaryHead(dim=D, n_slots=n_slots, n_heads=n_heads)
    out = head(torch.randn(B, L, D))
    assert out.shape == (B, n_slots, D)


@pytest.mark.parametrize(
    ("B", "n_slots", "D", "state_dim"),
    [
        (1, 8, 16, 6),
        (4, 3, 32, 14),
    ],
)
def test_future_state_predictor_output_shape(
    B: int, n_slots: int, D: int, state_dim: int
) -> None:
    predictor = FutureStatePredictor(dim=D, state_dim=state_dim, hidden=32)
    out = predictor(torch.randn(B, n_slots, D))
    assert out.shape == (B, state_dim)


def test_backward_populates_summary_head_grads() -> None:
    torch.manual_seed(0)
    head = ReasoningSummaryHead(dim=16, n_slots=4, n_heads=4)
    predictor = FutureStatePredictor(dim=16, state_dim=6, hidden=32)

    prefix = torch.randn(2, 9, 16)
    target = torch.randn(2, 6)

    tokens = head(prefix)
    pred = predictor(tokens)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
    # At least one parameter should receive a non-zero gradient.
    assert any(p.grad.abs().sum() > 0 for p in head.parameters())


def test_dim_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ReasoningSummaryHead(dim=10, n_slots=4, n_heads=4)
    # Divisible dims construct fine.
    ReasoningSummaryHead(dim=12, n_slots=4, n_heads=4)
