"""Tests for FullSeqMemBank (spec §7.1–§7.5)."""

from __future__ import annotations

import pytest
import torch

from memory_smolvla.memory.full_seq_bank import FullSeqMemBank


L = 16
D = 64


def _bank(**kwargs) -> FullSeqMemBank:
    defaults = dict(
        token_size=D,
        mem_length=4,
        retrieval_layers=2,
        use_timestep_pe=True,
        consolidate_type="tome",
        dataloader_type="group",
        group_size=2,
    )
    defaults.update(kwargs)
    return FullSeqMemBank(**defaults)


def test_shape_and_bank_fills():
    """Spec §7.1: output shape matches input; bank accumulates entries."""
    bank = _bank(group_size=2)
    bank.eval()
    bank.reset()

    B = 2
    tokens = torch.randn(B, L, D)
    out = bank.process_batch(tokens, episode_ids=[0, 0], timesteps=[0, 1])

    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    assert bank.num_entries(0) == 2


def test_episode_isolation():
    """Spec §7.2: each episode has its own bank."""
    bank = _bank(dataloader_type="stream", group_size=1)
    bank.eval()
    bank.reset()

    bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[0])
    bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[1])
    assert bank.num_entries(0) == 2

    bank.process_batch(torch.randn(1, L, D), episode_ids=[1], timesteps=[0])
    assert bank.num_entries(1) == 1
    assert bank.num_entries(0) == 2


def test_consolidation_caps_bank():
    """Spec §7.3: ToMe keeps the bank at ``mem_length``."""
    bank = _bank(
        mem_length=3,
        consolidate_type="tome",
        dataloader_type="stream",
        group_size=1,
    )
    bank.eval()
    bank.reset()

    for t in range(5):
        bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[t])
    assert bank.num_entries(0) == 3


def test_fifo_consolidation():
    """FIFO variant caps the bank to ``mem_length`` and keeps the most recent."""
    bank = _bank(
        mem_length=3,
        consolidate_type="fifo",
        dataloader_type="stream",
        group_size=1,
    )
    bank.eval()
    bank.reset()

    for t in range(5):
        bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[t])
    assert bank.num_entries(0) == 3
    timesteps = [ts for ts, _ in bank.bank[0]]
    assert timesteps == [2, 3, 4]


def test_gradients_flow():
    """Spec §7.4: gradient flows to input tokens and to trainable params."""
    bank = _bank(mem_length=4, group_size=2)
    bank.train()
    bank.reset()

    tokens = torch.randn(2, L, D, requires_grad=True)
    out = bank.process_batch(tokens, episode_ids=[0, 0], timesteps=[0, 1])
    loss = out.sum()
    loss.backward()

    assert tokens.grad is not None
    assert tokens.grad.abs().sum() > 0

    trainable = [(n, p) for n, p in bank.named_parameters() if p.requires_grad]
    assert len(trainable) > 0
    for name, p in trainable:
        # Timestep encoder only receives gradient when the bank is non-empty
        # during retrieval. In a 2-item batch the first item hits an empty
        # bank (retrieved = current), so timestep_encoder sees no gradient
        # from item 0; it DOES see one from item 1. So the assertion still
        # holds for every trainable parameter.
        assert p.grad is not None, f"{name} has no grad"


def test_cold_bank_passthrough():
    """Spec §7.5: on an empty bank, output equals input exactly.

    Because ``retrieved = working_mem`` when the bank is empty, and
    ``gate_fusion(x, x) = scale * x + (1 - scale) * x = x`` for any scale.
    """
    bank = _bank(group_size=1, dataloader_type="stream")
    bank.eval()
    bank.reset()

    tokens = torch.randn(1, L, D)
    out = bank.process_batch(tokens, episode_ids=[0], timesteps=[0])

    assert torch.allclose(out, tokens, atol=1e-5), (
        "Cold-bank pass-through failed: output differs from input on empty bank"
    )


def test_group_mode_clears_bank_between_batches():
    """In group mode, ``process_batch`` clears the bank at the start of each call."""
    bank = _bank(dataloader_type="group", group_size=2, mem_length=4)
    bank.train()
    bank.reset()

    bank.process_batch(torch.randn(2, L, D), episode_ids=[0, 0], timesteps=[0, 1])
    assert bank.num_entries(0) == 2

    # New batch — bank should be cleared at the top of process_batch
    bank.process_batch(torch.randn(2, L, D), episode_ids=[0, 0], timesteps=[2, 3])
    assert bank.num_entries(0) == 2, (
        "Group-mode bank should be reset at each batch start, not carry over."
    )


def test_gate_scale_tracked():
    """``last_gate_scale()`` returns per-token gate values after process_batch."""
    bank = _bank(group_size=1, dataloader_type="stream")
    bank.eval()
    bank.reset()

    tokens = torch.randn(1, L, D)
    bank.process_batch(tokens, episode_ids=[0], timesteps=[0])
    scale = bank.last_gate_scale()
    assert scale is not None
    assert scale.shape == (1, L, D)
    assert (scale >= 0).all() and (scale <= 1).all()


def test_invalid_inputs_raise():
    bank = _bank()
    with pytest.raises(ValueError):
        bank.process_batch(torch.randn(2, L, D), episode_ids=[0], timesteps=[0, 1])
    with pytest.raises(ValueError):
        FullSeqMemBank(token_size=D, consolidate_type="nope")
    with pytest.raises(ValueError):
        FullSeqMemBank(token_size=D, dataloader_type="nope")
