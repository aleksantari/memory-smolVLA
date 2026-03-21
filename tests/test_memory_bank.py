"""Tests for ConsolidatingMemoryBank."""

import pytest
import torch

from memory_smolvla.memory.bank import ConsolidatingMemoryBank


@pytest.fixture
def bank():
    return ConsolidatingMemoryBank(max_size=4)


def make_tokens(seed: int, n_tokens: int = 8, d_model: int = 16) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_tokens, d_model)


# ------------------------------------------------------------------
# Basic write / read
# ------------------------------------------------------------------

def test_write_and_read(bank):
    tokens = make_tokens(0)
    bank.write(tokens, timestamp=0)
    memories, timestamps = bank.read_all()
    assert memories.shape == (1, 8, 16)
    assert timestamps.tolist() == [0]


def test_read_all_empty_raises(bank):
    with pytest.raises(RuntimeError, match="empty"):
        bank.read_all()


def test_write_detaches(bank):
    tokens = make_tokens(0, d_model=16)
    tokens.requires_grad_(True)
    bank.write(tokens, timestamp=0)
    _, _ = bank.read_all()
    # Stored tensor should be detached (no grad_fn)
    stored = bank._entries[0][1]
    assert not stored.requires_grad


def test_read_all_device(bank):
    bank.write(make_tokens(0), timestamp=0)
    memories, timestamps = bank.read_all(device="cpu")
    assert memories.device.type == "cpu"
    assert timestamps.device.type == "cpu"


# ------------------------------------------------------------------
# Max size and consolidation
# ------------------------------------------------------------------

def test_max_size_enforced(bank):
    for i in range(6):  # max_size=4, so two consolidations needed
        bank.write(make_tokens(i), timestamp=i)
    assert len(bank) <= bank.max_size


def test_consolidation_merges_two(bank):
    # Fill to max_size
    for i in range(4):
        bank.write(make_tokens(i), timestamp=i)
    assert len(bank) == 4
    # One more should trigger consolidation → still ≤ max_size
    bank.write(make_tokens(99), timestamp=99)
    assert len(bank) == bank.max_size


def test_consolidation_keeps_newer_timestamp():
    bank = ConsolidatingMemoryBank(max_size=2)
    bank.write(make_tokens(0), timestamp=0)
    bank.write(make_tokens(1), timestamp=1)
    # Add a third entry identical to entry[0] so they are most similar
    bank.write(make_tokens(0), timestamp=5)  # same tokens as ts=0, newer stamp
    assert len(bank) == 2
    timestamps = [ts for ts, _ in bank._entries]
    # The merged entry should keep the newer timestamp (5, not 0)
    assert 5 in timestamps


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

def test_reset_clears_bank(bank):
    bank.write(make_tokens(0), timestamp=0)
    bank.reset()
    assert len(bank) == 0


def test_read_after_reset_raises(bank):
    bank.write(make_tokens(0), timestamp=0)
    bank.reset()
    with pytest.raises(RuntimeError):
        bank.read_all()


# ------------------------------------------------------------------
# Multi-entry stacking
# ------------------------------------------------------------------

def test_read_all_stacks_correctly():
    bank = ConsolidatingMemoryBank(max_size=8)
    for i in range(3):
        bank.write(make_tokens(i), timestamp=i)
    memories, timestamps = bank.read_all()
    assert memories.shape == (3, 8, 16)
    assert timestamps.shape == (3,)
    assert sorted(timestamps.tolist()) == [0, 1, 2]
