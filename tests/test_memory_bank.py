"""Tests for MemoryBank (with FIFO and consolidation eviction)."""

import pytest
import torch

from memory_smolvla.memory.bank import ConsolidatingMemoryBank, MemoryBank


@pytest.fixture
def fifo_bank():
    return MemoryBank(max_size=4, eviction="fifo")


@pytest.fixture
def consolidate_bank():
    return MemoryBank(max_size=4, eviction="consolidate")


def make_tokens(seed: int, n_tokens: int = 8, d_model: int = 16) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n_tokens, d_model)


# ------------------------------------------------------------------
# Basic write / read (shared by both strategies)
# ------------------------------------------------------------------

def test_write_and_read(fifo_bank):
    tokens = make_tokens(0)
    fifo_bank.write(tokens, timestamp=0)
    memories, timestamps = fifo_bank.read_all()
    assert memories.shape == (1, 8, 16)
    assert timestamps.tolist() == [0]


def test_read_all_empty_raises(fifo_bank):
    with pytest.raises(RuntimeError, match="empty"):
        fifo_bank.read_all()


def test_write_detaches(fifo_bank):
    tokens = make_tokens(0, d_model=16)
    tokens.requires_grad_(True)
    fifo_bank.write(tokens, timestamp=0)
    stored = fifo_bank._entries[0][1]
    assert not stored.requires_grad


def test_read_all_device(fifo_bank):
    fifo_bank.write(make_tokens(0), timestamp=0)
    memories, timestamps = fifo_bank.read_all(device="cpu")
    assert memories.device.type == "cpu"
    assert timestamps.device.type == "cpu"


# ------------------------------------------------------------------
# FIFO eviction
# ------------------------------------------------------------------

def test_fifo_max_size_enforced(fifo_bank):
    for i in range(6):
        fifo_bank.write(make_tokens(i), timestamp=i)
    assert len(fifo_bank) == fifo_bank.max_size


def test_fifo_drops_oldest(fifo_bank):
    for i in range(6):
        fifo_bank.write(make_tokens(i), timestamp=i)
    # Oldest entries (0, 1) should be gone; newest (2,3,4,5) remain
    timestamps = [ts for ts, _ in fifo_bank._entries]
    assert 0 not in timestamps
    assert 1 not in timestamps
    assert 5 in timestamps


def test_fifo_preserves_temporal_order(fifo_bank):
    for i in range(10):
        fifo_bank.write(make_tokens(i), timestamp=i)
    timestamps = [ts for ts, _ in fifo_bank._entries]
    assert timestamps == sorted(timestamps)


# ------------------------------------------------------------------
# Consolidation eviction (backward compat)
# ------------------------------------------------------------------

def test_consolidation_max_size_enforced(consolidate_bank):
    for i in range(6):
        consolidate_bank.write(make_tokens(i), timestamp=i)
    assert len(consolidate_bank) <= consolidate_bank.max_size


def test_consolidation_merges_two(consolidate_bank):
    for i in range(4):
        consolidate_bank.write(make_tokens(i), timestamp=i)
    assert len(consolidate_bank) == 4
    consolidate_bank.write(make_tokens(99), timestamp=99)
    assert len(consolidate_bank) == consolidate_bank.max_size


def test_consolidation_keeps_newer_timestamp():
    bank = MemoryBank(max_size=2, eviction="consolidate")
    bank.write(make_tokens(0), timestamp=0)
    bank.write(make_tokens(1), timestamp=1)
    bank.write(make_tokens(0), timestamp=5)
    assert len(bank) == 2
    timestamps = [ts for ts, _ in bank._entries]
    assert 5 in timestamps


# ------------------------------------------------------------------
# Invalid eviction strategy
# ------------------------------------------------------------------

def test_invalid_eviction_raises():
    with pytest.raises(ValueError, match="Invalid eviction"):
        MemoryBank(max_size=4, eviction="random")


# ------------------------------------------------------------------
# Backward compat alias
# ------------------------------------------------------------------

def test_backward_compat_alias():
    bank = ConsolidatingMemoryBank(max_size=4)
    assert isinstance(bank, MemoryBank)


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

def test_reset_clears_bank(fifo_bank):
    fifo_bank.write(make_tokens(0), timestamp=0)
    fifo_bank.reset()
    assert len(fifo_bank) == 0


def test_read_after_reset_raises(fifo_bank):
    fifo_bank.write(make_tokens(0), timestamp=0)
    fifo_bank.reset()
    with pytest.raises(RuntimeError):
        fifo_bank.read_all()


# ------------------------------------------------------------------
# Multi-entry stacking
# ------------------------------------------------------------------

def test_read_all_stacks_correctly():
    bank = MemoryBank(max_size=8, eviction="fifo")
    for i in range(3):
        bank.write(make_tokens(i), timestamp=i)
    memories, timestamps = bank.read_all()
    assert memories.shape == (3, 8, 16)
    assert timestamps.shape == (3,)
    assert sorted(timestamps.tolist()) == [0, 1, 2]
