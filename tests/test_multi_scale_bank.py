"""Tests for MultiScaleMemoryBank."""

import pytest
import torch

from memory_smolvla.memory.multi_scale_bank import MultiScaleMemoryBank


class TestMultiScaleMemoryBank:
    """Tests for multi-scale memory storage and retrieval."""

    @pytest.fixture
    def bank(self):
        return MultiScaleMemoryBank(
            short_size=2,
            medium_size=4,
            long_size=8,
            medium_pool_factor=4,
            long_pool_factor=8,
        )

    def test_write_populates_all_scales(self, bank):
        tokens = torch.randn(32, 64)
        bank.write(tokens, timestamp=0)

        assert len(bank.short) == 1
        assert len(bank.medium) == 1
        assert len(bank.long) == 1

    def test_short_stores_full_resolution(self, bank):
        tokens = torch.randn(32, 64)
        bank.write(tokens, timestamp=0)

        memories, _ = bank.short.read_all()
        assert memories.shape == (1, 32, 64)

    def test_medium_compresses_tokens(self, bank):
        tokens = torch.randn(32, 64)
        bank.write(tokens, timestamp=0)

        memories, _ = bank.medium.read_all()
        # 32 tokens / pool_factor 4 = 8 pooled tokens
        assert memories.shape == (1, 8, 64)

    def test_long_compresses_more(self, bank):
        tokens = torch.randn(32, 64)
        bank.write(tokens, timestamp=0)

        memories, _ = bank.long.read_all()
        # 32 tokens / pool_factor 8 = 4 pooled tokens
        assert memories.shape == (1, 4, 64)

    def test_read_all_concatenates_scales(self, bank):
        tokens = torch.randn(32, 64)
        bank.write(tokens, timestamp=0)

        memories, timestamps = bank.read_all()
        # Should have 1 entry from each scale, padded to max token dim (32)
        assert memories.shape[0] == 3  # one per scale
        assert memories.shape[2] == 64  # d_model preserved

    def test_fifo_eviction_per_scale(self, bank):
        # Write more than short_size entries
        for t in range(5):
            bank.write(torch.randn(32, 64), timestamp=t)

        # Short (size 2): only last 2
        assert len(bank.short) == 2
        # Medium (size 4): only last 4
        assert len(bank.medium) == 4
        # Long (size 8): all 5
        assert len(bank.long) == 5

    def test_reset_clears_all(self, bank):
        bank.write(torch.randn(32, 64), timestamp=0)
        bank.reset()

        assert len(bank.short) == 0
        assert len(bank.medium) == 0
        assert len(bank.long) == 0

    def test_len_is_total(self, bank):
        bank.write(torch.randn(32, 64), timestamp=0)
        assert len(bank) == 3  # one entry per scale

    def test_empty_read_raises(self, bank):
        with pytest.raises(RuntimeError):
            bank.read_all()

    def test_pool_tokens_handles_remainder(self):
        """Pool factor that doesn't evenly divide token count."""
        tokens = torch.randn(10, 64)
        # pool_factor=3: 3 full groups + 1 remainder
        pooled = MultiScaleMemoryBank._pool_tokens(tokens, 3)
        assert pooled.shape == (4, 64)

    def test_pool_tokens_factor_larger_than_input(self):
        """Pool factor >= N_tokens should collapse to single token."""
        tokens = torch.randn(5, 64)
        pooled = MultiScaleMemoryBank._pool_tokens(tokens, 10)
        assert pooled.shape == (1, 64)
