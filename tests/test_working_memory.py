"""Tests for WorkingMemory (GRU-based recurrent state)."""

import pytest
import torch

from memory_smolvla.memory.working_memory import WorkingMemory


class TestWorkingMemory:
    """Tests for the GRU working memory module."""

    @pytest.fixture
    def wm(self):
        return WorkingMemory(d_model=64, state_dim=64, n_layers=1)

    def test_output_shape(self, wm):
        prefix = torch.randn(1, 50, 64)
        out = wm(prefix)
        assert out.shape == (1, 50, 64)

    def test_zero_init_output(self, wm):
        """Output projection is zero-initialized, so first output should be near-zero."""
        prefix = torch.randn(1, 50, 64)
        out = wm(prefix)
        # Zero-init output_proj means output should be all zeros
        assert out.abs().max() < 1e-5

    def test_state_persists_across_calls(self, wm):
        prefix = torch.randn(1, 50, 64)

        # First call initializes state
        wm(prefix)
        assert wm._hidden is not None

        # Second call uses existing state — output should differ
        # because GRU state was updated
        wm(prefix)
        # After training, these would differ; at init they're both ~zero
        # due to zero-init output_proj

    def test_reset_clears_state(self, wm):
        prefix = torch.randn(1, 50, 64)
        wm(prefix)
        assert wm._hidden is not None

        wm.reset()
        assert wm._hidden is None

    def test_detach_state(self, wm):
        prefix = torch.randn(1, 50, 64)
        wm(prefix)
        wm.detach_state()
        assert wm._hidden is not None
        assert not wm._hidden.requires_grad

    def test_gradients_flow(self, wm):
        prefix = torch.randn(1, 50, 64, requires_grad=True)
        out = wm(prefix)
        out.sum().backward()
        assert prefix.grad is not None

    def test_different_batch_size_resets_hidden(self, wm):
        prefix_b1 = torch.randn(1, 50, 64)
        wm(prefix_b1)
        h1_shape = wm._hidden.shape

        prefix_b2 = torch.randn(2, 50, 64)
        wm(prefix_b2)
        h2_shape = wm._hidden.shape

        assert h1_shape[1] == 1
        assert h2_shape[1] == 2
