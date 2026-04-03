"""Integration tests for new memory features in MemorySmolVLAPolicy.

Tests the memory callback end-to-end with each new feature:
- FIFO eviction
- Memory compressor
- Write gate
- Working memory backend
- Multi-scale bank
- Gate regularization
- Step increment (temporal mismatch fix)

Uses direct callback invocation rather than a full SmolVLA forward pass,
avoiding the need for the full 450M model.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock

import sys
import types

# Stub all lerobot submodules so memory_smolvla.policy can import
# without the full lerobot package installed.
_LEROBOT_STUBS = [
    "lerobot",
    "lerobot.policies",
    "lerobot.policies.smolvla",
    "lerobot.policies.smolvla.configuration_smolvla",
    "lerobot.policies.smolvla.modeling_smolvla",
    "lerobot.policies.factory",
    "lerobot.datasets",
    "lerobot.datasets.lerobot_dataset",
]
for _name in _LEROBOT_STUBS:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        # Mark package-level stubs as packages
        if not _name.endswith(("_smolvla", "factory", "lerobot_dataset")):
            _mod.__path__ = []
        sys.modules[_name] = _mod

# Add class-level stubs that builder.py imports by name
sys.modules["lerobot.policies.smolvla.configuration_smolvla"].SmolVLAConfig = MagicMock
sys.modules["lerobot.policies.smolvla.modeling_smolvla"].SmolVLAPolicy = MagicMock

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy  # noqa: E402

D_MODEL = 64
L_PREFIX = 20


def _make_mock_policy(d_model: int = D_MODEL) -> MagicMock:
    """Minimal mock that satisfies MemorySmolVLAPolicy's attribute accesses."""
    text_config = MagicMock()
    text_config.hidden_size = d_model

    lm_expert = nn.Linear(d_model, d_model)
    action_out_proj = nn.Linear(d_model, 6)

    vlm_config = MagicMock()
    vlm_config.text_config = text_config

    vlm_with_expert = MagicMock()
    vlm_with_expert.config = vlm_config
    vlm_with_expert.lm_expert = lm_expert

    model = MagicMock()
    model.vlm_with_expert = vlm_with_expert
    model.action_out_proj = action_out_proj

    base_policy = MagicMock()
    base_policy.model = model

    all_params = list(lm_expert.parameters()) + list(action_out_proj.parameters())
    base_policy.parameters.side_effect = lambda: iter(all_params)
    base_policy.reset = MagicMock()
    return base_policy


def _make_prefix(b: int = 1, seq_len: int = L_PREFIX, d: int = D_MODEL) -> torch.Tensor:
    return torch.randn(b, seq_len, d)


# ------------------------------------------------------------------
# Episodic memory with FIFO eviction (default)
# ------------------------------------------------------------------


class TestEpisodicFIFO:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=4,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            eviction="fifo",
        )

    def test_callback_returns_correct_shape(self):
        prefix = _make_prefix()
        out = self.policy._memory_callback(prefix, layer_idx=8)
        assert out.shape == prefix.shape

    def test_bank_fills_and_evicts(self):
        for t in range(6):
            self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert len(self.policy.memory_bank) == 4

    def test_timestamps_advance(self):
        for t in range(3):
            self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert self.policy._timestamp == 3

    def test_gate_alpha_is_recorded(self):
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert self.policy._last_gate_alpha is not None
        assert self.policy._last_gate_alpha.shape == (1, L_PREFIX, 1)

    def test_reset_clears_everything(self):
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        self.policy.reset_memory()
        assert self.policy._timestamp == 0
        assert len(self.policy.memory_bank) == 0
        assert self.policy._last_gate_alpha is None


# ------------------------------------------------------------------
# Episodic memory with consolidation eviction
# ------------------------------------------------------------------


class TestEpisodicConsolidate:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=4,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            eviction="consolidate",
        )

    def test_consolidation_keeps_bank_at_max(self):
        for t in range(6):
            self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert len(self.policy.memory_bank) <= 4


# ------------------------------------------------------------------
# Memory compressor
# ------------------------------------------------------------------


class TestCompressorIntegration:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=4,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            use_compressor=True,
            compressor_n_slots=4,
        )

    def test_callback_returns_correct_shape(self):
        prefix = _make_prefix()
        out = self.policy._memory_callback(prefix, layer_idx=8)
        assert out.shape == prefix.shape

    def test_bank_stores_compressed_entries(self):
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        # Second call reads from bank — bank entry should have n_slots tokens
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        memories, _ = self.policy.memory_bank.read_all()
        # Each entry should have compressor_n_slots tokens, not L_PREFIX
        assert memories.shape[1] == 4  # n_slots

    def test_compressor_in_trainable_params(self):
        trainable_ids = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.compressor.parameters():
            assert id(p) in trainable_ids


# ------------------------------------------------------------------
# Write gate
# ------------------------------------------------------------------


class TestWriteGateIntegration:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=8,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            use_write_gate=True,
        )

    def test_callback_returns_correct_shape(self):
        prefix = _make_prefix()
        out = self.policy._memory_callback(prefix, layer_idx=8)
        assert out.shape == prefix.shape

    def test_write_prob_is_recorded(self):
        # First call: no recent memory → write_prob = 1.0
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        # Second call: has recent memory → write gate produces a probability
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert self.policy._last_write_prob is not None

    def test_write_gate_in_trainable_params(self):
        trainable_ids = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.write_gate.parameters():
            assert id(p) in trainable_ids


# ------------------------------------------------------------------
# Working memory backend
# ------------------------------------------------------------------


class TestWorkingMemoryBackend:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            memory_backend="working",
        )

    def test_callback_returns_correct_shape(self):
        prefix = _make_prefix()
        out = self.policy._memory_callback(prefix, layer_idx=8)
        assert out.shape == prefix.shape

    def test_no_memory_bank(self):
        assert self.policy.memory_bank is None

    def test_no_retrieval_module(self):
        assert self.policy.retrieval is None

    def test_working_mem_exists(self):
        assert self.policy.working_mem is not None

    def test_multiple_calls_work(self):
        """Working memory should handle sequential calls without error."""
        for _ in range(5):
            out = self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert out.shape == (1, L_PREFIX, D_MODEL)

    def test_reset_clears_gru_state(self):
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert self.policy.working_mem._hidden is not None
        self.policy.reset_memory()
        assert self.policy.working_mem._hidden is None

    def test_working_mem_in_trainable_params(self):
        trainable_ids = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.working_mem.parameters():
            assert id(p) in trainable_ids


# ------------------------------------------------------------------
# Multi-scale bank
# ------------------------------------------------------------------


class TestMultiScaleIntegration:
    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=8,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            use_multi_scale=True,
        )

    def test_callback_returns_correct_shape(self):
        prefix = _make_prefix()
        out = self.policy._memory_callback(prefix, layer_idx=8)
        assert out.shape == prefix.shape

    def test_multi_scale_bank_populated(self):
        self.policy._memory_callback(_make_prefix(), layer_idx=8)
        bank = self.policy.memory_bank
        assert len(bank.short) == 1
        assert len(bank.medium) == 1
        assert len(bank.long) == 1

    def test_retrieval_works_with_multi_scale(self):
        """Ensure cross-attention handles padded multi-scale memories."""
        # Write several entries so all scales have data
        for t in range(3):
            self.policy._memory_callback(_make_prefix(), layer_idx=8)
        # The callback should not crash on the concatenated output
        out = self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# Gate regularization
# ------------------------------------------------------------------


class TestGateRegularizationIntegration:
    def setup_method(self):
        base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            base,
            training_mode="memory_only",
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            alpha_target=0.2,
            alpha_reg_weight=0.1,
        )
        # Wire up a forward that triggers callback manually
        self.policy._memory_callback(_make_prefix(), layer_idx=8)

    def test_reg_loss_is_computed(self):
        alpha = self.policy._last_gate_alpha
        reg = self.policy.gate.regularization_loss(alpha)
        assert reg.item() > 0.0  # alpha != 0.2 at init

    def test_gate_alpha_target_stored(self):
        assert self.policy.gate.alpha_target == 0.2

    def test_gate_reg_weight_stored(self):
        assert self.policy.gate.alpha_reg_weight == 0.1


# ------------------------------------------------------------------
# Step increment (temporal mismatch fix)
# ------------------------------------------------------------------


class TestStepIncrement:
    def test_default_increment_is_1(self):
        policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            retrieval_n_heads=4,
            gate_hidden_dim=32,
        )
        policy._memory_callback(_make_prefix(), layer_idx=8)
        assert policy._timestamp == 1

    def test_custom_increment(self):
        policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            step_increment=50,
        )
        policy._memory_callback(_make_prefix(), layer_idx=8)
        assert policy._timestamp == 50
        policy._memory_callback(_make_prefix(), layer_idx=8)
        assert policy._timestamp == 100

    def test_step_increment_property(self):
        policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            retrieval_n_heads=4,
            gate_hidden_dim=32,
        )
        assert policy.step_increment == 1
        policy.step_increment = 50
        assert policy.step_increment == 50


# ------------------------------------------------------------------
# Combined features
# ------------------------------------------------------------------


class TestCombinedFeatures:
    """Test compressor + write gate + multi-scale + regularization together."""

    def setup_method(self):
        self.policy = MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            bank_max_size=8,
            retrieval_n_heads=4,
            gate_hidden_dim=32,
            use_compressor=True,
            compressor_n_slots=4,
            use_write_gate=True,
            use_multi_scale=True,
            alpha_target=0.2,
            alpha_reg_weight=0.05,
        )

    def test_full_episode_simulation(self):
        """Simulate a 10-frame episode with all features enabled."""
        for t in range(10):
            out = self.policy._memory_callback(_make_prefix(), layer_idx=8)
            assert out.shape == (1, L_PREFIX, D_MODEL)
            assert torch.isfinite(out).all()

        assert self.policy._timestamp == 10
        assert self.policy._last_gate_alpha is not None

    def test_reset_and_new_episode(self):
        """Simulate two episodes with reset in between."""
        for t in range(5):
            self.policy._memory_callback(_make_prefix(), layer_idx=8)

        self.policy.reset_memory()
        assert self.policy._timestamp == 0

        for t in range(5):
            out = self.policy._memory_callback(_make_prefix(), layer_idx=8)
        assert out.shape == (1, L_PREFIX, D_MODEL)


# ------------------------------------------------------------------
# Invalid backend
# ------------------------------------------------------------------


def test_invalid_memory_backend_raises():
    with pytest.raises(ValueError, match="Invalid memory_backend"):
        MemorySmolVLAPolicy(
            _make_mock_policy(),
            training_mode="memory_only",
            memory_backend="invalid",
        )
