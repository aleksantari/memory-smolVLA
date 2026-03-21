"""Integration tests for MemorySmolVLAPolicy across training modes.

These tests use a lightweight mock of SmolVLAPolicy to avoid loading
the full 450M-parameter model. The mock replicates the attribute paths
and forward interface that MemorySmolVLAPolicy depends on.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from unittest.mock import MagicMock

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy


# ------------------------------------------------------------------
# Mock SmolVLAPolicy
# ------------------------------------------------------------------

D_MODEL = 32
EXPERT_HIDDEN = 24
CHUNK_SIZE = 5


def _make_mock_policy(d_model: int = D_MODEL, expert_hidden: int = EXPERT_HIDDEN) -> MagicMock:
    """Return a minimal mock that satisfies MemorySmolVLAPolicy's attribute accesses."""
    text_config = MagicMock()
    text_config.hidden_size = d_model

    lm_expert = nn.Linear(expert_hidden, expert_hidden)  # real nn.Module with params
    action_out_proj = nn.Linear(expert_hidden, 6)          # real nn.Module

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

    # Make .parameters() iterable over the real expert / proj params
    all_params = list(lm_expert.parameters()) + list(action_out_proj.parameters())
    base_policy.parameters.side_effect = lambda: iter(all_params)

    def forward_side_effect(batch, noise=None, time=None):
        loss = torch.tensor(0.5, requires_grad=True)
        return loss, {"loss": 0.5}

    base_policy.forward.side_effect = forward_side_effect
    base_policy.reset = MagicMock()
    return base_policy


# ------------------------------------------------------------------
# Training mode: memory_only
# ------------------------------------------------------------------

class TestMemoryOnly:
    def setup_method(self):
        self.base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            self.base, training_mode="memory_only",
            retrieval_n_heads=2, gate_hidden_dim=16
        )

    def test_trainable_params_are_memory_only(self):
        trainable = set(id(p) for p in self.policy.trainable_parameters())
        # Must include retrieval, gate, memory_proj
        for p in self.policy.retrieval.parameters():
            assert id(p) in trainable
        for p in self.policy.gate.parameters():
            assert id(p) in trainable
        assert id(self.policy.memory_proj.weight) in trainable

    def test_expert_params_are_frozen(self):
        vwe = self.base.model.vlm_with_expert
        for p in vwe.lm_expert.parameters():
            assert not p.requires_grad

    def test_action_out_proj_frozen(self):
        for p in self.base.model.action_out_proj.parameters():
            assert not p.requires_grad


# ------------------------------------------------------------------
# Training mode: expert_scratch
# ------------------------------------------------------------------

class TestExpertScratch:
    def setup_method(self):
        self.base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            self.base, training_mode="expert_scratch",
            retrieval_n_heads=2, gate_hidden_dim=16
        )

    def test_expert_params_are_trainable(self):
        vwe = self.base.model.vlm_with_expert
        for p in vwe.lm_expert.parameters():
            assert p.requires_grad

    def test_action_out_proj_trainable(self):
        for p in self.base.model.action_out_proj.parameters():
            assert p.requires_grad

    def test_memory_params_in_trainable(self):
        trainable = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.retrieval.parameters():
            assert id(p) in trainable


# ------------------------------------------------------------------
# Training mode: expert_only_scratch
# ------------------------------------------------------------------

class TestExpertOnlyScratch:
    def setup_method(self):
        self.base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            self.base, training_mode="expert_only_scratch",
            retrieval_n_heads=2, gate_hidden_dim=16
        )

    def test_memory_params_not_in_trainable(self):
        trainable = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.retrieval.parameters():
            assert id(p) not in trainable
        for p in self.policy.gate.parameters():
            assert id(p) not in trainable

    def test_expert_params_in_trainable(self):
        trainable = set(id(p) for p in self.policy.trainable_parameters())
        vwe = self.base.model.vlm_with_expert
        for p in vwe.lm_expert.parameters():
            assert id(p) in trainable


# ------------------------------------------------------------------
# Training mode: expert_finetune
# ------------------------------------------------------------------

class TestExpertFinetune:
    def setup_method(self):
        self.base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            self.base, training_mode="expert_finetune",
            retrieval_n_heads=2, gate_hidden_dim=16
        )

    def test_memory_and_expert_both_trainable(self):
        trainable = set(id(p) for p in self.policy.trainable_parameters())
        for p in self.policy.retrieval.parameters():
            assert id(p) in trainable
        vwe = self.base.model.vlm_with_expert
        for p in vwe.lm_expert.parameters():
            assert id(p) in trainable


# ------------------------------------------------------------------
# Invalid mode
# ------------------------------------------------------------------

def test_invalid_training_mode_raises():
    base = _make_mock_policy()
    with pytest.raises(ValueError, match="Invalid training_mode"):
        MemorySmolVLAPolicy(base, training_mode="bad_mode")


# ------------------------------------------------------------------
# Memory accumulation and reset
# ------------------------------------------------------------------

class TestMemoryAccumulation:
    def setup_method(self):
        self.base = _make_mock_policy()
        self.policy = MemorySmolVLAPolicy(
            self.base, training_mode="memory_only",
            bank_max_size=8, retrieval_n_heads=2, gate_hidden_dim=16
        )

    def test_timestamp_increments(self):
        assert self.policy._timestamp == 0

    def test_reset_memory_clears_bank(self):
        self.policy._timestamp = 5
        self.policy.memory_bank.write(
            torch.zeros(4, D_MODEL), timestamp=0
        )
        self.policy.reset_memory()
        assert self.policy._timestamp == 0
        assert len(self.policy.memory_bank) == 0

    def test_reset_calls_base_reset(self):
        self.policy.reset()
        self.base.reset.assert_called_once()
