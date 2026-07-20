"""V10A Step 5 — orchestration invariants (Coconut thought pass + KV splicing).

Module-level identity / isolation of the Coconut *modules* lives in
``test_coconut_modules.py`` (Step 2). This file covers the *orchestration*:
the ``append_delta_kv`` / ``merge_deltas`` helpers, the forced-self-attn
``vlm_thought_pass`` (shape, one-token-per-layer delta, cache growth, adapters
identity at init), and the ``ForwardMode`` contract.

The full-batch integration invariants (K=0 bit-exact vs V9, per-example offset
decoupling under mixed padding, end-to-end gradient routing) run in
``scripts/test_v10a_smoke.py`` on a real LIBERO batch — that path needs the
dataset, so it stays out of the hermetic unit suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.policy.feature_extractor import ForwardMode
from memory_smolvla.policy.memory_smolvla import append_delta_kv, merge_deltas


def _kv(b, s, h, d, fill):
    return {"key_states": torch.full((b, s, h, d), float(fill)),
            "value_states": torch.full((b, s, h, d), float(fill) + 0.5)}


# --------------------------------------------------------------------------
# ForwardMode + helpers (hermetic)
# --------------------------------------------------------------------------

def test_forward_mode_values():
    assert ForwardMode.PREFIX.value == "prefix"
    assert ForwardMode.LATENT_THOUGHT.value == "thought"
    assert ForwardMode.FLOW_ACTION.value == "flow"


def test_append_delta_kv_grows_by_one_and_preserves_base():
    base = {0: _kv(2, 5, 3, 4, 1.0), 1: _kv(2, 5, 3, 4, 2.0)}
    delta = {0: _kv(2, 1, 3, 4, 9.0), 1: _kv(2, 1, 3, 4, 8.0)}
    out = append_delta_kv(base, delta)
    for l in (0, 1):
        assert out[l]["key_states"].shape[1] == 6            # 5 + 1
        # base rows unchanged, delta appended last
        assert torch.equal(out[l]["key_states"][:, :5], base[l]["key_states"])
        assert torch.equal(out[l]["key_states"][:, 5:], delta[l]["key_states"])
    # base dict not mutated
    assert base[0]["key_states"].shape[1] == 5


def test_append_delta_kv_does_not_mutate_base():
    base = {0: _kv(1, 3, 2, 2, 1.0)}
    delta = {0: _kv(1, 1, 2, 2, 5.0)}
    before = base[0]["key_states"].clone()
    append_delta_kv(base, delta)
    assert torch.equal(base[0]["key_states"], before)


def test_merge_deltas_concatenates_all_passes():
    d0 = {0: _kv(2, 1, 3, 4, 1.0)}
    d1 = {0: _kv(2, 1, 3, 4, 2.0)}
    d2 = {0: _kv(2, 1, 3, 4, 3.0)}
    merged = merge_deltas([d0, d1, d2])
    assert merged[0]["key_states"].shape[1] == 3
    assert torch.equal(merged[0]["key_states"][:, 0], d0[0]["key_states"][:, 0])
    assert torch.equal(merged[0]["key_states"][:, 2], d2[0]["key_states"][:, 0])


def test_merge_deltas_single_pass_identity():
    d0 = {0: _kv(2, 1, 3, 4, 7.0)}
    assert torch.equal(merge_deltas([d0])[0]["key_states"], d0[0]["key_states"])


# --------------------------------------------------------------------------
# vlm_thought_pass (needs the real VLM weights; cached checkpoint, no dataset)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def policy():
    from memory_smolvla.policy.builder import build_policy
    torch.manual_seed(1000)
    try:
        p = build_policy(
            base_checkpoint="lerobot/smolvla_base", injection_layer=15,
            inject_before=True, mem_length=4, retrieval_layers=2,
            consolidate_type="tome", dataloader_type="group", group_size=4,
            compression="reasoning", n_slots=8, aux_loss_weight=0.0,
            bptt_memory=True, coconut_enabled=True, num_thoughts=2,
            policy_overrides={"n_action_steps": 10, "prefix_length": -1,
                              "pad_language_to": "longest",
                              "scheduler_decay_steps": 100000},
        ).float().eval()
    except Exception as e:  # pragma: no cover - offline / no checkpoint
        pytest.skip(f"base checkpoint unavailable: {e}")
    return p


def _synthetic_cache(policy, B, S):
    vwe = policy.base_policy.model.vlm_with_expert
    n_kv = vwe.num_key_value_heads
    head_dim = vwe.vlm.config.text_config.head_dim
    n_layers = vwe.num_vlm_layers
    torch.manual_seed(0)
    return {l: {"key_states": torch.randn(B, S, n_kv, head_dim),
                "value_states": torch.randn(B, S, n_kv, head_dim)}
            for l in range(n_layers)}


def test_thought_pass_shapes_and_one_token_per_layer(policy):
    B, S, D = 2, 7, policy.d_model
    cache = _synthetic_cache(policy, B, S)
    z = torch.randn(B, 1, D)
    pos = torch.full((B, 1), S, dtype=torch.long)
    key_pad = torch.ones(B, 1, S + 1, dtype=torch.bool)
    adapters = {int(l): m for l, m in policy.coconut_adapters.items()}
    out = policy.feature_extractor.vlm_thought_pass(z, cache, pos, key_pad, adapters)

    n_layers = policy.base_policy.model.vlm_with_expert.num_vlm_layers
    assert out.hidden_state.shape == (B, 1, D)
    assert set(out.delta_kv.keys()) == set(range(n_layers))
    for l in range(n_layers):
        assert out.delta_kv[l]["key_states"].shape[1] == 1     # exactly one token
        assert out.delta_kv[l]["value_states"].shape[1] == 1


def test_thought_pass_cache_grows_by_one_per_append(policy):
    B, S, D = 1, 5, policy.d_model
    cache = _synthetic_cache(policy, B, S)
    z = torch.randn(B, 1, D)
    for k in range(3):
        cur = S + k
        pos = torch.full((B, 1), cur, dtype=torch.long)
        key_pad = torch.ones(B, 1, cur + 1, dtype=torch.bool)
        out = policy.feature_extractor.vlm_thought_pass(z, cache, pos, key_pad, {})
        cache = append_delta_kv(cache, out.delta_kv)
        assert cache[0]["key_states"].shape[1] == cur + 1       # grew by exactly one
        z = out.hidden_state


def test_thought_pass_adapters_identity_at_init_no_op(policy):
    """Zero-init latent adapters ⇒ passing them changes nothing at init."""
    B, S, D = 2, 6, policy.d_model
    cache = _synthetic_cache(policy, B, S)
    z = torch.randn(B, 1, D)
    pos = torch.full((B, 1), S, dtype=torch.long)
    key_pad = torch.ones(B, 1, S + 1, dtype=torch.bool)
    adapters = {int(l): m for l, m in policy.coconut_adapters.items()}
    with torch.no_grad():
        with_ad = policy.feature_extractor.vlm_thought_pass(z, cache, pos, key_pad, adapters)
        without = policy.feature_extractor.vlm_thought_pass(z, cache, pos, key_pad, {})
    assert torch.allclose(with_ad.hidden_state, without.hidden_state, atol=1e-6)
    assert torch.allclose(with_ad.delta_kv[15]["key_states"],
                          without.delta_kv[15]["key_states"], atol=1e-6)


def test_thought_pass_key_padding_masks_out_keys(policy):
    """Masking every cached key but the thought's own must change attention."""
    B, S, D = 1, 6, policy.d_model
    cache = _synthetic_cache(policy, B, S)
    z = torch.randn(B, 1, D)
    pos = torch.full((B, 1), S, dtype=torch.long)
    full = torch.ones(B, 1, S + 1, dtype=torch.bool)
    only_self = torch.zeros(B, 1, S + 1, dtype=torch.bool)
    only_self[:, :, -1] = True                                  # attend only this token
    with torch.no_grad():
        a = policy.feature_extractor.vlm_thought_pass(z, cache, pos, full, {})
        b = policy.feature_extractor.vlm_thought_pass(z, cache, pos, only_self, {})
    assert not torch.allclose(a.hidden_state, b.hidden_state)


def test_coconut_modules_in_trainable_params(policy):
    names = set()
    seed_ids = {id(p) for p in policy.coconut_seed.parameters()}
    seed_ids |= {id(p) for p in policy.coconut_adapters.parameters()}
    seed_ids |= {id(p) for p in policy.coconut_feedback.parameters()}
    trainable_ids = {id(p) for p in policy.trainable_parameters()}
    assert seed_ids <= trainable_ids, "Coconut params missing from trainable set"
