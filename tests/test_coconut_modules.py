"""Tests for the V10A Coconut latent-thought modules."""

from __future__ import annotations

import torch

from memory_smolvla.memory.reasoning import (
    CoconutFeedbackAdapter,
    CoconutState,
    CoconutThoughtSeed,
    LatentModeAdapter,
    MatchInputRMS,
    ThoughtPassOutput,
    masked_mean,
    masked_rms,
)

B, D, N_SLOTS, L, R = 2, 64, 8, 10, 16


def _mask(all_valid: bool = False) -> torch.Tensor:
    mask = torch.ones(B, L, dtype=torch.bool)
    if not all_valid:
        mask[1, L // 2 :] = False  # second row is padded
    return mask


def test_output_shapes() -> None:
    torch.manual_seed(0)
    seed = CoconutThoughtSeed(d=D)
    feedback = CoconutFeedbackAdapter(d=D, hid=2 * D)
    matcher = MatchInputRMS()
    adapter = LatentModeAdapter(d=D, r=R)

    memory_slots = torch.randn(B, N_SLOTS, D)
    retrieved_prefix = torch.randn(B, L, D)
    mask = _mask()

    z0 = seed(memory_slots, retrieved_prefix, mask)
    assert z0.shape == (B, 1, D)

    h = torch.randn(B, 1, D)
    assert feedback(h).shape == (B, 1, D)
    assert matcher(h, retrieved_prefix, mask).shape == (B, 1, D)
    assert adapter(h).shape == (B, 1, D)


def test_latent_mode_adapter_identity_at_init() -> None:
    torch.manual_seed(0)
    adapter = LatentModeAdapter(d=D, r=R)
    h = torch.randn(B, 1, D)
    assert torch.equal(adapter(h), h)


def test_feedback_adapter_identity_at_init() -> None:
    torch.manual_seed(0)
    feedback = CoconutFeedbackAdapter(d=D, hid=2 * D)
    h = torch.randn(B, 1, D)
    assert torch.equal(feedback(h), h)


def test_seed_gradient_isolation() -> None:
    torch.manual_seed(0)
    seed = CoconutThoughtSeed(d=D)
    memory_slots = torch.randn(B, N_SLOTS, D, requires_grad=True)
    retrieved_prefix = torch.randn(B, L, D, requires_grad=True)

    z0 = seed(memory_slots, retrieved_prefix, _mask())
    z0.sum().backward()

    # Stop-grad: inputs receive no gradient.
    assert memory_slots.grad is None
    assert retrieved_prefix.grad is None
    # Seed parameters do receive gradient.
    assert seed.e_think.grad is not None
    assert seed.e_think.grad.abs().sum() > 0
    assert seed.W_seed.weight.grad is not None
    assert seed.W_seed.weight.grad.abs().sum() > 0


def test_feedback_gate_changes_output_with_nonzero_fc2() -> None:
    torch.manual_seed(0)
    feedback = CoconutFeedbackAdapter(d=D, hid=2 * D)
    # With fc2 == 0 the gate test would be vacuous; make fc2 known-nonzero.
    with torch.no_grad():
        feedback.fc2.weight.fill_(0.1)
        feedback.fc2.bias.fill_(0.05)

    h = torch.randn(B, 1, D)
    out_default = feedback(h)  # gate = -1.0
    with torch.no_grad():
        feedback.gate.fill_(3.0)
    out_open = feedback(h)

    assert not torch.allclose(out_default, out_open)
    # Sanity: with nonzero fc2 the adapter is no longer identity.
    assert not torch.equal(out_default, h)


def test_match_input_rms_matches_reference_rms_at_init() -> None:
    torch.manual_seed(0)
    matcher = MatchInputRMS()
    assert matcher.log_scale_adjustment.item() == 0.0  # exp = 1 at init

    thought = torch.randn(B, 1, D) * 37.0  # arbitrary scale
    reference = torch.randn(B, L, D) * 3.0
    mask = _mask()

    out = matcher(thought, reference, mask)
    out_rms = out.square().mean(-1).sqrt().squeeze(1)  # (B,)
    ref_rms = masked_rms(reference, mask)  # (B,)
    assert torch.allclose(out_rms, ref_rms, atol=1e-4)

    # Manual check of the masked reference RMS for the padded row.
    valid = reference[1, : L // 2]
    manual = valid.square().mean().sqrt()
    assert torch.allclose(ref_rms[1], manual, atol=1e-4)


def test_all_params_fp32() -> None:
    modules = [
        CoconutThoughtSeed(d=D),
        CoconutFeedbackAdapter(d=D, hid=2 * D),
        MatchInputRMS(),
        LatentModeAdapter(d=D, r=R),
    ]
    for module in modules:
        for name, p in module.named_parameters():
            assert p.dtype == torch.float32, f"{type(module).__name__}.{name}"


def test_masked_mean_all_valid_and_padded() -> None:
    torch.manual_seed(0)
    x = torch.randn(B, L, D)
    full = torch.ones(B, L)
    assert torch.allclose(masked_mean(x, full), x.mean(1), atol=1e-6)

    mask = _mask()
    out = masked_mean(x, mask)
    assert torch.allclose(out[1], x[1, : L // 2].mean(0), atol=1e-6)

    # Fully-padded row: no NaN/inf, returns zeros.
    empty = torch.zeros(B, L)
    zeros = masked_mean(x, empty)
    assert torch.isfinite(zeros).all()
    assert torch.equal(zeros, torch.zeros(B, D))
    assert torch.isfinite(masked_rms(x, empty)).all()


def test_dataclasses() -> None:
    h = torch.randn(B, 1, D)
    tpo = ThoughtPassOutput(
        hidden_state=h,
        delta_kv={0: {"key_states": h, "value_states": h}},
    )
    assert tpo.hidden_state.shape == (B, 1, D)
    assert set(tpo.delta_kv[0]) == {"key_states", "value_states"}

    state = CoconutState(
        thought_prefix_cache={},
        thought_deltas=[],
        processed_hidden_states=[],
        feedback_embeddings=[],
    )
    assert state.thought_deltas == []
    assert state.feedback_embeddings == []
