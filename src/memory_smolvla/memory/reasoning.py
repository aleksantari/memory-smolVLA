"""V8 reasoning heads (reasoning-tokens-as-memory-content).

Two small modules implementing the V8 idea: instead of storing raw (or
mean-pooled) prefix hidden states in the memory bank as in V7, each frame's
prefix is first summarized into a handful of learned "reasoning tokens" and
*those* become the bank content.

- :class:`ReasoningSummaryHead` produces the per-frame reasoning tokens via
  a Perceiver-Resampler over the prefix.
- :class:`FutureStatePredictor` provides the auxiliary training signal: it
  predicts the robot's proprioceptive state ``future_horizon`` steps ahead
  (PTP-style) from the reasoning tokens, forcing them to encode
  task-dynamics-relevant information.

The auxiliary loss is necessary because of the project's no-BPTT invariant:
bank writes are ``.detach()``ed, so gradients from *future* steps' losses
can never reach the summary head through the bank. The aux prediction loss
is computed in the same step's differentiable forward, before detachment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from memory_smolvla.memory.compressor import MemoryCompressor

try:
    from torch.nn import RMSNorm
except ImportError:  # pragma: no cover - torch >= 2.4 ships nn.RMSNorm

    class RMSNorm(nn.Module):  # type: ignore[no-redef]
        """Minimal RMSNorm fallback for older torch versions."""

        def __init__(self, dim: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Tensor) -> Tensor:
            normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
            return normed * self.weight


class ReasoningSummaryHead(nn.Module):
    """Summarize the VLM prefix into learned reasoning tokens.

    ``(B, L, D) -> (B, n_slots, D)`` via a Perceiver-Resampler: ``n_slots``
    learned latent queries cross-attend the prefix tokens, followed by an
    FFN. In V8 the output tokens are what get written to the memory bank
    (vs V7 storing the raw/mean-pooled prefix), so retrieval later reads
    compact, learned summaries instead of raw hidden states.

    Because bank writes are detached (no BPTT through the bank), this head
    receives no gradient from future steps' retrievals; it is trained via
    the auxiliary loss produced by :class:`FutureStatePredictor` in the
    same step's forward.

    Internally this wraps :class:`MemoryCompressor`, which already
    implements exactly this resampler over a leading batch dim.

    Args:
        dim: Token dimension ``D``.
        n_slots: Number of reasoning tokens per frame.
        n_heads: Attention heads for the cross-attention.
        ffn_mult: FFN hidden-size multiplier.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 8,
        n_heads: int = 4,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.resampler = MemoryCompressor(
            dim, n_slots=n_slots, n_heads=n_heads, ffn_mult=ffn_mult
        )

    def forward(self, prefix: Tensor) -> Tensor:
        """Summarize ``prefix`` of shape ``(B, L, D)`` to ``(B, n_slots, D)``."""
        return self.resampler(prefix)


class FutureStatePredictor(nn.Module):
    """Predict the future proprioceptive state from reasoning tokens.

    ``(B, n_slots, D) -> (B, state_dim)``: mean-pool the reasoning tokens
    over the slot axis, then a 2-layer MLP. The target is the robot's
    proprioceptive state ``future_horizon`` steps ahead (PTP-style aux
    objective). The resulting loss is the training signal for
    :class:`ReasoningSummaryHead` — see its docstring for why the bank
    itself cannot provide one.

    Args:
        dim: Reasoning-token dimension ``D``.
        state_dim: Dimension of the proprioceptive state to predict.
        hidden: MLP hidden size.
    """

    def __init__(self, dim: int, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, reasoning_tokens: Tensor) -> Tensor:
        """Predict future state from ``reasoning_tokens`` of shape ``(B, n_slots, D)``."""
        pooled = reasoning_tokens.mean(dim=1)  # (B, D)
        return self.mlp(pooled)


# ---------------------------------------------------------------------------
# V10A "Coconut" latent-thought modules (RevB spec).
# ---------------------------------------------------------------------------


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``x`` ``(B, L, D)`` over the ``L`` axis, weighted by ``mask``.

    ``mask`` is ``(B, L)`` bool/float; the denominator is clamped so rows
    with zero valid tokens return zeros instead of NaN.
    """
    mask = mask.to(dtype=x.dtype)  # (B, L)
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, D)


def masked_rms(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-row RMS of ``x`` ``(B, L, D)`` over valid tokens and channels.

    ``mask`` is ``(B, L)`` bool/float; returns ``(B,)``. The element count
    is clamped so fully-padded rows do not divide by zero.
    """
    mask = mask.to(dtype=x.dtype)  # (B, L)
    sq_sum = (x.square() * mask.unsqueeze(-1)).sum(dim=(-2, -1))  # (B,)
    count = (mask.sum(dim=-1) * x.shape[-1]).clamp(min=1.0)  # (B,)
    return torch.sqrt(sq_sum / count + eps)


class CoconutThoughtSeed(nn.Module):
    """Produce the initial latent thought token ``z0`` ``(B, 1, D)``.

    ``z0 = e_think + W_seed[ LN(sg(mean(memory_slots))) ; LN(sg(maskmean(retrieved_prefix))) ]``

    All seed inputs are stop-grad (``.detach()``): the thought pass must not
    backprop into the memory bank contents or the retrieved prefix.

    Args:
        d: Token dimension ``D``.
    """

    def __init__(self, d: int = 960) -> None:
        super().__init__()
        self.e_think = nn.Parameter(torch.randn(d) * 0.02)
        self.ln_m = nn.LayerNorm(d)
        self.ln_r = nn.LayerNorm(d)
        self.W_seed = nn.Linear(2 * d, d)

    def forward(
        self, memory_slots: Tensor, retrieved_prefix: Tensor, r_mask: Tensor
    ) -> Tensor:
        """Seed from ``memory_slots`` ``(B, n_slots, D)`` and ``retrieved_prefix`` ``(B, L, D)``."""
        m = self.ln_m(memory_slots.mean(1).detach())  # (B, D), stop-grad
        r = self.ln_r(masked_mean(retrieved_prefix, r_mask).detach())  # (B, D), stop-grad
        z = self.e_think + self.W_seed(torch.cat([m, r], -1))  # (B, D)
        return z.unsqueeze(1)  # (B, 1, D)


class CoconutFeedbackAdapter(nn.Module):
    """Gated residual MLP mapping a thought hidden state to the next input.

    Residual form: exactly identity at init (``fc2`` zero => update == 0),
    with a learned scalar sigmoid gate on the update. No trailing norm.

    Args:
        d: Token dimension ``D``.
        hid: MLP hidden size.
    """

    def __init__(self, d: int = 960, hid: int = 1920) -> None:
        super().__init__()
        self.n1 = RMSNorm(d)
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)
        self.gate = nn.Parameter(torch.tensor(-1.0))  # fp32; sigmoid(-1)=0.269
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: Tensor) -> Tensor:
        """Map thought hidden ``h`` ``(B, 1, D)`` to the next thought input."""
        update = self.fc2(F.gelu(self.fc1(self.n1(h))))
        return h + torch.sigmoid(self.gate).to(h.dtype) * update


class MatchInputRMS(nn.Module):
    """Match the thought token's scale to the prefix residual-stream RMS.

    Normalizes the thought to unit RMS, then rescales by the reference
    tokens' masked RMS times a learned ``exp(log_scale_adjustment)`` factor
    (identity factor at init).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_scale_adjustment = nn.Parameter(torch.zeros(()))  # fp32

    def forward(
        self, thought: Tensor, reference_tokens: Tensor, reference_mask: Tensor
    ) -> Tensor:
        """Rescale ``thought`` ``(B, 1, D)`` to match ``reference_tokens`` ``(B, L, D)`` RMS."""
        ref_rms = masked_rms(reference_tokens.detach(), reference_mask)  # (B,)
        normed = thought * torch.rsqrt(thought.square().mean(-1, keepdim=True) + 1e-6)
        return normed * (ref_rms * self.log_scale_adjustment.exp())[:, None, None]


class LatentModeAdapter(nn.Module):
    """Rank-``r`` zero-init bottleneck applied on the thought row.

    ``h + up(down(h))`` with ``up`` zero-initialized, so it is a local
    identity at init.

    Args:
        d: Token dimension ``D``.
        r: Bottleneck rank.
    """

    def __init__(self, d: int = 960, r: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)  # local identity at init

    def forward(self, h: Tensor) -> Tensor:
        """Adapt the thought row ``h`` ``(B, 1, D)``."""
        return h + self.up(self.down(h))


@dataclass
class ThoughtPassOutput:
    """Result of one latent-thought forward pass."""

    hidden_state: Tensor  # last-layer hidden of this thought, (B, 1, D)
    delta_kv: dict  # {layer_idx: {"key_states", "value_states"}}, one token/layer


@dataclass
class CoconutState:
    """Accumulated state across the Coconut latent-thought loop."""

    thought_prefix_cache: dict  # per-tensor-detached prefix KV
    thought_deltas: list
    processed_hidden_states: list
    feedback_embeddings: list
