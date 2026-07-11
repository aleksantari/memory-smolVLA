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

from torch import Tensor, nn

from memory_smolvla.memory.compressor import MemoryCompressor


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
