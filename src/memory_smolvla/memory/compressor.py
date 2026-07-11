"""Learned memory-entry compressor (Perceiver-Resampler).

Compresses a stored bank entry of ``L`` tokens down to a fixed ``n_slots``
tokens via cross-attention from a small set of learned latent queries
(Flamingo/Perceiver-Resampler style; Alayrac et al., 2022).

Design note — where this runs and why it can train
--------------------------------------------------
The bank stores **detached** tensors (no BPTT through the bank, per the
project invariant). A *learned* compressor applied at **storage** time
would therefore never receive gradients: its output would be detached
before any future step could backprop into it.

So this module is applied at **read** time instead, inside
:meth:`FullSeqMemBank.process_batch`, on the stacked raw bank entries just
before retrieval. The stored entries remain frozen constants; the
compressor transforming them into keys/values sits in the current step's
differentiable forward, so its parameters get gradients from the current
flow-matching loss. Retrieval cost drops from ``T * L`` keys to
``T * n_slots`` — which is what makes a deep (episode-spanning) bank
affordable, and is the enabling piece for matching train/eval
consolidation depth.

``mean_pool`` compression is handled separately (parameter-free) at
storage time in the bank itself; this module is only used for the
``perceiver`` mode.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MemoryCompressor(nn.Module):
    """Perceiver-Resampler: ``(T, L, D) -> (T, n_slots, D)``.

    Args:
        dim: Token dimension ``D``.
        n_slots: Number of output tokens per bank entry.
        n_heads: Attention heads for the cross-attention.
        ffn_mult: FFN hidden-size multiplier.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 4,
        n_heads: int = 4,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads}).")
        self.dim = dim
        self.n_slots = n_slots
        self.n_heads = n_heads

        # Learned latent queries. Small init so at step 0 the resampler output
        # is a mild, near-uniform summary rather than a large perturbation.
        self.latents = nn.Parameter(torch.randn(n_slots, dim) * 0.02)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (T, S, D) -> (T, H, S, Dh)
        T, S, D = x.shape
        return x.view(T, S, self.n_heads, D // self.n_heads).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (T, H, S, Dh) -> (T, S, D)
        T, H, S, Dh = x.shape
        return x.transpose(1, 2).reshape(T, S, H * Dh)

    def forward(self, x: Tensor) -> Tensor:
        """Compress ``x`` of shape ``(T, L, D)`` to ``(T, n_slots, D)``."""
        T, L, D = x.shape
        latents = self.latents.unsqueeze(0).expand(T, -1, -1)  # (T, n_slots, D)

        q = self._split_heads(self.q_proj(latents))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = self.out_proj(self._merge_heads(attn))

        slots = self.attn_norm(latents + attn)
        slots = self.ffn_norm(slots + self.ffn(slots))
        return slots
