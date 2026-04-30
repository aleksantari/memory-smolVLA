"""Learned memory compressor (write head).

Compresses the full VLM prefix ``[L_prefix, d_model]`` into a small
fixed number of slots ``[n_slots, d_model]`` before writing to the
memory bank.  This reduces storage and retrieval cost, eliminates
redundancy across near-identical image tokens, and lets the model
learn *what* to memorize.

Architecture: Perceiver-Resampler style. Learnable query vectors
cross-attend over the prefix to produce compressed slots, then
pass through a feed-forward sublayer (with residual + LayerNorm,
pre-norm convention). This matches the Flamingo Perceiver-Resampler
recipe (Alayrac et al., 2022) more faithfully than the prior
single-layer cross-attention design.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MemoryCompressor(nn.Module):
    """Perceiver-Resampler-style compressor for memory bank writes.

    Learnable query vectors cross-attend to the VLM prefix, then an
    FFN refines each slot. Pre-norm residual structure throughout.

    Args:
        d_model: Dimension of feature vectors.
        n_slots: Number of compressed output slots.
        n_heads: Number of attention heads.
        ff_mult: FFN expansion factor (hidden = ff_mult * d_model).
        dropout: Attention/FFN dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots

        # Learnable query vectors — these define "what to remember"
        self.slot_queries = nn.Parameter(
            torch.randn(1, n_slots, d_model) * 0.02
        )

        # Pre-norm cross-attention sublayer
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pre-norm FFN sublayer
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, prefix_hidden: Tensor) -> Tensor:
        """Compress prefix into fixed-size memory slots.

        Args:
            prefix_hidden: VLM prefix features, ``[B, L_prefix, D]``.

        Returns:
            Compressed slots, ``[B, n_slots, D]``.
        """
        B = prefix_hidden.shape[0]
        queries = self.slot_queries.expand(B, -1, -1)

        # Cross-attention with residual: queries are the slots, keys
        # and values are the (LayerNorm'd) prefix tokens.
        q = self.norm_q(queries)
        kv = self.norm_kv(prefix_hidden)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        slots = queries + attn_out

        # FFN with residual.
        slots = slots + self.ff(self.norm_ff(slots))
        return slots
