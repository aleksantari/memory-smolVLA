"""Learned memory compressor (write head).

Compresses the full VLM prefix ``[L_prefix, d_model]`` into a small
fixed number of slots ``[n_slots, d_model]`` before writing to the
memory bank.  This reduces storage and retrieval cost, eliminates
redundancy across near-identical image tokens, and lets the model
learn *what* to memorize.

Uses a Perceiver-style cross-attention: a set of learnable query
vectors attend over the prefix tokens to produce compressed slots.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MemoryCompressor(nn.Module):
    """Perceiver-style compressor for memory bank writes.

    Learnable query vectors cross-attend to the VLM prefix, producing
    a fixed-size compressed representation for storage.

    Args:
        d_model: Dimension of feature vectors.
        n_slots: Number of compressed output slots.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 8,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots

        # Learnable query vectors — these define "what to remember"
        self.slot_queries = nn.Parameter(
            torch.randn(1, n_slots, d_model) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, prefix_hidden: Tensor) -> Tensor:
        """Compress prefix into fixed-size memory slots.

        Args:
            prefix_hidden: VLM prefix features, ``[B, L_prefix, D]``.

        Returns:
            Compressed slots, ``[B, n_slots, D]``.
        """
        B = prefix_hidden.shape[0]
        queries = self.slot_queries.expand(B, -1, -1)

        compressed, _ = self.cross_attn(
            query=queries,
            key=prefix_hidden,
            value=prefix_hidden,
        )
        compressed = self.norm(compressed)
        return compressed
