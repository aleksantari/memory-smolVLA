"""Memory building blocks adapted from MemoryVLA.

Ports the three primitives used by ``FullSeqMemBank``:
``TimestepEmbedder`` (sinusoidal temporal PE), ``CrossTransformerBlock``
(cross-attention retrieval block), and ``GateFusion`` (learned per-token
sigmoid gate).

Reference: ``vla/memory_vla.py`` in the MemoryVLA repo.
See ``memory_smolvla_implementation_spec.md`` §3.1–§3.3.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of integer timesteps, projected to ``hidden_size``."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            next(self.mlp.parameters()).dtype
        )
        return self.mlp(t_freq)


class CrossTransformerBlock(nn.Module):
    """Cross-attention + FFN block, post-norm style.

    Single-head attention via ``scaled_dot_product_attention``. Query is the
    current VLM output sequence; key/value come from the flattened bank
    (with temporal PE already added to keys by the caller).
    """

    def __init__(self, feature_dim: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(feature_dim * ffn_mult, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, query: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        x = self.attn_norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)


class GateFusion(nn.Module):
    """Learned per-token sigmoid gate over ``[current; retrieved]``.

    Init uses ``std=1e-3`` (matches MemoryVLA — **not** zero-init). At step 0
    the gate starts near ``sigmoid(0) = 0.5``, so the retrieved signal enters
    at ~50% weight when the bank is non-empty. With an empty bank the caller
    passes ``retrieved = current``, making the output identically ``current``.

    Fusion convention: ``scale * current + (1 - scale) * retrieved``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x_current: Tensor, x_retrieved: Tensor) -> Tensor:
        scale = torch.sigmoid(self.proj(torch.cat([x_current, x_retrieved], dim=-1)))
        return scale * x_current + (1 - scale) * x_retrieved

    def last_scale(self, x_current: Tensor, x_retrieved: Tensor) -> Tensor:
        """Return the sigmoid gate values for logging (no fusion)."""
        return torch.sigmoid(self.proj(torch.cat([x_current, x_retrieved], dim=-1)))
