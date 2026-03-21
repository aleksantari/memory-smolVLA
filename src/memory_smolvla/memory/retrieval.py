"""Cross-attention retrieval module for memory-augmented representations.

Retrieves relevant information from the episodic memory bank given
current VLM features, using multi-head cross-attention with temporal
positional encoding on the memory keys.
"""

from torch import Tensor, nn

from memory_smolvla.memory.temporal_pe import TemporalPositionalEncoding


class CrossAttentionRetrieval(nn.Module):
    """Multi-head cross-attention from current features to memory bank.

    Queries are the current VLM hidden states; keys and values come
    from stored memory entries. Temporal positional encoding is added
    to keys (not values) so attention weights are time-aware while
    retrieved content remains clean.

    Args:
        d_model: Dimension of feature vectors (must match VLM hidden size).
        n_heads: Number of attention heads.
        dropout: Attention dropout probability.
        temporal_min_period: Min wavelength for temporal PE.
        temporal_max_period: Max wavelength for temporal PE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        temporal_min_period: float = 1.0,
        temporal_max_period: float = 1000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.temporal_pe = TemporalPositionalEncoding(
            d_model,
            min_period=temporal_min_period,
            max_period=temporal_max_period,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        current_tokens: Tensor,
        memory_tokens: Tensor,
        time_deltas: Tensor,
    ) -> Tensor:
        """Retrieve memory-augmented features via cross-attention.

        Args:
            current_tokens: Current VLM hidden states, shape ``[B, L, D]``.
            memory_tokens: Flattened memory bank entries, shape ``[B, M, D]``
                where ``M = K * N_tokens`` (K entries, each N_tokens long).
            time_deltas: Time delta for each memory token, shape ``[B, M]``.
                Typically computed as ``current_timestamp - entry_timestamp``,
                broadcast to match each token in the entry.

        Returns:
            Retrieved features, shape ``[B, L, D]`` (same as current_tokens).
        """
        # Add temporal positional encoding to memory keys
        time_pe = self.temporal_pe(time_deltas)  # [B, M, D]
        memory_keys = memory_tokens + time_pe

        # Cross-attention: queries=current, keys=memory+PE, values=memory
        retrieved, _ = self.cross_attn(
            query=current_tokens,
            key=memory_keys,
            value=memory_tokens,
        )

        retrieved = self.norm(retrieved)
        return retrieved
