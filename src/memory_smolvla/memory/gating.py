"""Sigmoid gate for fusing current features with retrieved memory.

Produces a learned scalar gate per token position that controls how
much retrieved memory information is blended into the current
representation. Returns both the fused output and the gate values
for logging and visualization.
"""

import torch
from torch import Tensor, nn


class SigmoidGate(nn.Module):
    """MLP-based sigmoid gate for memory fusion.

    Takes concatenated ``[current, retrieved]`` features and produces
    a per-token scalar gate ``alpha`` in ``[0, 1]``. The fused output
    is ``alpha * retrieved + (1 - alpha) * current``.

    Args:
        d_model: Dimension of input feature vectors.
        hidden_dim: Hidden dimension of the gate MLP. If ``None``,
            defaults to ``d_model``.
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, current: Tensor, retrieved: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute gated fusion of current and retrieved features.

        Args:
            current: Current VLM hidden states, shape ``[B, L, D]``.
            retrieved: Retrieved memory features, shape ``[B, L, D]``.

        Returns:
            Tuple of:
                - fused: Gated combination, shape ``[B, L, D]``.
                - alpha: Gate values, shape ``[B, L, 1]``. Useful for
                  logging and visualization of memory utilization.
        """
        # Concatenate along feature dimension: [B, L, 2*D]
        combined = torch.cat([current, retrieved], dim=-1)

        # Compute per-token gate: [B, L, 1]
        alpha = self.gate_mlp(combined)

        # Gated fusion
        fused = alpha * retrieved + (1.0 - alpha) * current

        return fused, alpha
