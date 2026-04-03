"""Sigmoid gate for fusing current features with retrieved memory.

Produces a learned scalar gate per token position that controls how
much retrieved memory information is blended into the current
representation. Returns both the fused output and the gate values
for logging and visualization.

Supports an optional L2 regularization penalty on gate alpha to
prevent saturation (gate opening to ~1.0 and effectively replacing
the current features rather than augmenting them).
"""

import torch
from torch import Tensor, nn


class SigmoidGate(nn.Module):
    """MLP-based sigmoid gate for memory fusion with optional regularization.

    Takes concatenated ``[current, retrieved]`` features and produces
    a per-token scalar gate ``alpha`` in ``[0, 1]``. The fused output
    is ``alpha * retrieved + (1 - alpha) * current``.

    Args:
        d_model: Dimension of input feature vectors.
        hidden_dim: Hidden dimension of the gate MLP. If ``None``,
            defaults to ``d_model``.
        alpha_target: Target alpha value for regularization. The
            regularization loss penalizes deviation from this target.
        alpha_reg_weight: Weight for the alpha regularization loss.
            Set to 0 to disable regularization (default).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        alpha_target: float = 0.2,
        alpha_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.alpha_target = alpha_target
        self.alpha_reg_weight = alpha_reg_weight

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
        combined = torch.cat([current, retrieved], dim=-1)
        alpha = self.gate_mlp(combined)
        fused = alpha * retrieved + (1.0 - alpha) * current
        return fused, alpha

    def regularization_loss(self, alpha: Tensor) -> Tensor:
        """Compute L2 regularization loss on gate alpha.

        Penalizes deviation of alpha from ``alpha_target``, encouraging
        the gate to use memory as a supplement rather than a replacement.

        Args:
            alpha: Gate values from the most recent forward pass,
                shape ``[B, L, 1]``.

        Returns:
            Scalar regularization loss (zero if ``alpha_reg_weight == 0``).
        """
        if self.alpha_reg_weight <= 0.0:
            return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)

        return self.alpha_reg_weight * (
            (alpha - self.alpha_target) ** 2
        ).mean()
