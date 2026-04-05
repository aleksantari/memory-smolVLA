"""Gates for fusing current features with retrieved memory.

Provides two strategies:
- ``SigmoidGate``: learned per-token scalar gate (``alpha * retrieved +
  (1-alpha) * current``).  Works well when gradient magnitude is high
  enough to train the gate, but collapses to alpha≈0 on low-loss
  datasets like LIBERO.
- ``ResidualGate``: simple additive fusion (``current + retrieved``).
  No learned gating — the model controls memory contribution through
  the upstream ``memory_proj`` weights instead.  Robust to weak
  gradients because there is no gate to collapse.
"""

import torch
from torch import Tensor, nn


class ResidualGate(nn.Module):
    """Additive memory fusion — no learned gate.

    Fuses via ``current + retrieved``.  Reports ``alpha = 1.0`` for
    compatibility with logging code that expects gate statistics.
    """

    def forward(
        self, current: Tensor, retrieved: Tensor
    ) -> tuple[Tensor, Tensor]:
        fused = current + retrieved
        alpha = torch.ones(
            (*current.shape[:-1], 1), device=current.device, dtype=current.dtype
        )
        return fused, alpha

    def regularization_loss(self, alpha: Tensor) -> Tensor:
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype)


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
