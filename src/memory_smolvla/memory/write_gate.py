"""Selective write gate for memory bank.

Decides whether the current frame is worth memorizing by comparing
it to recent memory content.  Frames that are very similar to what
is already stored (e.g. static camera, no scene change) get a low
write score and are skipped.  This keeps the bank focused on
task-relevant transitions and reduces consolidation pressure.

During training the gate produces a soft probability so gradients
can flow.  During inference a hard threshold is applied.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WriteGate(nn.Module):
    """Learned gate that decides when to write to the memory bank.

    Takes the current (compressed) features and the mean of recently
    stored memory entries, and produces a scalar write probability.

    Args:
        d_model: Dimension of feature vectors.
        threshold: Hard threshold for inference-time write decisions.
            Features with write probability below this are not stored.
    """

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current: Tensor,
        recent_memory: Tensor | None,
    ) -> Tensor:
        """Compute write probability for the current features.

        Args:
            current: Current (possibly compressed) features,
                ``[B, N, D]`` — mean-pooled internally to ``[B, D]``.
            recent_memory: Mean-pooled representation of recent bank
                entries, ``[B, D]``.  ``None`` if the bank is empty
                (first timestep), in which case write probability is 1.

        Returns:
            Write probability, ``[B, 1]`` in ``[0, 1]``.
        """
        # Mean-pool current features to a single vector
        current_pooled = current.mean(dim=1)  # [B, D]

        if recent_memory is None:
            return torch.ones(
                current_pooled.shape[0], 1,
                device=current_pooled.device,
                dtype=current_pooled.dtype,
            )

        combined = torch.cat([current_pooled, recent_memory], dim=-1)
        return self.gate_mlp(combined)  # [B, 1]

    def should_write(
        self,
        current: Tensor,
        recent_memory: Tensor | None,
    ) -> tuple[bool, Tensor]:
        """Compute write decision (hard threshold for inference).

        Returns:
            Tuple of (should_write bool, write_prob tensor).
        """
        prob = self.forward(current, recent_memory)
        return bool((prob > self.threshold).any().item()), prob
