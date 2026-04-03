"""Sinusoidal temporal positional encoding for memory entries.

Encodes time deltas (how many steps ago a memory was stored) into
d_model-dimensional vectors using sin/cos with log-spaced frequencies.
This lets the cross-attention retrieval module distinguish recent
from distant memories.
"""

import math

import torch
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for arbitrary time deltas.

    Uses the same log-spaced frequency band approach as SmolVLA's
    flow-matching timestep encoding, but with default period ranges
    appropriate for temporal distances (1 to 1000 steps).

    Args:
        d_model: Dimension of the encoding vectors.
        min_period: Minimum wavelength (highest frequency). Controls
            sensitivity to small time differences.
        max_period: Maximum wavelength (lowest frequency). Controls
            the range of temporal distances that can be distinguished.
    """

    def __init__(
        self,
        d_model: int,
        min_period: float = 1.0,
        max_period: float = 1000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.min_period = min_period
        self.max_period = max_period

        # Precompute log-spaced frequencies: shape [d_model // 2]
        half_dim = d_model // 2
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for sinusoidal PE, got {d_model}."
            )
        freq_exponents = torch.linspace(0.0, 1.0, half_dim)
        # Log-space interpolation between min_period and max_period
        periods = min_period * (max_period / min_period) ** freq_exponents
        # frequencies = 2 * pi / period
        freqs = 2.0 * math.pi / periods
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, time_deltas: Tensor) -> Tensor:
        """Compute temporal positional encoding for given time deltas.

        Args:
            time_deltas: Time deltas of arbitrary shape ``[*]``,
                e.g. ``[K]`` for K memory entries or ``[B, K]`` for batched.

        Returns:
            Positional encoding of shape ``[*, d_model]``.
        """
        # time_deltas: [*] -> [*, 1] * [half_dim] -> [*, half_dim]
        angles = time_deltas.unsqueeze(-1).float() * self.freqs
        # Interleave sin and cos: [*, d_model]
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return pe.to(time_deltas.device)
