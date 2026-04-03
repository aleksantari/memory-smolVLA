"""GRU-based working memory as an alternative to episodic memory.

Instead of storing and retrieving from an explicit memory bank, this
module maintains a fixed-size recurrent state vector that is updated
at each timestep via a GRU.  The state is then projected and fused
into the VLM prefix via the same sigmoid gate.

Advantages over episodic memory:
- Simpler — no bank, no consolidation, no cross-attention retrieval.
- Allows truncated BPTT within episodes for learning temporal dynamics.
- Constant memory and compute cost regardless of episode length.
- No train/inference temporal mismatch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WorkingMemory(nn.Module):
    """GRU-based working memory for temporal context.

    At each timestep, the mean-pooled current prefix is fed as input
    to a GRU cell whose hidden state persists across timesteps within
    an episode.  The hidden state is then broadcast and projected to
    produce a memory signal for each token position.

    Args:
        d_model: VLM hidden dimension.
        state_dim: GRU hidden state dimension.  Defaults to ``d_model``.
        n_layers: Number of stacked GRU layers.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int | None = None,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim or d_model

        self.input_proj = nn.Linear(d_model, self.state_dim)
        self.gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.state_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output_proj = nn.Linear(self.state_dim, d_model)

        # Zero-init output projection so working memory starts as identity
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self._hidden: Tensor | None = None
        self._n_layers = n_layers

    def forward(self, prefix_hidden: Tensor) -> Tensor:
        """Update state with current features and return memory signal.

        Args:
            prefix_hidden: Current VLM prefix, ``[B, L, D]``.

        Returns:
            Memory signal, ``[B, L, D]``, broadcast from the GRU state.
        """
        B, L, D = prefix_hidden.shape
        device = prefix_hidden.device
        dtype = prefix_hidden.dtype

        # Initialize hidden state if needed
        if self._hidden is None or self._hidden.shape[1] != B:
            self._hidden = torch.zeros(
                self._n_layers, B, self.state_dim,
                device=device, dtype=dtype,
            )
        self._hidden = self._hidden.to(device=device, dtype=dtype)

        # Mean-pool current prefix → [B, D] → [B, 1, state_dim]
        pooled = prefix_hidden.mean(dim=1)
        gru_input = self.input_proj(pooled).unsqueeze(1)

        # GRU update: [B, 1, state_dim], [n_layers, B, state_dim]
        gru_out, self._hidden = self.gru(gru_input, self._hidden)

        # Project and broadcast: [B, 1, state_dim] → [B, 1, D] → [B, L, D]
        memory_signal = self.output_proj(gru_out)
        memory_signal = memory_signal.expand(-1, L, -1)

        return memory_signal

    def reset(self) -> None:
        """Clear hidden state at episode boundaries."""
        self._hidden = None

    def detach_state(self) -> None:
        """Detach hidden state from computation graph (for TBPTT)."""
        if self._hidden is not None:
            self._hidden = self._hidden.detach()
