"""Consolidating memory bank for episodic memory storage.

A fixed-capacity ring buffer that stores intermediate VLM features
with timestamps. When capacity is exceeded, the two most similar
entries are merged (averaged) to make room — a simple form of
memory consolidation that preserves the most distinctive memories.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class ConsolidatingMemoryBank:
    """Fixed-capacity memory bank with nearest-neighbor consolidation.

    Stores ``(timestamp, tensor)`` entries where each tensor has shape
    ``[N_tokens, d_model]``. When the bank exceeds ``max_size``,
    the two most similar entries (by cosine similarity of mean-pooled
    representations) are merged: their tensors are averaged, and the
    newer timestamp is kept.

    Args:
        max_size: Maximum number of entries before consolidation triggers.
    """

    def __init__(self, max_size: int = 16) -> None:
        self.max_size = max_size
        self._entries: list[tuple[int, Tensor]] = []

    def write(self, tokens: Tensor, timestamp: int) -> None:
        """Append a new memory entry, consolidating if over capacity.

        Args:
            tokens: Feature tensor of shape ``[N_tokens, d_model]``.
                Will be detached and moved to CPU for storage.
            timestamp: Integer timestep within the current episode.
        """
        # Detach to prevent retaining the computation graph
        stored = tokens.detach().cpu()
        self._entries.append((timestamp, stored))

        if len(self._entries) > self.max_size:
            self.consolidate()

    def consolidate(self) -> None:
        """Merge the two most similar entries.

        Similarity is computed via cosine similarity on mean-pooled
        representations. The merged entry gets the average of both
        tensors and the newer (larger) timestamp.
        """
        if len(self._entries) < 2:
            return

        # Mean-pool each entry: [N_tokens, d_model] -> [d_model]
        pooled = torch.stack([entry.mean(dim=0) for _, entry in self._entries])
        # Normalize for cosine similarity
        pooled_norm = F.normalize(pooled, dim=-1)
        # Pairwise cosine similarity: [K, K]
        sim = pooled_norm @ pooled_norm.T

        # Mask diagonal (self-similarity = 1.0)
        k = sim.shape[0]
        sim.fill_diagonal_(-float("inf"))

        # Find the most similar pair
        flat_idx = sim.argmax().item()
        i, j = divmod(flat_idx, k)
        # Ensure i < j for consistent removal order
        if i > j:
            i, j = j, i

        ts_i, tensor_i = self._entries[i]
        ts_j, tensor_j = self._entries[j]

        # Merge: average tensors, keep newer timestamp
        merged_tensor = (tensor_i + tensor_j) / 2.0
        merged_timestamp = max(ts_i, ts_j)

        # Remove both (j first since j > i)
        del self._entries[j]
        del self._entries[i]

        # Insert merged entry
        self._entries.append((merged_timestamp, merged_tensor))

    def read_all(self, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Return all stored memories as batched tensors.

        Args:
            device: Device to move tensors to.

        Returns:
            Tuple of:
                - memories: ``[K, N_tokens, d_model]`` stacked feature tensors
                - timestamps: ``[K]`` integer timestamps

        Raises:
            RuntimeError: If the bank is empty.
        """
        if len(self._entries) == 0:
            raise RuntimeError("Memory bank is empty. Write entries before reading.")

        timestamps = torch.tensor(
            [ts for ts, _ in self._entries], dtype=torch.long, device=device
        )
        memories = torch.stack([t.to(device) for _, t in self._entries])

        return memories, timestamps

    def reset(self) -> None:
        """Clear all entries. Call at episode boundaries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"ConsolidatingMemoryBank(max_size={self.max_size}, "
            f"current_size={len(self._entries)})"
        )
