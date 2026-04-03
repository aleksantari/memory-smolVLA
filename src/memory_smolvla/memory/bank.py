"""Fixed-capacity memory bank for episodic memory storage.

Supports two eviction strategies:
- ``"fifo"`` (default): Simple ring buffer — drops the oldest entry when
  full. Preserves temporal structure and avoids creating ghost states.
- ``"consolidate"``: Merges the two most similar entries (averaged) to
  make room. Preserves distinctive memories but blurs temporal precision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class MemoryBank:
    """Fixed-capacity memory bank with configurable eviction.

    Stores ``(timestamp, tensor)`` entries where each tensor has shape
    ``[N_tokens, d_model]``. When the bank exceeds ``max_size``, the
    eviction strategy determines which entry is dropped.

    Args:
        max_size: Maximum number of entries before eviction triggers.
        eviction: Eviction strategy — ``"fifo"`` (oldest-out ring buffer)
            or ``"consolidate"`` (merge most-similar pair).
    """

    _VALID_STRATEGIES = frozenset({"fifo", "consolidate"})

    def __init__(
        self, max_size: int = 16, eviction: str = "fifo"
    ) -> None:
        if eviction not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Invalid eviction strategy {eviction!r}. "
                f"Must be one of {sorted(self._VALID_STRATEGIES)}."
            )
        self.max_size = max_size
        self.eviction = eviction
        self._entries: list[tuple[int, Tensor]] = []

    def write(self, tokens: Tensor, timestamp: int) -> None:
        """Append a new memory entry, evicting if over capacity.

        Args:
            tokens: Feature tensor of shape ``[N_tokens, d_model]``.
                Will be detached and moved to CPU for storage.
            timestamp: Integer timestep within the current episode.
        """
        stored = tokens.detach().cpu()
        self._entries.append((timestamp, stored))

        if len(self._entries) > self.max_size:
            self._evict()

    def _evict(self) -> None:
        """Remove one entry according to the eviction strategy."""
        if self.eviction == "fifo":
            self._evict_fifo()
        else:
            self._evict_consolidate()

    def _evict_fifo(self) -> None:
        """Drop the oldest (first) entry."""
        if self._entries:
            del self._entries[0]

    def _evict_consolidate(self) -> None:
        """Merge the two most similar entries.

        Similarity is computed via cosine similarity on mean-pooled
        representations. The merged entry gets the average of both
        tensors and the newer (larger) timestamp.
        """
        if len(self._entries) < 2:
            return

        pooled = torch.stack([entry.mean(dim=0) for _, entry in self._entries])
        pooled_norm = F.normalize(pooled, dim=-1)
        sim = pooled_norm @ pooled_norm.T

        k = sim.shape[0]
        sim.fill_diagonal_(-float("inf"))

        flat_idx = sim.argmax().item()
        i, j = divmod(flat_idx, k)
        if i > j:
            i, j = j, i

        ts_i, tensor_i = self._entries[i]
        ts_j, tensor_j = self._entries[j]

        merged_tensor = (tensor_i + tensor_j) / 2.0
        merged_timestamp = max(ts_i, ts_j)

        del self._entries[j]
        del self._entries[i]

        # Insert at position i to preserve approximate temporal order
        self._entries.insert(i, (merged_timestamp, merged_tensor))

    def read_all(self, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Return all stored memories as batched tensors.

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
            f"MemoryBank(max_size={self.max_size}, "
            f"eviction={self.eviction!r}, "
            f"current_size={len(self._entries)})"
        )


# Backward compatibility alias
ConsolidatingMemoryBank = MemoryBank
