"""Multi-scale memory bank for capturing context at different timescales.

Instead of a single bank, maintains 2-3 banks at different temporal
resolutions:
- **Short-term** (small, recent): Last few frames at full resolution.
- **Medium-term** (medium, compressed): Recent history, pooled.
- **Long-term** (large, heavily compressed): Full episode context.

Each scale uses FIFO eviction independently.  During retrieval, entries
from all scales are concatenated and attended over jointly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from memory_smolvla.memory.bank import MemoryBank


class MultiScaleMemoryBank:
    """Multi-scale memory with short, medium, and long-term banks.

    Args:
        short_size: Max entries in short-term bank (recent, full-res).
        medium_size: Max entries in medium-term bank.
        long_size: Max entries in long-term bank.
        medium_pool_factor: Mean-pool tokens by this factor for medium bank.
        long_pool_factor: Mean-pool tokens by this factor for long bank.
        eviction: Eviction strategy for all sub-banks.
    """

    def __init__(
        self,
        short_size: int = 4,
        medium_size: int = 8,
        long_size: int = 16,
        medium_pool_factor: int = 4,
        long_pool_factor: int = 16,
        eviction: str = "fifo",
    ) -> None:
        self.short = MemoryBank(max_size=short_size, eviction=eviction)
        self.medium = MemoryBank(max_size=medium_size, eviction=eviction)
        self.long = MemoryBank(max_size=long_size, eviction=eviction)

        self.medium_pool_factor = medium_pool_factor
        self.long_pool_factor = long_pool_factor

    def write(self, tokens: Tensor, timestamp: int) -> None:
        """Write to all three scales with appropriate compression.

        Args:
            tokens: Feature tensor ``[N_tokens, d_model]``.
            timestamp: Integer timestep.
        """
        # Short-term: full resolution
        self.short.write(tokens, timestamp)

        # Medium-term: pool along token dimension
        medium_tokens = self._pool_tokens(tokens, self.medium_pool_factor)
        self.medium.write(medium_tokens, timestamp)

        # Long-term: heavy pooling
        long_tokens = self._pool_tokens(tokens, self.long_pool_factor)
        self.long.write(long_tokens, timestamp)

    def read_all(
        self, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor]:
        """Read from all scales, concatenating along the entry dimension.

        Returns memories and timestamps from all non-empty banks,
        flattened into a single sequence suitable for cross-attention.

        Returns:
            Tuple of:
                - memories: ``[K_total, N_tokens_max, d_model]`` (padded)
                - timestamps: ``[K_total]``

        Raises:
            RuntimeError: If all banks are empty.
        """
        all_memories = []
        all_timestamps = []

        for bank in (self.short, self.medium, self.long):
            if len(bank) > 0:
                memories, timestamps = bank.read_all(device=device)
                all_memories.append(memories)
                all_timestamps.append(timestamps)

        if not all_memories:
            raise RuntimeError("All memory banks are empty.")

        # Pad token dimensions to match the largest and concatenate
        max_tokens = max(m.shape[1] for m in all_memories)
        d_model = all_memories[0].shape[2]

        padded = []
        for mem in all_memories:
            if mem.shape[1] < max_tokens:
                pad = torch.zeros(
                    mem.shape[0],
                    max_tokens - mem.shape[1],
                    d_model,
                    device=device,
                    dtype=mem.dtype,
                )
                mem = torch.cat([mem, pad], dim=1)
            padded.append(mem)

        combined_memories = torch.cat(padded, dim=0)
        combined_timestamps = torch.cat(all_timestamps, dim=0)

        return combined_memories, combined_timestamps

    def reset(self) -> None:
        """Clear all banks. Call at episode boundaries."""
        self.short.reset()
        self.medium.reset()
        self.long.reset()

    @staticmethod
    def _pool_tokens(tokens: Tensor, factor: int) -> Tensor:
        """Mean-pool tokens along the token dimension by ``factor``.

        If the token count is not divisible by factor, the remainder
        tokens are averaged into a final slot.
        """
        N, D = tokens.shape
        if factor >= N:
            return tokens.mean(dim=0, keepdim=True)

        # Pool full groups
        n_full = (N // factor) * factor
        pooled = tokens[:n_full].reshape(-1, factor, D).mean(dim=1)

        # Handle remainder
        if n_full < N:
            remainder = tokens[n_full:].mean(dim=0, keepdim=True)
            pooled = torch.cat([pooled, remainder], dim=0)

        return pooled

    def __len__(self) -> int:
        return len(self.short) + len(self.medium) + len(self.long)

    def __repr__(self) -> str:
        return (
            f"MultiScaleMemoryBank("
            f"short={len(self.short)}/{self.short.max_size}, "
            f"medium={len(self.medium)}/{self.medium.max_size}, "
            f"long={len(self.long)}/{self.long.max_size})"
        )
