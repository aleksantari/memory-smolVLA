"""Grouped-episode loader for full-sequence memory training.

Yields batches of ``B = G * group_size`` frames laid out as ``G`` groups
of ``group_size`` contiguous frames, each group drawn from a single
episode at a random starting offset. Per-batch metadata arrays
``episode_ids`` and ``timesteps`` (length ``B``) tell the memory bank
which entries belong to which episode and what their temporal positions
are.

See ``memory_smolvla_implementation_spec.md`` §4 and
``FullSeqMemBank.process_batch`` for how the memory bank consumes this
layout.
"""

from __future__ import annotations

import logging
import random
from typing import Iterator

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from memory_smolvla.data.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


class GroupedEpisodeLoader:
    """Yields batches of ``G`` episodes × ``group_size`` contiguous frames.

    Within each batch, positions ``[k*group_size : (k+1)*group_size)``
    come from the same episode with monotonically increasing frame
    indices. Groups are drawn from distinct episodes; when the set of
    episodes is exhausted it is reshuffled and reused.

    Args:
        cfg: Dataset configuration.
        group_size: Number of contiguous frames per group.
        num_groups: Number of groups per batch (``G``).
        mem_length: Memory bank capacity; the loader requires
            ``group_size >= mem_length`` so the bank fills during training.
        shuffle_episodes: Whether to shuffle the episode pool each pass.

    Yields:
        Dict with the same keys as a single LeRobot frame, plus:
        - ``episode_ids``: ``list[int]`` of length ``B``.
        - ``timesteps``: ``list[int]`` of length ``B``.
        All tensor keys are stacked along dim 0 to produce ``(B, ...)``.
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        group_size: int,
        num_groups: int,
        mem_length: int = 1,
        shuffle_episodes: bool = True,
    ) -> None:
        if not cfg.repo_ids:
            raise ValueError("DatasetConfig.repo_ids must contain at least one entry.")
        if group_size < mem_length:
            raise ValueError(
                f"group_size ({group_size}) must be >= mem_length ({mem_length}) "
                f"so the memory bank fills during training."
            )
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")

        self._datasets: list[LeRobotDataset] = []
        for repo_id in cfg.repo_ids:
            ds = LeRobotDataset(
                repo_id=repo_id,
                delta_timestamps=cfg.delta_timestamps or None,
                root=cfg.local_cache_dir,
            )
            self._datasets.append(ds)
            logger.info(
                "Loaded dataset %s: %d episodes, %d frames",
                repo_id, ds.num_episodes, len(ds),
            )

        self._group_size = group_size
        self._num_groups = num_groups
        self._shuffle = shuffle_episodes

        # Build the global episode pool. Each entry is
        # (global_episode_id, dataset_index, from_index, to_index).
        self._episode_pool: list[tuple[int, int, int, int]] = []
        global_eid = 0
        for ds_idx, ds in enumerate(self._datasets):
            for ep_idx in range(ds.num_episodes):
                ep_meta = ds.meta.episodes[ep_idx]
                start = int(ep_meta["dataset_from_index"])
                end = int(ep_meta["dataset_to_index"])
                if end - start >= group_size:
                    self._episode_pool.append((global_eid, ds_idx, start, end))
                global_eid += 1

        if not self._episode_pool:
            raise ValueError(
                f"No episodes with length >= group_size ({group_size}) "
                f"in any of the provided datasets."
            )

        logger.info(
            "GroupedEpisodeLoader: %d usable episodes, group_size=%d, num_groups=%d, "
            "batch_size=%d",
            len(self._episode_pool), group_size, num_groups, group_size * num_groups,
        )

    @property
    def batch_size(self) -> int:
        return self._group_size * self._num_groups

    @property
    def total_episodes(self) -> int:
        return len(self._episode_pool)

    @property
    def total_frames(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        pool_order: list[int] = []
        cursor = 0
        while True:
            if cursor + self._num_groups > len(pool_order):
                pool_order = list(range(len(self._episode_pool)))
                if self._shuffle:
                    random.shuffle(pool_order)
                cursor = 0

            group_indices = pool_order[cursor : cursor + self._num_groups]
            cursor += self._num_groups

            yield self._build_batch(group_indices)

    def _build_batch(self, group_indices: list[int]) -> dict:
        frames: list[dict] = []
        episode_ids: list[int] = []
        timesteps: list[int] = []

        for pool_idx in group_indices:
            global_eid, ds_idx, start, end = self._episode_pool[pool_idx]
            max_offset = end - start - self._group_size
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
            ds = self._datasets[ds_idx]

            for k in range(self._group_size):
                frame = ds[start + offset + k]
                frames.append(frame)
                episode_ids.append(global_eid)
                timesteps.append(offset + k)

        batch = _collate(frames)
        batch["episode_ids"] = episode_ids
        batch["timesteps"] = timesteps
        return batch


def _collate(frames: list[dict]) -> dict:
    """Stack tensor fields along dim 0; collect non-tensor fields into lists."""
    if not frames:
        return {}

    out: dict = {}
    keys = frames[0].keys()
    for key in keys:
        vals = [f[key] for f in frames]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
