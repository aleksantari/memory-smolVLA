"""Grouped-episode loader for full-sequence memory training on LIBERO.

Yields batches of ``B = G * group_size`` frames laid out as ``G`` groups
of ``group_size`` contiguous frames, each group drawn from a single
episode at a random starting offset. Per-batch metadata arrays
``episode_ids`` and ``timesteps`` (length ``B``) tell the memory bank
which entries belong to which episode and what their temporal positions
are.

The underlying dataset is hardcoded to ``HuggingFaceVLA/libero`` to
mirror the baseline v2 training setup.

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
from lerobot.datasets.transforms import ImageTransforms

from memory_smolvla.data.dataset_config import LIBERO_REPO_ID, DatasetConfig

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
        image_transforms: Optional torchvision transform pipeline applied
            to image keys by :class:`LeRobotDataset`. Pass
            :class:`lerobot.datasets.transforms.ImageTransforms` built
            from the baseline-v2 ``ImageTransformsConfig`` to match the
            reference augmentation regime.
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
        image_transforms: ImageTransforms | None = None,
        shuffle_episodes: bool = True,
    ) -> None:
        if group_size < mem_length:
            raise ValueError(
                f"group_size ({group_size}) must be >= mem_length ({mem_length}) "
                f"so the memory bank fills during training."
            )
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")

        self._dataset = LeRobotDataset(
            repo_id=LIBERO_REPO_ID,
            delta_timestamps=cfg.delta_timestamps or None,
            root=cfg.local_cache_dir,
            image_transforms=image_transforms,
        )
        logger.info(
            "Loaded dataset %s: %d episodes, %d frames",
            LIBERO_REPO_ID, self._dataset.num_episodes, len(self._dataset),
        )

        self._group_size = group_size
        self._num_groups = num_groups
        self._shuffle = shuffle_episodes

        # Episode pool: (episode_id, from_index, to_index).
        self._episode_pool: list[tuple[int, int, int]] = []
        for ep_idx in range(self._dataset.num_episodes):
            ep_meta = self._dataset.meta.episodes[ep_idx]
            start = int(ep_meta["dataset_from_index"])
            end = int(ep_meta["dataset_to_index"])
            if end - start >= group_size:
                self._episode_pool.append((ep_idx, start, end))

        if not self._episode_pool:
            raise ValueError(
                f"No episodes with length >= group_size ({group_size}) "
                f"in {LIBERO_REPO_ID}."
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
        return len(self._dataset)

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
            eid, start, end = self._episode_pool[pool_idx]
            max_offset = end - start - self._group_size
            offset = random.randint(0, max_offset) if max_offset > 0 else 0

            for k in range(self._group_size):
                frame = self._dataset[start + offset + k]
                frames.append(frame)
                episode_ids.append(eid)
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
