"""Episode-sequential data loader for memory-augmented training.

Wraps one or more ``LeRobotDataset`` instances and yields frames in
strict temporal order within each episode. Between episodes an
``EpisodeBoundary`` sentinel is emitted so the trainer can call
``policy.reset_memory()``.

This is necessary because the memory bank accumulates hidden-state
snapshots across timesteps — random frame sampling would break the
temporal structure the memory system depends on.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from memory_smolvla.data.dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeBoundary:
    """Sentinel emitted between episodes.

    The trainer uses this to call ``policy.reset_memory()`` and flush
    any pending gradient accumulation before the next episode begins.

    Attributes:
        episode_index: The zero-based index of the episode that just
            finished (within the originating dataset).
        dataset_index: Index into the list of datasets (relevant when
            multiple datasets are concatenated).
    """

    episode_index: int
    dataset_index: int


class EpisodeSequentialLoader:
    """Yields frames in temporal order, with episode boundary sentinels.

    Iterates over all episodes (optionally in shuffled order), yielding
    every frame of each episode sequentially before moving to the next.
    Between episodes an :class:`EpisodeBoundary` sentinel is emitted.

    Args:
        cfg: Dataset configuration specifying repo IDs, delta timestamps,
            split, and optional local cache directory.
        shuffle_episodes: If ``True``, the episode visitation order is
            randomized at each full pass through the data. Frames within
            each episode are always in temporal order.
        repeat: If ``True``, loop indefinitely (standard training mode).
            If ``False``, stop after one full pass through all episodes.
        max_window_size: If set, each visit to an episode yields at
            most this many consecutive frames starting from a random
            offset within the episode, then emits ``EpisodeBoundary``
            and moves to the next episode. With ``max_window_size``
            comparable to ``grad_accum_steps``, successive optimizer
            steps see different episodes — recovers most of the
            random-batch gradient diversity that the default
            full-episode loader gives up. ``None`` (default) yields
            the entire episode (legacy behavior).
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        shuffle_episodes: bool = True,
        repeat: bool = True,
        max_window_size: int | None = None,
    ) -> None:
        if not cfg.repo_ids:
            raise ValueError("DatasetConfig.repo_ids must contain at least one entry.")
        if max_window_size is not None and max_window_size < 1:
            raise ValueError(
                f"max_window_size must be >= 1 or None, got {max_window_size}"
            )

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
                repo_id,
                ds.num_episodes,
                len(ds),
            )

        self._shuffle_episodes = shuffle_episodes
        self._repeat = repeat
        self._max_window_size = max_window_size

    # ------------------------------------------------------------------
    # Public iteration API
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict | EpisodeBoundary]:
        """Yield frames and episode-boundary sentinels indefinitely."""
        while True:
            yield from self._one_pass()
            if not self._repeat:
                break

    def _one_pass(self) -> Iterator[dict | EpisodeBoundary]:
        """One full pass through all episodes across all datasets."""
        # Build list of (dataset_index, episode_index) pairs
        episode_list: list[tuple[int, int]] = []
        for ds_idx, ds in enumerate(self._datasets):
            for ep_idx in range(ds.num_episodes):
                episode_list.append((ds_idx, ep_idx))

        if self._shuffle_episodes:
            random.shuffle(episode_list)

        for ds_idx, ep_idx in episode_list:
            yield from self._yield_episode(ds_idx, ep_idx)
            yield EpisodeBoundary(episode_index=ep_idx, dataset_index=ds_idx)

    def _yield_episode(
        self, ds_idx: int, ep_idx: int
    ) -> Iterator[dict]:
        """Yield frames of a single episode in temporal order.

        If ``max_window_size`` is set, truncate the visit to a window
        of that many consecutive frames starting from a random offset
        within the episode. This forces successive optimizer steps to
        see different episodes — the cross-episode diversity that the
        default full-episode visit gives up. The bank still fills
        sequentially within each window, preserving temporal-order
        semantics the memory pathway depends on.
        """
        ds = self._datasets[ds_idx]
        ep_meta = ds.meta.episodes[ep_idx]
        start = int(ep_meta["dataset_from_index"])
        end = int(ep_meta["dataset_to_index"])

        if self._max_window_size is not None:
            ep_len = end - start
            if ep_len > self._max_window_size:
                # Random offset so we don't only train on episode openings.
                offset = random.randint(0, ep_len - self._max_window_size)
                start = start + offset
                end = start + self._max_window_size

        for frame_idx in range(start, end):
            item = ds[frame_idx]
            # Add batch dimension so the trainer can call policy.forward()
            # directly without a DataLoader collation step.
            # Skip scalar tensors (e.g. episode_index) — they are metadata,
            # not batched model inputs.
            yield {
                key: val.unsqueeze(0) if isinstance(val, torch.Tensor) and val.ndim >= 1 else val
                for key, val in item.items()
            }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def total_episodes(self) -> int:
        """Total number of episodes across all datasets."""
        return sum(ds.num_episodes for ds in self._datasets)

    @property
    def total_frames(self) -> int:
        """Total number of frames across all datasets."""
        return sum(len(ds) for ds in self._datasets)

    def __repr__(self) -> str:
        repo_ids = [ds.repo_id for ds in self._datasets]
        return (
            f"EpisodeSequentialLoader("
            f"datasets={repo_ids}, "
            f"total_episodes={self.total_episodes}, "
            f"total_frames={self.total_frames}, "
            f"shuffle_episodes={self._shuffle_episodes})"
        )
