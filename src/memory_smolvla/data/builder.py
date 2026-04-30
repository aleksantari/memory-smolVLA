"""Factory for building data loaders in the correct mode.

Two loading strategies are supported:

- ``"sequential"`` — episode-sequential via :class:`EpisodeSequentialLoader`.
  Required for any training mode that uses the memory bank, since frames
  must arrive in temporal order within each episode.

- ``"random"`` — standard random-batch sampling via PyTorch ``DataLoader``
  with ``EpisodeAwareSampler``. Used for ``expert_only_scratch`` where
  there is no memory state to maintain.
"""

from __future__ import annotations

import logging
from typing import Union

from torch.utils.data import ConcatDataset, DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler

from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.data.episode_loader import EpisodeSequentialLoader

logger = logging.getLogger(__name__)


def build_dataloader(
    cfg: DatasetConfig,
    mode: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_episodes: bool = True,
    max_window_size: int | None = None,
) -> Union[EpisodeSequentialLoader, DataLoader]:
    """Return the appropriate data loader for the requested mode.

    Args:
        cfg: Dataset configuration (repo IDs, delta timestamps, split,
            optional local cache dir).
        mode: ``"sequential"`` for episode-ordered loading (memory
            training), or ``"random"`` for shuffled batch loading
            (expert-only baseline).
        batch_size: Batch size for ``"random"`` mode (ignored in
            ``"sequential"`` mode, which always yields B=1 frames).
        num_workers: Worker count for ``"random"`` DataLoader.
        shuffle_episodes: Whether to randomise the episode visitation
            order in ``"sequential"`` mode.
        max_window_size: Sequential-mode only. If set, each visit to
            an episode yields at most this many consecutive frames
            from a random offset, then moves to the next episode.
            Enables cross-episode diversity within ``grad_accum_steps``
            optimizer steps.

    Returns:
        An :class:`EpisodeSequentialLoader` for ``"sequential"`` mode,
        or a ``torch.utils.data.DataLoader`` for ``"random"`` mode.

    Raises:
        ValueError: If ``mode`` is not ``"sequential"`` or ``"random"``.
    """
    if mode not in ("sequential", "random"):
        raise ValueError(f"mode must be 'sequential' or 'random', got {mode!r}")

    if mode == "sequential":
        return EpisodeSequentialLoader(
            cfg,
            shuffle_episodes=shuffle_episodes,
            max_window_size=max_window_size,
        )

    # --- random batch mode ---
    datasets = []
    for repo_id in cfg.repo_ids:
        ds = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=cfg.delta_timestamps or None,
            root=cfg.local_cache_dir,
        )
        datasets.append(ds)
        logger.info("Loaded dataset %s: %d frames", repo_id, len(ds))

    if len(datasets) == 1:
        combined = datasets[0]
        from_indices = [
            int(combined.meta.episodes[i]["dataset_from_index"])
            for i in range(combined.num_episodes)
        ]
        to_indices = [
            int(combined.meta.episodes[i]["dataset_to_index"])
            for i in range(combined.num_episodes)
        ]
    else:
        combined = ConcatDataset(datasets)
        # Build global from/to indices across concatenated datasets
        offset = 0
        from_indices, to_indices = [], []
        for ds in datasets:
            for i in range(ds.num_episodes):
                ep = ds.meta.episodes[i]
                from_indices.append(int(ep["dataset_from_index"]) + offset)
                to_indices.append(int(ep["dataset_to_index"]) + offset)
            offset += len(ds)

    sampler = EpisodeAwareSampler(
        dataset_from_indices=from_indices,
        dataset_to_indices=to_indices,
        shuffle=True,
    )

    return DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
