"""Grouped-episode loader for full-sequence memory training on LIBERO.

Yields batches of ``B = G * group_size`` frames laid out as ``G`` groups
of ``group_size`` contiguous frames, each group drawn from a single
episode at a random starting offset. Per-batch metadata arrays
``episode_ids`` and ``timesteps`` (length ``B``) tell the memory bank
which entries belong to which episode and what their temporal positions
are.

Wraps a :class:`torch.utils.data.DataLoader` around an
:class:`IterableDataset` that produces whole batches. ``num_workers``
processes build batches in parallel; pinned memory + prefetch keep the
GPU fed so training is compute-bound rather than I/O-bound.

The underlying dataset is hardcoded to ``HuggingFaceVLA/libero`` to
mirror the baseline v2 training setup.

See ``memory_smolvla_implementation_spec.md`` §4 and
``FullSeqMemBank.process_batch`` for how the memory bank consumes this
layout.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import ImageTransforms

from memory_smolvla.data.dataset_config import LIBERO_REPO_ID, DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class _EpisodeRef:
    pool_idx: int
    episode_id: int
    from_idx: int
    to_idx: int


class _GroupBatchIterable(IterableDataset):
    """Per-worker iterable that yields fully-assembled batch dicts.

    Each worker lazily constructs its own :class:`LeRobotDataset` on
    first iteration so that parquet/video handles are not forked from
    the parent. The episode pool (start/end indices) is computed once
    in the parent and passed down — cheap metadata only.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        root: str | None,
        delta_timestamps: dict | None,
        image_transforms: ImageTransforms | None,
        episode_pool: list[_EpisodeRef],
        group_size: int,
        num_groups: int,
        shuffle: bool,
        base_seed: int,
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._root = root
        self._delta_timestamps = delta_timestamps
        self._image_transforms = image_transforms
        self._episode_pool = episode_pool
        self._group_size = group_size
        self._num_groups = num_groups
        self._shuffle = shuffle
        self._base_seed = base_seed

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self._base_seed + worker_id * 9973)

        dataset = LeRobotDataset(
            repo_id=self._repo_id,
            delta_timestamps=self._delta_timestamps,
            root=self._root,
            image_transforms=self._image_transforms,
        )

        pool_order: list[int] = []
        cursor = 0
        while True:
            if cursor + self._num_groups > len(pool_order):
                pool_order = list(range(len(self._episode_pool)))
                if self._shuffle:
                    rng.shuffle(pool_order)
                cursor = 0

            group_indices = pool_order[cursor : cursor + self._num_groups]
            cursor += self._num_groups
            yield self._build_batch(dataset, group_indices, rng)

    def _build_batch(
        self, dataset: LeRobotDataset, group_indices: list[int], rng: random.Random,
    ) -> dict:
        frames: list[dict] = []
        episode_ids: list[int] = []
        timesteps: list[int] = []

        for pool_idx in group_indices:
            ref = self._episode_pool[pool_idx]
            max_offset = ref.to_idx - ref.from_idx - self._group_size
            offset = rng.randint(0, max_offset) if max_offset > 0 else 0

            for k in range(self._group_size):
                frame = dataset[ref.from_idx + offset + k]
                frames.append(frame)
                episode_ids.append(ref.episode_id)
                timesteps.append(offset + k)

        batch = _collate(frames)
        batch["episode_ids"] = episode_ids
        batch["timesteps"] = timesteps
        return batch


def _identity_collate(x):
    """No-op collate: the IterableDataset already yields complete batches."""
    return x


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
            to image keys by :class:`LeRobotDataset`.
        shuffle_episodes: Whether to shuffle the episode pool each pass.
        policy_config: If provided, auto-resolves missing ``delta_timestamps``
            keys (notably ``action``) from policy + dataset fps.
        num_workers: Number of DataLoader worker processes. ``0`` keeps
            batch construction in the main thread.
        prefetch_factor: Per-worker prefetch depth (ignored when
            ``num_workers == 0``).
        pin_memory: Whether the DataLoader should pin host memory for
            faster H→D transfers.
        seed: RNG seed base for per-worker ``random.Random`` instances.

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
        policy_config=None,
        num_workers: int = 0,
        prefetch_factor: int = 4,
        pin_memory: bool = True,
        seed: int = 1000,
    ) -> None:
        if group_size < mem_length:
            raise ValueError(
                f"group_size ({group_size}) must be >= mem_length ({mem_length}) "
                f"so the memory bank fills during training."
            )
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}.")

        delta_timestamps = self._resolve_deltas(cfg, policy_config)

        meta = LeRobotDatasetMetadata(LIBERO_REPO_ID, root=cfg.local_cache_dir)
        episode_pool: list[_EpisodeRef] = []
        for ep_idx in range(meta.total_episodes):
            ep_meta = meta.episodes[ep_idx]
            start = int(ep_meta["dataset_from_index"])
            end = int(ep_meta["dataset_to_index"])
            if end - start >= group_size:
                episode_pool.append(
                    _EpisodeRef(pool_idx=len(episode_pool), episode_id=ep_idx,
                                from_idx=start, to_idx=end),
                )
        if not episode_pool:
            raise ValueError(
                f"No episodes with length >= group_size ({group_size}) "
                f"in {LIBERO_REPO_ID}."
            )

        total_frames = sum(ref.to_idx - ref.from_idx for ref in episode_pool)
        logger.info(
            "Loaded dataset %s: %d episodes (usable for group_size=%d), %d frames",
            LIBERO_REPO_ID, len(episode_pool), group_size, total_frames,
        )

        self._iterable = _GroupBatchIterable(
            repo_id=LIBERO_REPO_ID,
            root=cfg.local_cache_dir,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            episode_pool=episode_pool,
            group_size=group_size,
            num_groups=num_groups,
            shuffle=shuffle_episodes,
            base_seed=seed,
        )
        self._episode_pool = episode_pool
        self._total_frames = total_frames
        self._group_size = group_size
        self._num_groups = num_groups
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory

        logger.info(
            "GroupedEpisodeLoader: %d usable episodes, group_size=%d, num_groups=%d, "
            "batch_size=%d, num_workers=%d, prefetch_factor=%d, pin_memory=%s",
            len(episode_pool), group_size, num_groups, group_size * num_groups,
            num_workers, prefetch_factor, pin_memory,
        )

    @staticmethod
    def _resolve_deltas(cfg: DatasetConfig, policy_config) -> dict | None:
        """Merge YAML-provided deltas with policy-auto-resolved action deltas.

        Parity target: ``lerobot-train`` resolves ``delta_timestamps`` from
        ``policy.{action,observation}_delta_indices`` + ``dataset.fps`` via
        :func:`resolve_delta_timestamps`. If the YAML block omits a key
        (e.g. ``action``), we auto-resolve from the policy config so that
        a 10 Hz vs 30 Hz dataset does not silently change the action
        chunk horizon.

        YAML entries (if present) take precedence over auto-resolved ones.
        """
        yaml_deltas = dict(cfg.delta_timestamps or {})

        if policy_config is None:
            return yaml_deltas or None

        meta = LeRobotDatasetMetadata(LIBERO_REPO_ID, root=cfg.local_cache_dir)
        auto = resolve_delta_timestamps(policy_config, meta) or {}
        logger.info("Dataset fps=%s. Auto-resolved deltas: %s", meta.fps, {
            k: (f"[{v[0]:.4f},{v[1]:.4f},...,{v[-1]:.4f}] (n={len(v)})" if len(v) > 2 else v)
            for k, v in auto.items()
        })

        merged = {**auto, **yaml_deltas}
        return merged or None

    @property
    def batch_size(self) -> int:
        return self._group_size * self._num_groups

    @property
    def total_episodes(self) -> int:
        return len(self._episode_pool)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        loader = DataLoader(
            self._iterable,
            batch_size=None,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            prefetch_factor=self._prefetch_factor if self._num_workers > 0 else None,
            persistent_workers=self._num_workers > 0,
            collate_fn=_identity_collate,
        )
        yield from loader


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
