"""Tests for EpisodeSequentialLoader and DatasetConfig.

Uses a lightweight mock of LeRobotDataset to avoid downloading real data.
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch

from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.data.episode_loader import EpisodeBoundary, EpisodeSequentialLoader


# ------------------------------------------------------------------
# Mock LeRobotDataset
# ------------------------------------------------------------------

def _make_mock_dataset(n_episodes: int = 3, frames_per_episode: int = 4) -> MagicMock:
    """Build a minimal mock of LeRobotDataset."""
    total = n_episodes * frames_per_episode

    episodes = {}
    for i in range(n_episodes):
        episodes[i] = {
            "dataset_from_index": i * frames_per_episode,
            "dataset_to_index": (i + 1) * frames_per_episode,
        }

    meta = MagicMock()
    meta.total_episodes = n_episodes
    meta.episodes = episodes

    ds = MagicMock()
    ds.meta = meta
    ds.repo_id = "mock/dataset"
    ds.num_episodes = n_episodes

    def getitem(idx):
        return {
            "observation.state": torch.tensor([float(idx)]),
            "action": torch.zeros(6),
            "episode_index": torch.tensor(idx // frames_per_episode),
        }

    ds.__getitem__ = MagicMock(side_effect=getitem)
    ds.__len__ = MagicMock(return_value=total)
    return ds


# ------------------------------------------------------------------
# EpisodeBoundary
# ------------------------------------------------------------------

def test_episode_boundary_is_dataclass():
    b = EpisodeBoundary(episode_index=2, dataset_index=0)
    assert b.episode_index == 2
    assert b.dataset_index == 0


# ------------------------------------------------------------------
# EpisodeSequentialLoader
# ------------------------------------------------------------------

@pytest.fixture
def cfg():
    return DatasetConfig(
        repo_ids=["mock/dataset"],
        delta_timestamps={"observation.state": [0.0], "action": [0.0]},
    )


def test_loader_yields_frames_then_boundary(cfg):
    """One episode worth of frames followed by a boundary sentinel."""
    mock_ds = _make_mock_dataset(n_episodes=1, frames_per_episode=3)
    with patch(
        "memory_smolvla.data.episode_loader.LeRobotDataset",
        return_value=mock_ds,
    ):
        loader = EpisodeSequentialLoader(cfg, shuffle_episodes=False, repeat=False)
        items = list(loader)

    # 3 frames + 1 boundary
    assert len(items) == 4
    assert isinstance(items[3], EpisodeBoundary)
    assert items[3].episode_index == 0


def test_loader_frame_order(cfg):
    """Frames within an episode arrive in ascending index order."""
    mock_ds = _make_mock_dataset(n_episodes=1, frames_per_episode=5)
    with patch(
        "memory_smolvla.data.episode_loader.LeRobotDataset",
        return_value=mock_ds,
    ):
        loader = EpisodeSequentialLoader(cfg, shuffle_episodes=False, repeat=False)
        frames = [item for item in loader if not isinstance(item, EpisodeBoundary)]

    states = [f["observation.state"].item() for f in frames]
    assert states == sorted(states), "Frames must be in temporal order"


def test_loader_multiple_episodes(cfg):
    """Each episode ends with exactly one boundary sentinel."""
    n_eps = 3
    fps = 4
    mock_ds = _make_mock_dataset(n_episodes=n_eps, frames_per_episode=fps)
    with patch(
        "memory_smolvla.data.episode_loader.LeRobotDataset",
        return_value=mock_ds,
    ):
        loader = EpisodeSequentialLoader(cfg, shuffle_episodes=False, repeat=False)
        items = list(loader)

    boundaries = [i for i in items if isinstance(i, EpisodeBoundary)]
    frames = [i for i in items if not isinstance(i, EpisodeBoundary)]

    assert len(boundaries) == n_eps
    assert len(frames) == n_eps * fps


def test_loader_adds_batch_dim(cfg):
    """Each yielded frame tensor should have a leading batch dimension."""
    mock_ds = _make_mock_dataset(n_episodes=1, frames_per_episode=2)
    with patch(
        "memory_smolvla.data.episode_loader.LeRobotDataset",
        return_value=mock_ds,
    ):
        loader = EpisodeSequentialLoader(cfg, shuffle_episodes=False, repeat=False)
        frames = [i for i in loader if not isinstance(i, EpisodeBoundary)]

    for frame in frames:
        for v in frame.values():
            if isinstance(v, torch.Tensor):
                assert v.ndim >= 2, "Batch dimension missing"


def test_total_episodes_and_frames(cfg):
    mock_ds = _make_mock_dataset(n_episodes=5, frames_per_episode=10)
    with patch(
        "memory_smolvla.data.episode_loader.LeRobotDataset",
        return_value=mock_ds,
    ):
        loader = EpisodeSequentialLoader(cfg, shuffle_episodes=False, repeat=False)
    assert loader.total_episodes == 5
    assert loader.total_frames == 50


def test_empty_repo_ids_raises():
    cfg = DatasetConfig(repo_ids=[])
    with pytest.raises(ValueError, match="repo_ids"):
        EpisodeSequentialLoader(cfg)
