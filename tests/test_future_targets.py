"""Tests for the V8 PTP future-state targets emitted by the group loader.

Uses a fake in-memory frame source (no LeRobotDataset / no downloads) and
calls ``_GroupBatchIterable._build_batch`` directly to verify that
``future_states`` and ``future_valid`` are computed per within-group
position with clamping at the group boundary.
"""

import random

import pytest
import torch

from memory_smolvla.data.group_loader import _EpisodeRef, _GroupBatchIterable

STATE_DIM = 8


class _FakeDataset:
    """Stand-in for LeRobotDataset: indexable frame source.

    Frame ``i`` carries an ``observation.state`` of shape (1, STATE_DIM)
    filled with the value ``i`` so states are uniquely identifiable, plus
    an extra tensor field and a non-tensor field to exercise ``_collate``.
    """

    def __init__(self, num_frames: int) -> None:
        self._num_frames = num_frames

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, idx: int) -> dict:
        return {
            "observation.state": torch.full((1, STATE_DIM), float(idx)),
            "action": torch.full((3,), float(idx)),
            "task": f"task_{idx}",
        }


def _make_iterable(group_size: int, num_groups: int, future_horizon: int,
                   episode_pool: list[_EpisodeRef]) -> _GroupBatchIterable:
    return _GroupBatchIterable(
        repo_id="fake/repo",
        root=None,
        delta_timestamps=None,
        image_transforms=None,
        episode_pool=episode_pool,
        group_size=group_size,
        num_groups=num_groups,
        shuffle=False,
        base_seed=0,
        future_horizon=future_horizon,
    )


def _build(group_size: int, num_groups: int, future_horizon: int, seed: int = 0):
    """Build one batch from a fake dataset; returns (batch, group_size, num_groups)."""
    ep_len = group_size + 7  # longer than group_size so random offsets occur
    pool = [
        _EpisodeRef(pool_idx=g, episode_id=100 + g,
                    from_idx=g * ep_len, to_idx=(g + 1) * ep_len)
        for g in range(num_groups)
    ]
    it = _make_iterable(group_size, num_groups, future_horizon, pool)
    dataset = _FakeDataset(num_groups * ep_len)
    batch = it._build_batch(dataset, list(range(num_groups)), random.Random(seed))
    return batch


@pytest.mark.parametrize("group_size,num_groups,horizon", [
    (10, 2, 5),
    (6, 3, 5),
    (10, 1, 1),
    (4, 2, 10),  # horizon larger than group: everything clamps, all invalid
])
def test_shapes(group_size, num_groups, horizon):
    batch = _build(group_size, num_groups, horizon)
    B = group_size * num_groups

    assert isinstance(batch["future_states"], torch.Tensor)
    assert isinstance(batch["future_valid"], torch.Tensor)
    assert batch["future_valid"].dtype == torch.float32

    # future_states must match _collate's stacking of observation.state
    assert batch["observation.state"].shape == (B, 1, STATE_DIM)
    assert batch["future_states"].shape == batch["observation.state"].shape
    assert batch["future_valid"].shape == (B,)

    # Existing metadata unchanged
    assert isinstance(batch["episode_ids"], list) and len(batch["episode_ids"]) == B
    assert isinstance(batch["timesteps"], list) and len(batch["timesteps"]) == B


@pytest.mark.parametrize("group_size,num_groups,horizon", [
    (10, 2, 5),
    (6, 3, 2),
    (5, 2, 5),   # horizon == group_size: fully clamped
    (8, 1, 0),   # horizon 0: target is the frame itself, always valid
])
def test_future_state_values_and_validity(group_size, num_groups, horizon):
    batch = _build(group_size, num_groups, horizon, seed=123)
    states = batch["observation.state"]     # (B, 1, STATE_DIM)
    future = batch["future_states"]
    valid = batch["future_valid"]

    for g in range(num_groups):
        base = g * group_size
        for k in range(group_size):
            i = base + k
            j = min(k + horizon, group_size - 1)

            # (b) future state equals the group's frame min(k+H, group_size-1)
            assert torch.equal(future[i], states[base + j]), (
                f"group {g}, k={k}: future_states mismatch (expected frame {j})"
            )

            # (c) valid is 0 exactly when k + H > group_size - 1
            expected_valid = 1.0 if k + horizon <= group_size - 1 else 0.0
            assert valid[i].item() == expected_valid, (
                f"group {g}, k={k}: expected valid={expected_valid}, got {valid[i].item()}"
            )


def test_validity_boundary_exact():
    """The last valid position is exactly k = group_size - 1 - horizon."""
    group_size, horizon = 10, 5
    batch = _build(group_size, 1, horizon)
    valid = batch["future_valid"]

    boundary = group_size - 1 - horizon
    for k in range(group_size):
        assert valid[k].item() == (1.0 if k <= boundary else 0.0)
    assert valid.sum().item() == boundary + 1


def test_targets_stay_within_own_group():
    """Future targets never cross a group boundary into the next episode."""
    group_size, num_groups, horizon = 6, 3, 4
    batch = _build(group_size, num_groups, horizon, seed=7)
    states = batch["observation.state"]
    future = batch["future_states"]

    for g in range(num_groups):
        group_vals = {states[g * group_size + k, 0, 0].item() for k in range(group_size)}
        for k in range(group_size):
            assert future[g * group_size + k, 0, 0].item() in group_vals


def test_default_future_horizon_is_5():
    pool = [_EpisodeRef(pool_idx=0, episode_id=0, from_idx=0, to_idx=20)]
    it = _GroupBatchIterable(
        repo_id="fake/repo", root=None, delta_timestamps=None,
        image_transforms=None, episode_pool=pool, group_size=10,
        num_groups=1, shuffle=False, base_seed=0,
    )
    assert it._future_horizon == 5
