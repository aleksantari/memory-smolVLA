"""Dataset configuration for memory-augmented SmolVLA training.

The dataset is hardcoded to ``HuggingFaceVLA/libero`` — the dataset used
to train ``outputs/libero_baseline_v2`` (87.75% LIBERO). Keeping a single
target dataset makes the memory-port directly comparable to that
reference checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field


LIBERO_REPO_ID = "HuggingFaceVLA/libero"


@dataclass
class DatasetConfig:
    """Configuration for the LIBERO training dataset.

    Args:
        delta_timestamps: Per-feature timestamp offsets following the
            LeRobot convention. Maps feature keys (e.g.
            ``"observation.images.image"``) to lists of float offsets in
            seconds relative to the current frame.
        split: Dataset split to use (default ``"train"``).
        local_cache_dir: Optional local directory for caching the
            downloaded dataset. Defaults to ``~/.cache/lerobot``.
    """

    delta_timestamps: dict[str, list[float]] = field(default_factory=dict)
    split: str = "train"
    local_cache_dir: str | None = None
