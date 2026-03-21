"""Dataset configuration for memory-augmented SmolVLA training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for one or more LeRobot datasets.

    Multiple datasets are concatenated in the order listed. Both
    community HuggingFace datasets and datasets from a personal account
    are supported — specify any ``username/dataset-name`` repo ID.

    Authentication for private datasets requires either:
    - ``HF_TOKEN`` environment variable, or
    - ``huggingface-cli login`` run before training.

    Args:
        repo_ids: List of HuggingFace dataset repo IDs to train on.
            Examples:
            - ``["danaaubakirova/svla_so100_task1_v3"]`` (community)
            - ``["myusername/my-robot-dataset"]`` (personal)
            - ``["community/dataset1", "myuser/dataset2"]`` (mixed)
        delta_timestamps: Per-feature timestamp offsets following the
            LeRobot convention. Maps feature keys (e.g.
            ``"observation.images.top"``) to lists of float offsets in
            seconds relative to the current frame. E.g.
            ``{"observation.state": [0.0], "action": [0.0]}``.
        split: Dataset split to use (default ``"train"``).
        local_cache_dir: Optional local directory for caching downloaded
            datasets. Defaults to ``~/.cache/lerobot``.
    """

    repo_ids: list[str] = field(default_factory=list)
    delta_timestamps: dict[str, list[float]] = field(default_factory=dict)
    split: str = "train"
    local_cache_dir: str | None = None
