"""Training entry point for memory-augmented SmolVLA on LIBERO.

Usage:
    python scripts/train.py --config configs/memvla_libero.yaml
    python scripts/train.py --config configs/memvla_libero.yaml --steps 1000
    python scripts/train.py --config configs/memvla_libero.yaml --resume checkpoints/.../step_0050000.pt

Authentication for private HuggingFace datasets:
    HF_TOKEN=hf_... python scripts/train.py --config ...
    # or: huggingface-cli login  (once, persists in ~/.huggingface/token)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memory_smolvla.data._video_compat  # noqa: F401  # patch pyav for torchvision >= 0.26

from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)

from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.data.group_loader import GroupedEpisodeLoader
from memory_smolvla.policy.builder import build_policy
from memory_smolvla.training.config import TrainerConfig
from memory_smolvla.training.trainer import MemorySmolVLATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg.pop("_base_")
        base = load_config(str(base_path))
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _build_image_transforms(cfg_dict: dict | None) -> ImageTransforms | None:
    """Construct an :class:`ImageTransforms` from a YAML dict.

    Returns ``None`` when the block is absent or ``enable: false`` — no
    augmentation, matching ``LeRobotDataset``'s default.
    """
    if not cfg_dict:
        return None

    tfs_in = cfg_dict.get("tfs") or {}
    tfs = {
        name: ImageTransformConfig(
            weight=tf.get("weight", 1.0),
            type=tf.get("type", "Identity"),
            kwargs=tf.get("kwargs", {}),
        )
        for name, tf in tfs_in.items()
    }
    it_cfg_fields = {f.name for f in dataclasses.fields(ImageTransformsConfig)}
    it_cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in it_cfg_fields and k != "tfs"}
    it_cfg = ImageTransformsConfig(**it_cfg_kwargs, **({"tfs": tfs} if tfs else {}))
    if not it_cfg.enable:
        return None
    return ImageTransforms(it_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory-augmented SmolVLA on LIBERO")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" in os.environ:
        logger.info("Using HF_TOKEN from environment")

    project_root = Path(__file__).resolve().parents[1]

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})
    trainer_cfg_dict = cfg.get("trainer", {})
    dataset_cfg_dict = cfg.get("dataset", {})

    ckpt_dir = trainer_cfg_dict.get("checkpoint_dir", "checkpoints")
    if not Path(ckpt_dir).is_absolute():
        trainer_cfg_dict["checkpoint_dir"] = str(project_root / ckpt_dir)

    if args.steps is not None:
        trainer_cfg_dict["total_steps"] = args.steps
        logger.info("Overriding total_steps to %d", args.steps)

    seed = int(trainer_cfg_dict.get("seed", 1000))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("Seeded RNGs with seed=%d (baseline v2 parity).", seed)

    # Keep policy and loader in sync on group_size / dataloader_type.
    group_size = policy_cfg.get("group_size", trainer_cfg_dict.get("group_size", 8))
    num_groups = trainer_cfg_dict.get("num_groups", 8)
    mem_length = policy_cfg.get("mem_length", 8)

    policy = build_policy(
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        injection_layer=policy_cfg.get("injection_layer", 15),
        inject_before=policy_cfg.get("inject_before", True),
        mem_length=mem_length,
        retrieval_layers=policy_cfg.get("retrieval_layers", 2),
        use_timestep_pe=policy_cfg.get("use_timestep_pe", True),
        consolidate_type=policy_cfg.get("consolidate_type", "tome"),
        update_fused=policy_cfg.get("update_fused", False),
        dataloader_type=policy_cfg.get("dataloader_type", "group"),
        group_size=group_size,
        policy_overrides=policy_cfg.get("overrides") or None,
    )

    dataset_cfg = DatasetConfig(
        delta_timestamps=dataset_cfg_dict.get("delta_timestamps", {}),
        split=dataset_cfg_dict.get("split", "train"),
        local_cache_dir=dataset_cfg_dict.get("local_cache_dir"),
    )

    image_transforms = _build_image_transforms(dataset_cfg_dict.get("image_transforms"))
    if image_transforms is not None:
        logger.info("Image transforms enabled (matches baseline v2 augmentation).")

    train_loader = GroupedEpisodeLoader(
        cfg=dataset_cfg,
        group_size=group_size,
        num_groups=num_groups,
        mem_length=mem_length,
        image_transforms=image_transforms,
        policy_config=policy.base_policy.config,
        num_workers=int(trainer_cfg_dict.get("num_workers", 0)),
        prefetch_factor=int(trainer_cfg_dict.get("prefetch_factor", 4)),
        pin_memory=bool(trainer_cfg_dict.get("pin_memory", True)),
        seed=seed,
    )

    from lerobot.policies.factory import make_pre_post_processors
    # The hub-saved preprocessor bakes in padding="max_length"; baseline_v2
    # saved padding="longest". Override from the (now-resolved) policy
    # config so the yaml's pad_language_to propagates.
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        preprocessor_overrides={
            "device_processor": {
                "device": trainer_cfg_dict.get("device", "cuda"),
            },
            "tokenizer_processor": {
                "padding": policy.base_policy.config.pad_language_to,
            },
        },
    )

    trainer_config = TrainerConfig(**{
        k: v for k, v in trainer_cfg_dict.items()
        if k in TrainerConfig.__dataclass_fields__
    })

    trainer = MemorySmolVLATrainer(
        policy=policy,
        cfg=trainer_config,
        train_loader=train_loader,
        preprocessor=preprocessor,
    )
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
