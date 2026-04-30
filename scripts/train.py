"""Training entry point for memory-augmented SmolVLA.

Usage:
    python scripts/train.py --config configs/memory_only.yaml
    python scripts/train.py --config configs/expert_scratch_8layers.yaml
    python scripts/train.py --config configs/memory_only.yaml --steps 1000  # quick smoke test
    python scripts/train.py --config configs/memory_only.yaml --resume checkpoints/step_0050000.pt

Authentication for private HuggingFace datasets:
    HF_TOKEN=hf_... python scripts/train.py --config ...
    # or: huggingface-cli login  (once, persists in ~/.huggingface/token)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memory_smolvla.data._video_compat  # noqa: F401  # patch pyav for torchvision >= 0.26

from memory_smolvla.data.builder import build_dataloader
from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.policy.builder import build_policy
from memory_smolvla.training.config import TrainerConfig
from memory_smolvla.training.trainer import MemorySmolVLATrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# Modes that need episode-sequential loading
_SEQUENTIAL_MODES = {"memory_only", "expert_scratch", "expert_finetune"}


def load_config(path: str) -> dict:
    """Load YAML config, resolving ``_base_`` inheritance."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train memory-augmented SmolVLA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    args = parser.parse_args()

    # Authenticate with HuggingFace if token provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" in os.environ:
        logger.info("Using HF_TOKEN from environment")

    # Resolve paths relative to project root (parent of scripts/)
    project_root = Path(__file__).resolve().parents[1]

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})
    trainer_cfg_dict = cfg.get("trainer", {})
    dataset_cfg_dict = cfg.get("dataset", {})

    # Make checkpoint_dir absolute so it doesn't depend on CWD
    ckpt_dir = trainer_cfg_dict.get("checkpoint_dir", "checkpoints")
    if not Path(ckpt_dir).is_absolute():
        trainer_cfg_dict["checkpoint_dir"] = str(project_root / ckpt_dir)

    if args.steps is not None:
        trainer_cfg_dict["total_steps"] = args.steps
        logger.info("Overriding total_steps to %d", args.steps)

    training_mode = policy_cfg.get("training_mode", trainer_cfg_dict.get("training_mode", "memory_only"))

    # Build policy
    policy = build_policy(
        training_mode=training_mode,
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        injection_layer=policy_cfg.get("injection_layer", 8),
        bank_max_size=policy_cfg.get("bank_max_size", 16),
        retrieval_n_heads=policy_cfg.get("retrieval_n_heads", 4),
        gate_hidden_dim=policy_cfg.get("gate_hidden_dim", 256),
        inject_before=policy_cfg.get("inject_before", False),
        memory_backend=policy_cfg.get("memory_backend", "episodic"),
        use_compressor=policy_cfg.get("use_compressor", False),
        compressor_n_slots=policy_cfg.get("compressor_n_slots", 8),
        use_write_gate=policy_cfg.get("use_write_gate", False),
        use_multi_scale=policy_cfg.get("use_multi_scale", False),
        eviction=policy_cfg.get("eviction", "fifo"),
        alpha_target=policy_cfg.get("alpha_target", 0.2),
        alpha_reg_weight=policy_cfg.get("alpha_reg_weight", 0.0),
        step_increment=policy_cfg.get("step_increment", 1),
        write_stride=policy_cfg.get("write_stride", 1),
        gate_init_bias=policy_cfg.get("gate_init_bias", -5.0),
        gate_type=policy_cfg.get("gate_type", "sigmoid"),
    )

    # Build dataset config
    dataset_cfg = DatasetConfig(
        repo_ids=dataset_cfg_dict.get("repo_ids", []),
        delta_timestamps=dataset_cfg_dict.get("delta_timestamps", {}),
        split=dataset_cfg_dict.get("split", "train"),
        local_cache_dir=dataset_cfg_dict.get("local_cache_dir"),
    )

    # Build data loader
    loader_mode = "sequential" if training_mode in _SEQUENTIAL_MODES else "random"
    train_loader = build_dataloader(
        cfg=dataset_cfg,
        mode=loader_mode,
        batch_size=trainer_cfg_dict.get("batch_size", 32),
        num_workers=trainer_cfg_dict.get("num_workers", 4),
    )

    # Feature map: remap dataset keys to policy-expected keys
    feature_map = dataset_cfg_dict.get("feature_map", {})
    if feature_map:
        logger.info("Feature map: %s", feature_map)

    # Build preprocessor pipeline (tokenizes language, normalizes, moves to device)
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        preprocessor_overrides={
            "device_processor": {
                "device": trainer_cfg_dict.get("device", "cuda"),
            },
        },
    )

    # Build trainer config
    trainer_config = TrainerConfig(**{
        k: v for k, v in trainer_cfg_dict.items()
        if k in TrainerConfig.__dataclass_fields__
    })
    trainer_config.training_mode = training_mode

    # Train
    trainer = MemorySmolVLATrainer(
        policy=policy,
        cfg=trainer_config,
        train_loader=train_loader,
        preprocessor=preprocessor,
        feature_map=feature_map,
    )
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
