"""Evaluation entry point for memory-augmented SmolVLA on LIBERO.

Usage:
    python scripts/eval.py --checkpoint checkpoints/step_0100000.pt \\
                           --config configs/memory_only.yaml \\
                           --tasks libero_10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.eval.evaluator import MemorySmolVLAEvaluator
from memory_smolvla.policy.builder import build_policy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# LIBERO task suites
TASK_SUITES = {
    "libero_10": [
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "LIVING_ROOM_SCENE1_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE2_put_the_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE3_pick_up_the_book",
        "OFFICE_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "OFFICE_SCENE2_put_the_red_mug_on_the_right_plate",
        "OFFICE_SCENE3_pick_up_the_book_and_place_it_in_the_front_rack",
        "OFFICE_SCENE4_put_the_white_mug_on_the_right_plate",
    ],
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory-augmented SmolVLA")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--tasks",
        default="libero_10",
        help="Task suite name (e.g. 'libero_10') or comma-separated task names",
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})
    training_mode = policy_cfg.get(
        "training_mode", cfg.get("trainer", {}).get("training_mode", "memory_only")
    )

    # Build policy and load checkpoint
    policy = build_policy(
        training_mode=training_mode,
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        injection_layer=policy_cfg.get("injection_layer", 8),
        bank_max_size=policy_cfg.get("bank_max_size", 16),
        retrieval_n_heads=policy_cfg.get("retrieval_n_heads", 4),
        gate_hidden_dim=policy_cfg.get("gate_hidden_dim", 256),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"])
    logger.info("Loaded checkpoint: %s (step %d)", args.checkpoint, ckpt.get("step", -1))

    # Resolve task list
    if args.tasks in TASK_SUITES:
        task_names = TASK_SUITES[args.tasks]
    else:
        task_names = [t.strip() for t in args.tasks.split(",")]

    # Import LIBERO env factory (requires libero extra: pip install -e ".[libero]")
    try:
        from lerobot.envs.libero import make_libero_env
        env_factory = make_libero_env
    except ImportError:
        logger.error(
            "LIBERO environment not available. "
            "Install with: pip install -e '.[libero]'"
        )
        sys.exit(1)

    evaluator = MemorySmolVLAEvaluator(
        policy=policy,
        env_factory=env_factory,
        output_dir=args.output_dir,
        n_rollouts_per_task=args.n_rollouts,
        max_steps_per_episode=args.max_steps,
        wandb_project=args.wandb_project,
    )

    results = evaluator.evaluate(task_names)
    logger.info(
        "Overall success rate: %.2f",
        results["aggregate"]["mean_success_rate"],
    )


if __name__ == "__main__":
    main()
