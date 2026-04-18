"""LIBERO simulation eval for memory-augmented SmolVLA models.

Uses lerobot's official LiberoEnv (which works) combined with our
memory wrapper. Bypasses the policy registry by directly instantiating
the memory model and running inference manually.

Setup requirements (Linux/WSL):
    sudo apt install libosmesa6
    pip install robosuite==1.4.1 mujoco==3.6.0
    export MUJOCO_GL=osmesa
    export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH

Usage:
    # Single suite
    python scripts/eval_memory_libero.py \
        --checkpoint checkpoints/memvla_libero/final.pt \
        --config configs/memvla_libero.yaml \
        --suite libero_object \
        --n-episodes 3

    # All four suites (logs per-suite + avg to wandb if --wandb is set)
    python scripts/eval_memory_libero.py \
        --checkpoint checkpoints/memvla_libero/final.pt \
        --config configs/memvla_libero.yaml \
        --all-suites \
        --n-episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Patch torch.load before any imports that use it
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.policy.builder import build_policy  # noqa: E402

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


def build_state_from_obs(raw_obs):
    """Build 8-dim state matching smolvla_libero training format:
    [eef_x, eef_y, eef_z, rotvec_x, rotvec_y, rotvec_z, gripper_L, gripper_R]
    """
    from scipy.spatial.transform import Rotation
    eef_pos = raw_obs["robot0_eef_pos"]
    eef_quat = raw_obs["robot0_eef_quat"]  # [x,y,z,w]
    rotvec = Rotation.from_quat(eef_quat).as_rotvec()
    gripper = raw_obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, rotvec, gripper])


def run_rollout(env, policy, preprocessor, postprocessor, max_steps):
    """Run a single rollout using lerobot's LiberoEnv. Returns success."""
    from lerobot.processor.core import PolicyAction

    policy.reset()
    obs, info = env.reset()

    gate_alphas = []
    success = False

    for step in range(max_steps):
        # Get raw obs from underlying env (state info we need)
        raw_obs = env.unwrapped._env.env._get_observations()

        # Build batch
        agentview = raw_obs["agentview_image"]
        eye_in_hand = raw_obs["robot0_eye_in_hand_image"]
        state = build_state_from_obs(raw_obs)

        batch = {
            "observation.images.image": torch.from_numpy(agentview).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.images.image2": torch.from_numpy(eye_in_hand).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.state": torch.from_numpy(state).float().unsqueeze(0),
            "task": env.unwrapped.task_description,
        }
        batch_p = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch_p)

        action_out = postprocessor(PolicyAction(action))
        action_np = action_out.squeeze(0).cpu().numpy()
        # Gripper is binary in training data
        action_np[6] = 1.0 if action_np[6] > 0 else -1.0

        # Track gate alpha
        if hasattr(policy, "get_gate_statistics"):
            stats = policy.get_gate_statistics()
            if stats:
                gate_alphas.append(stats.get("gate_alpha_mean", 0.0))

        obs, reward, terminated, truncated, info = env.step(action_np)

        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break

    return success, gate_alphas


ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _build_policy_from_cfg(policy_cfg: dict):
    return build_policy(
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        injection_layer=policy_cfg.get("injection_layer", 15),
        inject_before=policy_cfg.get("inject_before", True),
        mem_length=policy_cfg.get("mem_length", 4),
        retrieval_layers=policy_cfg.get("retrieval_layers", 2),
        use_timestep_pe=policy_cfg.get("use_timestep_pe", True),
        consolidate_type=policy_cfg.get("consolidate_type", "tome"),
        update_fused=policy_cfg.get("update_fused", False),
        dataloader_type=policy_cfg.get("dataloader_type", "group"),
        group_size=policy_cfg.get("group_size", 8),
        policy_overrides=policy_cfg.get("overrides") or None,
    )


def _eval_one_suite(policy, preprocessor, postprocessor, suite_name, args):
    """Run eval on a single suite; returns (per_task dict, avg success rate)."""
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(suite_name)
    n_tasks = suite.n_tasks
    task_ids = args.task_ids if args.task_ids else list(range(n_tasks))
    logger.info(
        "Suite %s: evaluating %d tasks, %d episodes each",
        suite_name, len(task_ids), args.n_episodes,
    )

    per_task: dict = {}
    for task_id in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=args.max_steps,
        )
        task_name = env.task
        logger.info("Task %d: %s", task_id, env.task_description)

        successes = []
        all_gate_alphas = []
        for ep in range(args.n_episodes):
            success, gate_alphas = run_rollout(env, policy, preprocessor, postprocessor, args.max_steps)
            successes.append(success)
            all_gate_alphas.extend(gate_alphas)
            logger.info("  ep %d/%d: success=%s", ep + 1, args.n_episodes, success)

        env.close()

        sr = sum(successes) / len(successes) * 100
        avg_gate = float(np.mean(all_gate_alphas)) if all_gate_alphas else 0.0
        per_task[task_name] = {
            "task_id": task_id,
            "success_rate": sr,
            "successes": successes,
            "avg_gate_alpha": avg_gate,
        }
        logger.info("  => success_rate=%.1f%% gate_alpha=%.4f", sr, avg_gate)

    rates = [v["success_rate"] for v in per_task.values()]
    suite_avg = float(np.mean(rates)) if rates else 0.0
    return per_task, suite_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to memory model .pt checkpoint")
    parser.add_argument("--config", required=True, help="YAML config used during training")
    suite_group = parser.add_mutually_exclusive_group()
    suite_group.add_argument("--suite", default=None, help="LIBERO suite name (single-suite mode)")
    suite_group.add_argument("--all-suites", action="store_true",
                             help="Loop over all four LIBERO suites and log per-suite + avg metrics.")
    parser.add_argument("--n-episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Optional subset of task ids")
    parser.add_argument("--output-dir", default="results/sim_memory", help="Where to save results")
    parser.add_argument("--wandb", action="store_true", help="Log per-suite metrics to wandb")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    if not args.all_suites and args.suite is None:
        args.suite = "libero_object"  # preserve prior default

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})

    logger.info("Building policy from %s", args.config)
    policy = _build_policy_from_cfg(policy_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    policy = policy.cuda().eval()
    logger.info("Loaded checkpoint at step %d", ckpt.get("step", -1))

    from lerobot.policies.factory import make_pre_post_processors
    # Mirror scripts/train.py: override tokenizer padding from policy config
    # so eval matches the actual tokenizer behavior training used.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        preprocessor_overrides={
            "tokenizer_processor": {
                "padding": policy.base_policy.config.pad_language_to,
            },
        },
    )

    wandb_run = None
    if args.wandb:
        import wandb
        trainer_cfg = cfg.get("trainer", {})
        wandb_run = wandb.init(
            project=trainer_cfg.get("wandb_project", "memory-smolvla"),
            name=args.wandb_run_name or f"eval_{Path(args.checkpoint).parent.name}",
            config={"checkpoint": args.checkpoint, "n_episodes": args.n_episodes},
            job_type="eval",
        )

    suites = list(ALL_SUITES) if args.all_suites else [args.suite]
    all_results = {"checkpoint": args.checkpoint, "suites": {}}

    for suite_name in suites:
        per_task, suite_avg = _eval_one_suite(policy, preprocessor, postprocessor, suite_name, args)
        all_results["suites"][suite_name] = {"per_task": per_task, "success_rate": suite_avg}
        logger.info("%s success_rate=%.1f%%", suite_name, suite_avg)
        if wandb_run is not None:
            wandb_run.log({f"eval/{suite_name}/success_rate": suite_avg})

    if args.all_suites:
        rates = [s["success_rate"] for s in all_results["suites"].values()]
        avg = float(np.mean(rates)) if rates else 0.0
        all_results["libero_avg"] = avg
        logger.info("LIBERO average (4 suites): %.1f%%", avg)
        if wandb_run is not None:
            wandb_run.log({"eval/libero_avg/success_rate": avg})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.all_suites else args.suite
    out_path = out_dir / f"{tag}_{Path(args.checkpoint).parent.name}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
