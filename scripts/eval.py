"""Evaluation entry point for memory-augmented SmolVLA on LIBERO.

Usage:
    python scripts/eval.py --checkpoint checkpoints/final.pt \
                           --config configs/memory_only.yaml \
                           --suite libero_10 --n-rollouts 10
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# LIBERO's torch.load calls don't pass weights_only=False, patch globally
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from memory_smolvla.policy.builder import build_policy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LIBERO helpers
# ---------------------------------------------------------------------------

def make_libero_env(task_description, bddl_file, init_states, camera_names):
    """Create a single LIBERO OffScreenRenderEnv."""
    from libero.libero.envs import OffScreenRenderEnv

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": camera_names,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def get_libero_tasks(suite_name):
    """Get tasks from a LIBERO benchmark suite."""
    from libero.libero import benchmark, get_libero_path

    bddl_dir = Path(get_libero_path("bddl_files")) / suite_name
    bench_dict = benchmark.get_benchmark_dict()
    suite = bench_dict[suite_name]()
    n_tasks = suite.n_tasks
    tasks = []
    for i in range(n_tasks):
        task = suite.get_task(i)
        bddl_path = str(bddl_dir / task.bddl_file)
        tasks.append({
            "task_id": i,
            "name": task.name,
            "description": task.language,
            "bddl_file": bddl_path,
            "init_states": suite.get_task_init_states(i),
        })
    return tasks


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _build_state(obs):
    """Build observation state matching training format:
    [eef_x, eef_y, eef_z, rotvec_x, rotvec_y, rotvec_z, gripper_L, gripper_R]
    where rotvec is axis-angle representation.
    """
    from scipy.spatial.transform import Rotation
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    rotvec = Rotation.from_quat(eef_quat).as_rotvec()
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, rotvec, gripper])


def run_rollout(env, policy, preprocessor, postprocessor, init_state, max_steps, camera_names, task_language):
    """Run a single episode, return (success, gate_alphas)."""
    from lerobot.processor.core import PolicyAction

    policy.reset()
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)

    # Warmup: lift robot to match training data starting position
    for _ in range(35):
        warmup_action = np.array([0, 0, 1.0, 0, 0, 0, -1.0])
        obs, _, _, _ = env.step(warmup_action)

    gate_alphas = []

    for step in range(max_steps):
        batch = {
            "observation.images.image": torch.from_numpy(obs["agentview_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.images.image2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.state": torch.from_numpy(_build_state(obs)).float().unsqueeze(0),
            "task": task_language,
        }
        batch = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch)

        action_out = postprocessor(PolicyAction(action))
        action_np = action_out.squeeze(0).cpu().numpy()
        # Gripper (dim 6) is binary in training data: clip to {-1, 1}
        action_np[6] = 1.0 if action_np[6] > 0 else -1.0

        obs, reward, done, info = env.step(action_np)

        if hasattr(policy, "get_gate_statistics"):
            stats = policy.get_gate_statistics()
            if stats:
                gate_alphas.append(stats.get("gate_alpha_mean", 0.0))

        if done:
            break

    success = bool(env.check_success()) if hasattr(env, "check_success") else False
    return bool(success), gate_alphas


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory-augmented SmolVLA on LIBERO")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--suite", default="libero_10", help="LIBERO suite name")
    parser.add_argument("--n-rollouts", type=int, default=10, help="Rollouts per task")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--output-dir", default="results", help="Output directory")
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
        memory_backend=policy_cfg.get("memory_backend", "episodic"),
        use_compressor=policy_cfg.get("use_compressor", False),
        compressor_n_slots=policy_cfg.get("compressor_n_slots", 8),
        use_write_gate=policy_cfg.get("use_write_gate", False),
        use_multi_scale=policy_cfg.get("use_multi_scale", False),
        eviction=policy_cfg.get("eviction", "fifo"),
        alpha_target=policy_cfg.get("alpha_target", 0.2),
        alpha_reg_weight=policy_cfg.get("alpha_reg_weight", 0.0),
        # Use chunk_size as step_increment so temporal PE matches training
        step_increment=policy_cfg.get("step_increment", 50),
        gate_type=policy_cfg.get("gate_type", "sigmoid"),
        two_stream=policy_cfg.get("two_stream", False),
        n_image_tokens=policy_cfg.get("n_image_tokens", 0),
        perceptual_n_slots=policy_cfg.get("perceptual_n_slots", 16),
        task_n_slots=policy_cfg.get("task_n_slots", 1),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    policy = policy.cuda().eval()
    logger.info("Loaded checkpoint: %s (step %d)", args.checkpoint, ckpt.get("step", -1))

    # Build pre/post processors for the underlying smolvla_libero base
    from lerobot.policies.factory import make_pre_post_processors
    base_policy = policy.base_policy if hasattr(policy, "base_policy") else policy
    preprocessor, postprocessor = make_pre_post_processors(
        base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
    )

    # Get LIBERO tasks
    camera_names = ["agentview", "robot0_eye_in_hand"]
    tasks = get_libero_tasks(args.suite)
    logger.info("Suite %s: %d tasks, %d rollouts each", args.suite, len(tasks), args.n_rollouts)

    # Evaluate
    results = {"suite": args.suite, "checkpoint": args.checkpoint, "per_task": {}}

    for task_info in tasks:
        task_name = task_info["name"]
        task_desc = task_info["description"]
        logger.info("Task: %s (%s)", task_name, task_desc)

        env = make_libero_env(
            task_description=task_desc,
            bddl_file=task_info["bddl_file"],
            init_states=task_info["init_states"],
            camera_names=camera_names,
        )

        successes = []
        all_gate_alphas = []

        for ep in range(args.n_rollouts):
            init_state = task_info["init_states"][ep % len(task_info["init_states"])]
            success, gate_alphas = run_rollout(
                env, policy, preprocessor, postprocessor, init_state,
                args.max_steps, camera_names, task_desc,
            )
            successes.append(success)
            all_gate_alphas.append(gate_alphas)
            logger.info("  ep %d/%d: success=%s", ep + 1, args.n_rollouts, success)

        env.close()

        task_sr = sum(successes) / len(successes) * 100
        results["per_task"][task_name] = {
            "success_rate": task_sr,
            "successes": successes,
            "avg_gate_alpha": float(np.mean([a for ep in all_gate_alphas for a in ep])) if any(all_gate_alphas) else 0.0,
        }
        logger.info("  => success_rate=%.1f%%", task_sr)

    # Aggregate
    task_rates = [v["success_rate"] for v in results["per_task"].values()]
    results["average"] = float(np.mean(task_rates))
    logger.info("Overall: %.1f%%", results["average"])

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.suite}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
