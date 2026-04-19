"""Evaluation entry point for memory-augmented SmolVLA on LIBERO.

Usage:
    python scripts/eval.py --checkpoint checkpoints/memvla_libero/final.pt \
                           --config configs/memvla_libero.yaml \
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
# LIBERO dataset keys -> SmolVLA policy keys. Must match the trainer
# (src/memory_smolvla/training/trainer.py::_LIBERO_FEATURE_MAP) or the
# base policy's prepare_images() will fail with "All image features
# are missing from the batch" (it expects camera1/2/3 from the pretrained
# lerobot/smolvla_base config).
_LIBERO_FEATURE_MAP: dict[str, str] = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


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
    """Build observation state matching training format."""
    from scipy.spatial.transform import Rotation
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    rotvec = Rotation.from_quat(eef_quat).as_rotvec()
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, rotvec, gripper])


def run_rollout(
    env, policy, preprocessor, postprocessor, init_state,
    max_steps, camera_names, task_language, episode_id,
):
    """Run a single episode, return (success, gate_means).

    ``episode_id`` keys the memory bank so rollouts stay isolated.
    """
    from lerobot.processor.core import PolicyAction

    policy.reset()
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)

    # Warmup: lift robot to match training data starting position
    for _ in range(35):
        warmup_action = np.array([0, 0, 1.0, 0, 0, 0, -1.0])
        obs, _, _, _ = env.step(warmup_action)

    gate_means: list[float] = []

    for step in range(max_steps):
        batch = {
            "observation.images.image": torch.from_numpy(obs["agentview_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.images.image2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.state": torch.from_numpy(_build_state(obs)).float().unsqueeze(0),
            "task": task_language,
        }
        batch = {_LIBERO_FEATURE_MAP.get(k, k): v for k, v in batch.items()}
        batch = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch, episode_id=episode_id)

        action_out = postprocessor(PolicyAction(action))
        action_np = action_out.squeeze(0).cpu().numpy()
        action_np[6] = 1.0 if action_np[6] > 0 else -1.0

        obs, reward, done, info = env.step(action_np)

        scale = policy.mem_bank.last_gate_scale()
        if scale is not None:
            gate_means.append(float(scale.mean().item()))

        if done:
            break

    success = bool(env.check_success()) if hasattr(env, "check_success") else False
    return bool(success), gate_means


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

    policy = build_policy(
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        injection_layer=policy_cfg.get("injection_layer", 15),
        inject_before=policy_cfg.get("inject_before", True),
        mem_length=policy_cfg.get("mem_length", 8),
        retrieval_layers=policy_cfg.get("retrieval_layers", 2),
        use_timestep_pe=policy_cfg.get("use_timestep_pe", True),
        consolidate_type=policy_cfg.get("consolidate_type", "tome"),
        update_fused=policy_cfg.get("update_fused", False),
        dataloader_type=policy_cfg.get("dataloader_type", "group"),
        group_size=policy_cfg.get("group_size", 8),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    policy = policy.cuda().eval()
    logger.info("Loaded checkpoint: %s (step %d)", args.checkpoint, ckpt.get("step", -1))

    from lerobot.policies.factory import make_pre_post_processors
    base_policy = policy.base_policy if hasattr(policy, "base_policy") else policy
    preprocessor, postprocessor = make_pre_post_processors(
        base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
    )

    camera_names = ["agentview", "robot0_eye_in_hand"]
    tasks = get_libero_tasks(args.suite)
    logger.info("Suite %s: %d tasks, %d rollouts each", args.suite, len(tasks), args.n_rollouts)

    results = {"suite": args.suite, "checkpoint": args.checkpoint, "per_task": {}}
    global_episode_id = 0

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
        all_gate_means = []

        for ep in range(args.n_rollouts):
            init_state = task_info["init_states"][ep % len(task_info["init_states"])]
            success, gate_means = run_rollout(
                env, policy, preprocessor, postprocessor, init_state,
                args.max_steps, camera_names, task_desc,
                episode_id=global_episode_id,
            )
            global_episode_id += 1
            successes.append(success)
            all_gate_means.append(gate_means)
            logger.info("  ep %d/%d: success=%s", ep + 1, args.n_rollouts, success)

        env.close()

        task_sr = sum(successes) / len(successes) * 100
        results["per_task"][task_name] = {
            "success_rate": task_sr,
            "successes": successes,
            "avg_gate_mean": float(np.mean([a for ep in all_gate_means for a in ep])) if any(all_gate_means) else 0.0,
        }
        logger.info("  => success_rate=%.1f%%", task_sr)

    task_rates = [v["success_rate"] for v in results["per_task"].values()]
    results["average"] = float(np.mean(task_rates))
    logger.info("Overall: %.1f%%", results["average"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.suite}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
