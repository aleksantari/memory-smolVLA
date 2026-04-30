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
    python scripts/eval_memory_libero.py \
        --checkpoint checkpoints/libero_injection_half_v3/final.pt \
        --config configs/libero_injection_half_v3.yaml \
        --suite libero_object \
        --n-episodes 3
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to memory model .pt checkpoint")
    parser.add_argument("--config", required=True, help="YAML config used during training")
    parser.add_argument("--suite", default="libero_object", help="LIBERO suite name")
    parser.add_argument("--n-episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Optional subset of task ids")
    parser.add_argument("--output-dir", default="results/sim_memory", help="Where to save results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})

    # Build memory policy
    logger.info("Building policy from %s", args.config)
    policy = build_policy(
        training_mode=policy_cfg.get("training_mode", "memory_only"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        base_checkpoint=policy_cfg.get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
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
        step_increment=policy_cfg.get("step_increment", 50),
        write_stride=policy_cfg.get("write_stride", 1),
        gate_init_bias=policy_cfg.get("gate_init_bias", -5.0),
        gate_type=policy_cfg.get("gate_type", "sigmoid"),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    policy = policy.cuda().eval()
    logger.info("Loaded checkpoint at step %d", ckpt.get("step", -1))

    # Build pre/post processors from the underlying base smolvla policy
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
    )

    # Use lerobot's LiberoEnv (the working one)
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(args.suite)
    n_tasks = suite.n_tasks
    task_ids = args.task_ids if args.task_ids else list(range(n_tasks))
    logger.info("Suite %s: evaluating %d tasks, %d episodes each", args.suite, len(task_ids), args.n_episodes)

    results = {"suite": args.suite, "checkpoint": args.checkpoint, "per_task": {}}

    for task_id in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=args.suite,
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
        results["per_task"][task_name] = {
            "task_id": task_id,
            "success_rate": sr,
            "successes": successes,
            "avg_gate_alpha": avg_gate,
        }
        logger.info("  => success_rate=%.1f%% gate_alpha=%.4f", sr, avg_gate)

    rates = [v["success_rate"] for v in results["per_task"].values()]
    results["average"] = float(np.mean(rates)) if rates else 0.0
    logger.info("Overall: %.1f%%", results["average"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.suite}_{Path(args.checkpoint).parent.name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
