"""Baseline LIBERO sim eval — raw SmolVLA (no memory) through same pipeline.

Confirms the eval pipeline works before comparing memory-augmented models.

Usage:
    python scripts/eval_baseline.py --suite libero_10 --n-rollouts 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# LIBERO's torch.load calls don't pass weights_only=False, patch globally
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def make_libero_env(task_description, bddl_file, init_states, camera_names):
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


def _build_state(obs):
    """Build observation state matching training format:
    [eef_x, eef_y, eef_z, rotvec_x, rotvec_y, rotvec_z, gripper_L, gripper_R]
    where rotvec is axis-angle representation.
    """
    from scipy.spatial.transform import Rotation
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]  # robosuite returns [x,y,z,w]
    rotvec = Rotation.from_quat(eef_quat).as_rotvec()
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, rotvec, gripper])


def run_rollout(env, policy, preprocessor, postprocessor, init_state, max_steps, camera_names, task_language):
    from lerobot.processor.core import PolicyAction

    policy.reset()
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)

    # Warmup: lift robot to match training data starting position
    # Training data has eef_z ~0.68, env starts at ~0.24. Lift via +z actions.
    for _ in range(35):
        warmup_action = np.array([0, 0, 1.0, 0, 0, 0, -1.0])  # +z, open gripper
        obs, _, _, _ = env.step(warmup_action)

    for step in range(max_steps):
        # Build raw batch — preprocessor handles normalization + tokenization
        batch = {
            "observation.images.image": torch.from_numpy(obs["agentview_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.images.image2": torch.from_numpy(obs["robot0_eye_in_hand_image"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
            "observation.state": torch.from_numpy(_build_state(obs)).float().unsqueeze(0),
            "task": task_language,
        }
        batch = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(batch)

        # Postprocessor handles unnormalization
        action_out = postprocessor(PolicyAction(action))
        action_np = action_out.squeeze(0).cpu().numpy()
        # Gripper (dim 6) is binary in training data: clip to {-1, 1}
        action_np[6] = 1.0 if action_np[6] > 0 else -1.0
        obs, reward, done, info = env.step(action_np)

        if done:
            break

    success = bool(env.check_success()) if hasattr(env, "check_success") else False
    return bool(success)


def main():
    parser = argparse.ArgumentParser(description="Baseline SmolVLA eval on LIBERO (no memory)")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-dir", default="results/sim_baseline")
    args = parser.parse_args()

    # Load raw smolvla_libero — no memory wrapper
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
    from lerobot.policies.smolvla.configuration_smolvla import PolicyFeature, FeatureType

    from lerobot.policies.factory import make_pre_post_processors

    logger.info("Loading raw SmolVLA LIBERO policy (no memory)...")
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").cuda().eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path="HuggingFaceVLA/smolvla_libero"
    )
    logger.info("Policy and processors loaded")

    camera_names = ["agentview", "robot0_eye_in_hand"]
    tasks = get_libero_tasks(args.suite)
    logger.info("Suite %s: %d tasks, %d rollouts each", args.suite, len(tasks), args.n_rollouts)

    results = {"suite": args.suite, "model": "smolvla_libero_baseline", "per_task": {}}

    for task_info in tasks:
        task_name = task_info["name"]
        task_desc = task_info["description"]
        logger.info("Task: %s", task_name)

        env = make_libero_env(
            task_description=task_desc,
            bddl_file=task_info["bddl_file"],
            init_states=task_info["init_states"],
            camera_names=camera_names,
        )

        successes = []
        for ep in range(args.n_rollouts):
            init_state = task_info["init_states"][ep % len(task_info["init_states"])]
            success = run_rollout(env, policy, preprocessor, postprocessor, init_state, args.max_steps, camera_names, task_desc)
            successes.append(success)
            logger.info("  ep %d/%d: success=%s", ep + 1, args.n_rollouts, success)

        env.close()

        task_sr = sum(successes) / len(successes) * 100
        results["per_task"][task_name] = {
            "success_rate": task_sr,
            "successes": successes,
        }
        logger.info("  => success_rate=%.1f%%", task_sr)

    task_rates = [v["success_rate"] for v in results["per_task"].values()]
    results["average"] = float(np.mean(task_rates))
    logger.info("Overall: %.1f%%", results["average"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.suite}_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
