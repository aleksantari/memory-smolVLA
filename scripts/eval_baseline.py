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


def run_rollout(env, policy, init_state, max_steps, camera_names, lang_tokens):
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)

    for step in range(max_steps):
        batch = {}

        # Images — smolvla_libero expects "observation.images.image" and "observation.images.image2"
        image_key_names = ["image"] + [f"image{i+1}" for i in range(1, len(camera_names))]
        for cam, key_name in zip(camera_names, image_key_names):
            img = obs[f"{cam}_image"]
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            batch[f"observation.images.{key_name}"] = img_tensor.unsqueeze(0).cuda()

        # Robot state — smolvla_libero expects 8-dim (7 joint pos + 1 gripper)
        joint_pos = obs.get("robot0_joint_pos", np.zeros(7))
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
        robot_state = np.concatenate([joint_pos, gripper_qpos[:1]])
        batch["observation.state"] = torch.from_numpy(robot_state).float().unsqueeze(0).cuda()

        batch["observation.language.tokens"] = lang_tokens["input_ids"].cuda()
        batch["observation.language.attention_mask"] = lang_tokens["attention_mask"].bool().cuda()

        with torch.no_grad():
            action = policy.select_action(batch)

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(action_np)

        if done:
            break

    success = env.is_success()["task"] if hasattr(env, "is_success") else False
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

    logger.info("Loading raw SmolVLA LIBERO policy (no memory)...")
    config = SmolVLAConfig(
        pretrained_path="HuggingFaceVLA/smolvla_libero",
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )
    policy = SmolVLAPolicy(config)
    policy = policy.cuda().eval()
    logger.info("Policy loaded")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    camera_names = ["agentview", "robot0_eye_in_hand"]
    tasks = get_libero_tasks(args.suite)
    logger.info("Suite %s: %d tasks, %d rollouts each", args.suite, len(tasks), args.n_rollouts)

    results = {"suite": args.suite, "model": "smolvla_libero_baseline", "per_task": {}}

    for task_info in tasks:
        task_name = task_info["name"]
        task_desc = task_info["description"]
        logger.info("Task: %s", task_name)

        lang_tokens = tokenizer(task_desc, return_tensors="pt", padding=True, truncation=True)

        env = make_libero_env(
            task_description=task_desc,
            bddl_file=task_info["bddl_file"],
            init_states=task_info["init_states"],
            camera_names=camera_names,
        )

        successes = []
        for ep in range(args.n_rollouts):
            init_state = task_info["init_states"][ep % len(task_info["init_states"])]
            success = run_rollout(env, policy, init_state, args.max_steps, camera_names, lang_tokens)
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
