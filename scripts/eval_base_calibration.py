"""Calibration: run the known-good finetuned base `HuggingFaceVLA/smolvla_libero`
through the SAME rollout pipeline as eval_memory_libero.py.

If this reads ~high (object ~90%, spatial ~85%), the eval pipeline (image
orientation, state assembly, normalization) is correct and any V7 shortfall is
real. If it reads low too, there's a pipeline-vs-training calibration offset
that depresses all numbers equally (mem-on vs bypass comparison still valid).

Usage:
    MUJOCO_GL=glfw DISPLAY=:0 python scripts/eval_base_calibration.py \
        --suites libero_object libero_spatial --n-episodes 5
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import memory_smolvla.data._video_compat  # noqa

# reuse the exact rollout + helpers from the memory eval
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_memory_libero as ev

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from memory_smolvla.policy.builder import _apply_libero_feature_shapes

BASE = "HuggingFaceVLA/smolvla_libero"


def base_rollout(env, policy, preprocessor, postprocessor, max_steps, seed):
    """Same as eval_memory_libero.run_rollout but with the BASE model's native
    image keys (observation.images.image / image2), not the memory model's
    camera1 / camera2 remap. Uses ev._img_tensor / ev._state8 for identical
    image + 8-dim state handling."""
    from lerobot.processor import PolicyAction
    policy.reset()
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        batch = {
            "observation.images.image": ev._img_tensor(obs["pixels"]["image"]),
            "observation.images.image2": ev._img_tensor(obs["pixels"]["image2"]),
            "observation.state": torch.from_numpy(ev._state8(obs)).float().unsqueeze(0),
            "task": env.task_description,
        }
        bp = preprocessor(batch)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action = policy.select_action(bp)
        a = postprocessor(PolicyAction(action)).squeeze(0).cpu().numpy()
        a[6] = 1.0 if a[6] > 0 else -1.0
        obs, _, terminated, truncated, info = env.step(a)
        if info.get("is_success", False):
            return True
        if terminated or truncated:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["libero_object", "libero_spatial"])
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--output-dir", default="results/v7_eval")
    args = ap.parse_args()

    policy = SmolVLAPolicy.from_pretrained(BASE)
    _apply_libero_feature_shapes(policy)
    # match training/eval inference config
    policy.config.n_action_steps = 10
    policy.config.pad_language_to = "longest"
    policy = policy.cuda().eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=BASE,
        preprocessor_overrides={"tokenizer_processor": {"padding": policy.config.pad_language_to}},
    )

    from lerobot.envs.libero import LiberoEnv, _get_suite
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    overall = {}
    for suite_name in args.suites:
        suite = _get_suite(suite_name)
        per_task = {}
        for task_id in range(suite.n_tasks):
            peek = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                             obs_type="pixels_agent_pos", episode_index=0)
            max_steps = peek._max_episode_steps; peek.close()
            succ = []
            for ep in range(args.n_episodes):
                env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=suite_name,
                                obs_type="pixels_agent_pos", episode_index=ep)
                s = base_rollout(env, policy, preprocessor, postprocessor, max_steps,
                                 seed=args.start_seed + ep)
                succ.append(s); env.close()
            sr = 100.0 * sum(succ) / len(succ)
            per_task[f"task_{task_id}"] = sr
            print(f"[base] {suite_name} task {task_id}: {sr:.0f}%", flush=True)
        sr = float(np.mean(list(per_task.values())))
        overall[suite_name] = sr
        print(f"[base] {suite_name} OVERALL {sr:.1f}%", flush=True)
        json.dump({"suite": suite_name, "overall": sr, "per_task": per_task},
                  open(f"{args.output_dir}/{suite_name}_BASE_calibration.json", "w"), indent=2)
    print("[base] CALIB_DONE", overall, flush=True)


if __name__ == "__main__":
    main()
