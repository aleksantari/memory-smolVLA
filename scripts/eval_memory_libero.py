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


def _img_tensor(arr):
    # 180deg flip (both H and W): lerobot 0.5.1's LiberoEnv obs returns the raw
    # sim render, which is upside-down relative to the HuggingFaceVLA/libero
    # training data (verified by comparing a dataset frame vs a sim reset frame).
    # Without this, every policy is fed inverted images and success collapses.
    arr = arr[::-1, ::-1]
    # .copy() because the flip (and LiberoEnv's own negative-stride view) yield
    # non-contiguous arrays torch.from_numpy can't consume.
    return torch.from_numpy(arr.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _state8(obs):
    """Build the 8-dim LIBERO training state from lerobot 0.5.1's obs.

    Training/baseline state = eef_pos(3) + eef axis-angle(3) + gripper_qpos(2).
    lerobot <=0.4 exposed this flat as ``obs["agent_pos"]``; 0.5.1 exposes the
    raw pieces under ``obs["robot_state"]`` (obs_type="pixels_agent_pos"), so we
    reassemble it here to match the normalizer the policy was trained with.
    """
    import numpy as _np
    if "agent_pos" in obs:  # lerobot <= 0.4 fast path
        return _np.asarray(obs["agent_pos"], dtype=_np.float32)
    from robosuite.utils.transform_utils import quat2axisangle
    rs = obs["robot_state"]
    eef_pos = _np.asarray(rs["eef"]["pos"], dtype=_np.float32)              # (3,)
    axisangle = _np.asarray(quat2axisangle(rs["eef"]["quat"]), dtype=_np.float32)  # (3,)
    gripper = _np.asarray(rs["gripper"]["qpos"], dtype=_np.float32)         # (2,)
    return _np.concatenate([eef_pos, axisangle, gripper]).astype(_np.float32)


def run_rollout(env, policy, preprocessor, postprocessor, max_steps, seed):
    """Run a single rollout using lerobot's LiberoEnv. Returns (success, gate_values).

    Mirrors ``lerobot.scripts.lerobot_eval.rollout`` so we match baseline_v2's
    eval protocol: env-formatted obs (rotated images, ``quat2axisangle`` state),
    seeded reset, suite-specific ``max_steps``.
    """
    try:
        from lerobot.processor import PolicyAction  # lerobot >= 0.5
    except ImportError:
        from lerobot.processor.core import PolicyAction  # lerobot <= 0.4

    policy.reset()
    obs, info = env.reset(seed=seed)

    gate_values = []
    for _ in range(max_steps):
        batch = {
            "observation.images.camera1": _img_tensor(obs["pixels"]["image"]),
            "observation.images.camera2": _img_tensor(obs["pixels"]["image2"]),
            "observation.state": torch.from_numpy(_state8(obs)).float().unsqueeze(0),
            "task": env.task_description,
        }
        batch_p = preprocessor(batch)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch_p)

        action_out = postprocessor(PolicyAction(action))
        action_np = action_out.squeeze(0).cpu().numpy()
        action_np[6] = 1.0 if action_np[6] > 0 else -1.0

        if hasattr(policy, "get_gate_statistics"):
            stats = policy.get_gate_statistics()
            if stats:
                gate_values.append(stats.get("gate_value_mean", 0.0))

        obs, reward, terminated, truncated, info = env.step(action_np)

        if info.get("is_success", False):
            return True, gate_values
        if terminated or truncated:
            return False, gate_values

    return False, gate_values


def run_rollout_ensemble(env, policy, preprocessor, postprocessor, max_steps, seed,
                         query_every=10, decay=0.1):
    """ACT-style temporal action ensembling (Zhao et al. 2023), zero extra compute.

    SmolVLA predicts a long action chunk (chunk_size, e.g. 50) but normally executes
    only ``n_action_steps`` (10) and discards the rest. Here we re-query at the SAME
    cadence (every ``query_every`` steps) but KEEP the full chunks: each executed
    timestep is covered by ~chunk_size/query_every overlapping predictions, which we
    exp-weight by chunk age. Same # of forward passes as the normal eval (so same
    memory-bank write cadence — we pin ``_infer_timestep`` to the env step to match),
    just reusing predictions we already computed. Smooths actions / reduces
    compounding error, especially on long-horizon.
    """
    from collections import defaultdict
    try:
        from lerobot.processor import PolicyAction
    except ImportError:
        from lerobot.processor.core import PolicyAction

    policy.reset()
    obs, _ = env.reset(seed=seed)
    buf: dict[int, list] = defaultdict(list)  # exec_step -> [(action_np, issue_step)]
    for t in range(max_steps):
        if t % query_every == 0 or t not in buf:
            batch = {
                "observation.images.camera1": _img_tensor(obs["pixels"]["image"]),
                "observation.images.camera2": _img_tensor(obs["pixels"]["image2"]),
                "observation.state": torch.from_numpy(_state8(obs)).float().unsqueeze(0),
                "task": env.task_description,
            }
            bp = preprocessor(batch)
            policy._infer_timestep = t  # pin bank timestamp to env step (match V7 cadence)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = policy.predict_action_chunk(bp)          # (1, H, A) normalized
            chunk = postprocessor(PolicyAction(chunk))           # unnormalize
            chunk_np = chunk.squeeze(0).cpu().numpy()            # (H, A)
            for k in range(chunk_np.shape[0]):
                buf[t + k].append((chunk_np[k], t))
        preds = buf[t]
        acts = np.stack([a for a, _ in preds])                  # (n, A)
        ages = np.array([t - s for _, s in preds], dtype=np.float32)
        w = np.exp(-decay * ages); w = w / w.sum()
        a = (acts * w[:, None]).sum(0)
        a[6] = 1.0 if a[6] > 0 else -1.0                        # gripper: threshold, don't blend
        obs, _, terminated, truncated, info = env.step(a)
        buf.pop(t, None)
        if info.get("is_success", False):
            return True, []
        if terminated or truncated:
            return False, []
    return False, []


ALL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _build_policy_from_cfg(policy_cfg: dict, coconut_cfg: dict | None = None):
    coconut_cfg = coconut_cfg or {}
    # Eval must instantiate the Coconut modules at the SAME K used in training so
    # the checkpoint's thought params load (strict=False would silently drop them).
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
        compression=policy_cfg.get("compression", "none"),
        n_slots=policy_cfg.get("n_slots", 4),
        coconut_enabled=bool(coconut_cfg.get("enabled", False)),
        num_thoughts=int(coconut_cfg.get("num_thoughts_eval",
                                         coconut_cfg.get("num_thoughts_train", 0))),
        coconut_adapter_layers=tuple(coconut_cfg.get("adapter_layers", (12, 13, 14, 15))),
        coconut_feedback_hidden=int(coconut_cfg.get("feedback_hidden", 1920)),
        coconut_expert_visibility=str(coconut_cfg.get("expert_visibility", "final_only")),
        coconut_feedback_gate_init=float(coconut_cfg.get("feedback_gate_init", -1.0)),
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
        # Peek one env for task metadata + suite-specific max_episode_steps,
        # then instantiate a fresh env per episode with ``episode_index=ep`` so
        # each rollout uses a different init_state (matches lerobot-train eval).
        peek = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            obs_type="pixels_agent_pos",
            episode_index=0,
        )
        task_name = peek.task
        task_desc = peek.task_description
        max_steps = args.max_steps if args.max_steps is not None else peek._max_episode_steps
        peek.close()
        logger.info("Task %d: %s (max_steps=%d)", task_id, task_desc, max_steps)

        successes = []
        all_gate_values = []
        for ep in range(args.n_episodes):
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=suite_name,
                obs_type="pixels_agent_pos",
                episode_index=ep,
            )
            if getattr(args, "ensemble", False):
                success, gate_values = run_rollout_ensemble(
                    env, policy, preprocessor, postprocessor, max_steps,
                    seed=args.start_seed + ep,
                    query_every=args.query_every, decay=args.ensemble_decay,
                )
            else:
                success, gate_values = run_rollout(
                    env, policy, preprocessor, postprocessor, max_steps,
                    seed=args.start_seed + ep,
                )
            successes.append(success)
            all_gate_values.extend(gate_values)
            env.close()
            logger.info("  ep %d/%d: success=%s", ep + 1, args.n_episodes, success)

        sr = sum(successes) / len(successes) * 100
        avg_gate = float(np.mean(all_gate_values)) if all_gate_values else 0.0
        per_task[task_name] = {
            "task_id": task_id,
            "success_rate": sr,
            "successes": successes,
            "avg_gate_value": avg_gate,
        }
        logger.info("  => success_rate=%.1f%% gate_value=%.4f", sr, avg_gate)

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
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode. Default: suite-specific "
                             "limit from LiberoEnv (spatial/object=280, goal=300, libero_10=520).")
    parser.add_argument("--start-seed", type=int, default=1000,
                        help="Seed for episode 0; subsequent episodes use start_seed+ep.")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Optional subset of task ids")
    parser.add_argument("--output-dir", default="results/sim_memory", help="Where to save results")
    parser.add_argument("--wandb", action="store_true", help="Log per-suite metrics to wandb")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--bypass-memory", action="store_true",
                        help="Force gate scale=1 (memory pathway bypassed). Ablation mode.")
    parser.add_argument("--bypass-thoughts", action="store_true",
                        help="V10A: force K=0 at eval (no Coconut thought passes) regardless "
                             "of config. Collapses to the V9 memory path for Gate 2 / ablation.")
    parser.add_argument("--ensemble", action="store_true",
                        help="ACT-style temporal action ensembling (V9 A1). Same compute, "
                             "reuses the overlapping chunk predictions.")
    parser.add_argument("--query-every", type=int, default=10,
                        help="Ensemble: env steps between policy queries (default 10 = n_action_steps).")
    parser.add_argument("--ensemble-decay", type=float, default=0.1,
                        help="Ensemble: exp weight decay by chunk age (default 0.1).")
    args = parser.parse_args()

    if not args.all_suites and args.suite is None:
        args.suite = "libero_object"  # preserve prior default

    cfg = load_config(args.config)
    policy_cfg = cfg.get("policy", {})

    logger.info("Building policy from %s", args.config)
    policy = _build_policy_from_cfg(policy_cfg, cfg.get("coconut", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
    policy = policy.cuda().eval()
    logger.info("Loaded checkpoint at step %d", ckpt.get("step", -1))

    if args.bypass_memory:
        policy.mem_bank.bypass = True
        logger.info("*** BYPASS MODE: memory pathway disabled (gate forced to 1.0) ***")

    if args.bypass_thoughts and getattr(policy, "num_thoughts", 0) > 0:
        policy.num_thoughts = 0
        logger.info("*** BYPASS THOUGHTS: V10A Coconut disabled at eval (K=0, V9 path) ***")

    from lerobot.policies.factory import make_pre_post_processors
    # Mirror scripts/train.py: use baseline_v2's preprocessor so the LIBERO
    # normalizer stats (state=(8,), action=(7,), flat keys) match training.
    project_root = Path(__file__).resolve().parents[1]
    preprocessor_path = policy_cfg.get(
        "preprocessor_path",
        str(project_root / "outputs/libero_baseline_v2/checkpoints/last/pretrained_model"),
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.base_policy.config,
        pretrained_path=preprocessor_path,
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
        gate_vals = [v["avg_gate_value"] for v in per_task.values()]
        suite_gate = float(np.mean(gate_vals)) if gate_vals else 0.0
        all_results["suites"][suite_name] = {
            "per_task": per_task,
            "success_rate": suite_avg,
            "avg_gate_value": suite_gate,
        }
        logger.info("%s success_rate=%.1f%% gate_value=%.4f", suite_name, suite_avg, suite_gate)
        if wandb_run is not None:
            wandb_run.log({
                f"eval/{suite_name}/success_rate": suite_avg,
                f"eval/{suite_name}/avg_gate_value": suite_gate,
            })

    if args.all_suites:
        rates = [s["success_rate"] for s in all_results["suites"].values()]
        gates = [s["avg_gate_value"] for s in all_results["suites"].values()]
        avg = float(np.mean(rates)) if rates else 0.0
        avg_gate = float(np.mean(gates)) if gates else 0.0
        all_results["libero_avg"] = avg
        all_results["libero_avg_gate_value"] = avg_gate
        logger.info("LIBERO average (4 suites): success=%.1f%% gate_value=%.4f", avg, avg_gate)
        if wandb_run is not None:
            wandb_run.log({
                "eval/libero_avg/success_rate": avg,
                "eval/libero_avg/avg_gate_value": avg_gate,
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if args.all_suites else args.suite
    suffix = "_bypass" if args.bypass_memory else ""
    out_path = out_dir / f"{tag}_{Path(args.checkpoint).parent.name}{suffix}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
