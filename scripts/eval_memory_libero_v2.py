"""LIBERO sim eval for memory-augmented SmolVLA — using lerobot's official pipeline.

Replaces the broken `eval_memory_libero.py` (which built obs/state/action by hand
and bypassed lerobot's `LiberoProcessorStep`, ending up feeding the policy
upside-down images and skipping env-level processors). This version uses
`lerobot.envs.factory.make_env`, `make_env_pre_post_processors`, the policy
pre/post processors, and `lerobot.scripts.lerobot_eval.rollout` exactly the way
upstream evaluates `smolvla_libero` (which gives ~89% on libero_object on this
machine — confirmed).

The only thing different from a vanilla lerobot eval is that we plug in a
`MemorySmolVLAPolicy` (built via `memory_smolvla.policy.builder.build_policy`)
where lerobot would normally use `make_policy`. We bypass `eval_policy`'s
`PreTrainedPolicy` assertion and call the underlying `rollout` directly,
since our memory wrapper is `nn.Module` (which is what `rollout` actually
requires per its own assertion).

Usage:
    python scripts/eval_memory_libero_v2.py \\
        --checkpoint checkpoints/v5_v4match/final.pt \\
        --config configs/libero_v5_v4match.yaml \\
        --suite libero_object \\
        --n-episodes 10 \\
        --output-dir results/v5_v4match_v2

If --checkpoint is omitted, the wrapper is built but no checkpoint is loaded;
memory_proj is at zero-init so the residual gate adds zero — effectively
runs vanilla smolvla_libero and is a useful pipeline sanity check.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import einops
import numpy as np
import torch
import yaml

# Patch torch.load before any LIBERO imports use it (LIBERO loads init states
# without weights_only=False on torch>=2.4).
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.policy.builder import build_policy  # noqa: E402

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # type: ignore  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.scripts.lerobot_eval import rollout  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("eval_memory_libero_v2")


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg.pop("_base_")
        base = _load_config(str(base_path))
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


def _build_memory_policy(cfg: dict, checkpoint: str | None):
    policy_cfg = cfg.get("policy", {})
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
        compression_mode=policy_cfg.get("compression_mode", "none"),
        two_stream=policy_cfg.get("two_stream", False),
        n_image_tokens=policy_cfg.get("n_image_tokens", 0),
        perceptual_n_slots=policy_cfg.get("perceptual_n_slots", 16),
        task_n_slots=policy_cfg.get("task_n_slots", 1),
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        miss, unexp = policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        logger.info(
            "Loaded checkpoint %s at step %d (missing=%d, unexpected=%d)",
            checkpoint, ckpt.get("step", -1), len(miss), len(unexp),
        )
    else:
        logger.info(
            "No checkpoint — running with memory at init (memory_proj=0 → no-op pathway, ~vanilla smolvla_libero)."
        )
    policy = policy.cuda().eval()
    return policy


def _aggregate_per_episode(rollout_data: dict) -> list[dict]:
    """Mirror lerobot's `eval_policy` reduction: per-episode {sum_reward, max_reward, success}."""
    n_steps = rollout_data["done"].shape[1]
    done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
    mask = (
        torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)
    ).int()
    sum_rewards = einops.reduce((rollout_data["reward"] * mask).float(), "b s -> b", "sum")
    max_rewards = einops.reduce((rollout_data["reward"] * mask).float(), "b s -> b", "max")
    successes = einops.reduce(rollout_data["success"] * mask, "b s -> b", "any")
    return [
        {"sum_reward": float(s), "max_reward": float(m), "success": bool(x)}
        for s, m, x in zip(sum_rewards.tolist(), max_rewards.tolist(), successes.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Memory model .pt checkpoint (optional — omit for base smolvla_libero pipeline check)")
    parser.add_argument("--config", required=True, help="YAML training config (used to rebuild policy)")
    parser.add_argument("--suite", default="libero_object", help="LIBERO suite (libero_spatial / libero_object / libero_goal / libero_10)")
    parser.add_argument("--n-episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--task-ids", type=int, nargs="*", default=None, help="Subset of task_ids to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Vec-env batch size per task")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", default="results/v5_v2_eval")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    policy = _build_memory_policy(cfg, args.checkpoint)

    # Build LIBERO env config for the requested suite. Keep all other defaults
    # (obs_type='pixels_agent_pos', 360x360, control_mode='relative', etc.) which
    # are what the smolvla_libero baseline was trained against.
    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=args.task_ids)
    envs = make_env(env_cfg, n_envs=args.batch_size)  # dict[suite][task_id] -> VectorEnv

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.base_policy.config
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=cfg.get("policy", {}).get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_per_task: list[dict] = []
    overall_succ = []

    n_batches = max(args.n_episodes // args.batch_size, 1)
    if args.n_episodes % args.batch_size != 0:
        n_batches += 1

    amp_ctx = (
        torch.autocast(device_type="cuda") if args.use_amp else nullcontext()
    )

    with torch.no_grad(), amp_ctx:
        for task_group, task_envs in envs.items():
            logger.info("=== suite %s: %d tasks ===", task_group, len(task_envs))
            for task_id, env in task_envs.items():
                rollout_episodes: list[dict] = []
                for batch_ix in range(n_batches):
                    seeds = list(range(
                        args.seed + batch_ix * args.batch_size,
                        args.seed + (batch_ix + 1) * args.batch_size,
                    ))
                    rollout_data = rollout(
                        env=env,
                        policy=policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        seeds=seeds,
                    )
                    rollout_episodes.extend(_aggregate_per_episode(rollout_data))
                # Truncate to the requested n_episodes (in case batch_size doesn't divide it).
                rollout_episodes = rollout_episodes[: args.n_episodes]
                successes = [ep["success"] for ep in rollout_episodes]
                rate = 100.0 * sum(successes) / max(len(successes), 1)
                overall_succ.extend(successes)
                logger.info("  task_id=%d  %d/%d  (%.0f%%)", task_id, sum(successes), len(successes), rate)
                all_per_task.append({
                    "task_group": task_group,
                    "task_id": task_id,
                    "metrics": {
                        "sum_rewards": [ep["sum_reward"] for ep in rollout_episodes],
                        "max_rewards": [ep["max_reward"] for ep in rollout_episodes],
                        "successes": successes,
                    },
                })
                env.close()

    overall_rate = 100.0 * sum(overall_succ) / max(len(overall_succ), 1)
    logger.info("=== Overall %s: %d/%d  (%.1f%%) ===", args.suite, sum(overall_succ), len(overall_succ), overall_rate)

    suffix = (
        Path(args.checkpoint).parent.name if args.checkpoint else "base_smolvla_libero"
    )
    out_path = out_dir / f"{args.suite}_{suffix}.json"
    out_path.write_text(json.dumps({
        "suite": args.suite,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "per_task": all_per_task,
        "overall_success_rate": overall_rate,
        "overall_n_episodes": len(overall_succ),
    }, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
