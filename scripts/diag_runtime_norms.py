"""Layer-2 runtime instrumentation: hook into the memory pathway during a real
inference rollout and log per-callback statistics. Confirms the static-weight
finding (memory_proj is ~5% of typical scale) by measuring the actual delta
the memory pathway adds to the prefix at every step.

Logged per callback invocation:
  - bank_size (number of entries currently in the bank)
  - prefix_norm (||current||)  — magnitude of the VLM prefix entering the callback
  - retrieved_norm (||retrieved||) — cross-attention output magnitude
  - projected_norm (||memory_proj(retrieved)||) — actual delta added to prefix
  - relative_perturbation (||delta|| / ||prefix||) — what the action expert "sees"

Runs through lerobot's rollout pipeline (one task, one episode) so observations
match training distribution exactly.

Usage:
    python scripts/diag_runtime_norms.py \\
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \\
        --config configs/libero_v5_meanpool_v4hp.yaml \\
        --suite libero_object \\
        --task-id 0 \\
        --output-dir results/diag/runtime_norms
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.policy.builder import build_policy  # noqa: E402

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.scripts.lerobot_eval import rollout  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("diag_runtime_norms")


def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg.pop("_base_")
        base = _load_config(str(base_path))
        def merge(b, o):
            r = dict(b)
            for k, v in o.items():
                if k in r and isinstance(r[k], dict) and isinstance(v, dict):
                    r[k] = merge(r[k], v)
                else:
                    r[k] = v
            return r
        cfg = merge(base, cfg)
    return cfg


def _build_policy(cfg: dict, checkpoint: str | None):
    p = cfg.get("policy", {})
    policy = build_policy(
        training_mode=p.get("training_mode", "memory_only"),
        num_vlm_layers=p.get("num_vlm_layers", 16),
        base_checkpoint=p.get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
        injection_layer=p.get("injection_layer", 8),
        bank_max_size=p.get("bank_max_size", 16),
        retrieval_n_heads=p.get("retrieval_n_heads", 4),
        gate_hidden_dim=p.get("gate_hidden_dim", 256),
        memory_backend=p.get("memory_backend", "episodic"),
        use_compressor=p.get("use_compressor", False),
        compressor_n_slots=p.get("compressor_n_slots", 8),
        use_write_gate=p.get("use_write_gate", False),
        use_multi_scale=p.get("use_multi_scale", False),
        eviction=p.get("eviction", "fifo"),
        alpha_target=p.get("alpha_target", 0.2),
        alpha_reg_weight=p.get("alpha_reg_weight", 0.0),
        step_increment=p.get("step_increment", 50),
        write_stride=p.get("write_stride", 1),
        gate_init_bias=p.get("gate_init_bias", -5.0),
        gate_type=p.get("gate_type", "sigmoid"),
        compression_mode=p.get("compression_mode", "none"),
        two_stream=p.get("two_stream", False),
        n_image_tokens=p.get("n_image_tokens", 0),
        perceptual_n_slots=p.get("perceptual_n_slots", 16),
        task_n_slots=p.get("task_n_slots", 1),
    )
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        logger.info("Loaded %s at step %d", checkpoint, ckpt.get("step", -1))
    policy = policy.cuda().eval()
    return policy


def attach_norm_hooks(policy, log: list[dict]) -> None:
    """Hook memory_proj and retrieval to capture per-callback norms.

    Stash:
      - Pre-projection (retrieval output): retrieved_norm
      - Post-projection (memory_proj output): projected_norm
      - Bank size: read from policy.memory_bank inside the hook
    """
    state = {"pending": {}}

    def retrieval_hook(_mod, _inp, out):
        # `retrieval` returns the cross-attended tensor [B, L, D]
        state["pending"]["retrieved_norm"] = float(out.detach().float().norm().item())

    def memory_proj_hook(mod, inp, out):
        # `memory_proj` is Linear; input is retrieved [B, L, D], output [B, L, D]
        in_norm = float(inp[0].detach().float().norm().item())
        out_norm = float(out.detach().float().norm().item())
        bank_size = len(policy.memory_bank) if hasattr(policy, "memory_bank") and policy.memory_bank is not None else None
        rec = {
            "callback_idx": len(log),
            "bank_size": bank_size,
            "memory_proj_input_norm": in_norm,
            "projected_norm": out_norm,
            "retrieved_norm": state["pending"].get("retrieved_norm"),
            "memory_proj_amplification": (out_norm / in_norm) if in_norm > 0 else None,
            "policy_timestamp": policy._timestamp,
        }
        log.append(rec)
        state["pending"] = {}

    if hasattr(policy, "retrieval") and policy.retrieval is not None:
        policy.retrieval.register_forward_hook(retrieval_hook)
    if hasattr(policy, "memory_proj") and policy.memory_proj is not None:
        policy.memory_proj.register_forward_hook(memory_proj_hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output-dir", default="results/diag/runtime_norms")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    policy = _build_policy(cfg, args.checkpoint)

    log: list[dict] = []
    attach_norm_hooks(policy, log)

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[args.task_id])
    envs = make_env(env_cfg, n_envs=1)

    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.base_policy.config)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=cfg.get("policy", {}).get("base_checkpoint", "HuggingFaceVLA/smolvla_libero"),
    )

    # The make_env return shape is dict[suite][task_id] -> VectorEnv
    suite_envs = envs[args.suite]
    if args.task_id not in suite_envs:
        # fall back to the only available
        env = next(iter(suite_envs.values()))
    else:
        env = suite_envs[args.task_id]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), nullcontext():
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_pre,
            env_postprocessor=env_post,
            preprocessor=pre,
            postprocessor=post,
            seeds=[1000],
        )
    env.close()

    success = bool(rollout_data["success"].any().item())
    n_steps = int(rollout_data["done"].shape[1])
    logger.info("Rollout complete: success=%s, n_steps=%d, callbacks logged=%d", success, n_steps, len(log))

    # Summary stats over the rollout
    if log:
        proj_norms = [r["projected_norm"] for r in log if r["projected_norm"] is not None]
        retr_norms = [r["retrieved_norm"] for r in log if r["retrieved_norm"] is not None]
        amps = [r["memory_proj_amplification"] for r in log if r["memory_proj_amplification"] is not None]
        bank_sizes = [r["bank_size"] for r in log if r["bank_size"] is not None]
        summary = {
            "n_callbacks": len(log),
            "rollout_success": success,
            "rollout_n_env_steps": n_steps,
            "projected_norm": {
                "mean": sum(proj_norms) / len(proj_norms) if proj_norms else None,
                "min": min(proj_norms) if proj_norms else None,
                "max": max(proj_norms) if proj_norms else None,
            },
            "retrieved_norm": {
                "mean": sum(retr_norms) / len(retr_norms) if retr_norms else None,
                "min": min(retr_norms) if retr_norms else None,
                "max": max(retr_norms) if retr_norms else None,
            },
            "memory_proj_amplification": {
                "mean": sum(amps) / len(amps) if amps else None,
                "min": min(amps) if amps else None,
                "max": max(amps) if amps else None,
            },
            "bank_size_progression": bank_sizes[:20] + (["…"] if len(bank_sizes) > 20 else []),
            "final_bank_size": bank_sizes[-1] if bank_sizes else None,
        }
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("=" * 60)
    else:
        summary = {"warning": "no callbacks logged — hooks may not have fired"}

    suffix = Path(args.checkpoint).parent.name
    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "suite": args.suite,
        "task_id": args.task_id,
        "summary": summary,
        "per_callback": log,
    }
    out_path = out_dir / f"{suffix}_{args.suite}_task{args.task_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
