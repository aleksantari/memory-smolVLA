"""Static-weight diagnostics: inspect what the memory modules of a trained
checkpoint actually learned. Compares the trained checkpoint to a freshly
built (init-only) policy so we can quantify how far each component drifted
from its initialization.

Reports for each memory submodule:
- weight Frobenius norm (and per-row L2 norms for memory_proj since it's
  the gate from memory features into the residual addition)
- mean / std / max-abs of weights
- min/max singular value of memory_proj (controls the "loudness" of
  memory's contribution into the prefix)

Usage:
    python scripts/diag_static_weights.py \
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --output-dir results/diag/v5_meanpool_v4hp_weights
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Patch torch.load before any LIBERO imports use it.
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_smolvla.policy.builder import build_policy  # noqa: E402

import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("diag_static_weights")


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
    return policy


def _tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach().float().flatten()
    return {
        "shape": list(t.shape),
        "frobenius_norm": float(t.norm().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
        "max_abs": float(t.abs().max().item()),
    }


def _per_row_norms_summary(W: torch.Tensor) -> dict:
    """For an [out, in] linear weight, summarize per-output-row L2 norms."""
    norms = W.detach().float().norm(dim=1)
    return {
        "n_rows": int(norms.shape[0]),
        "mean": float(norms.mean().item()),
        "std": float(norms.std().item()),
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
        "median": float(norms.median().item()),
    }


def _svd_summary(W: torch.Tensor, top_k: int = 5) -> dict:
    """Top/bottom singular values of a 2D weight matrix."""
    W = W.detach().float()
    if W.dim() != 2:
        return {"note": f"non-2D, dim={W.dim()}"}
    try:
        s = torch.linalg.svdvals(W)
    except Exception as e:
        return {"error": str(e)}
    return {
        "min_sv": float(s.min().item()),
        "max_sv": float(s.max().item()),
        "top_k": [float(x.item()) for x in s[:top_k]],
        "rank_eff_1pct": int((s > 0.01 * s.max()).sum().item()),
    }


def diag_module(name: str, mod: torch.nn.Module, init_mod: torch.nn.Module | None) -> dict:
    out = {"name": name, "params": {}}
    init_named = dict(init_mod.named_parameters()) if init_mod is not None else {}
    for pname, p in mod.named_parameters():
        rec = {"trained": _tensor_stats(p)}
        if pname in init_named:
            init_p = init_named[pname]
            rec["init"] = _tensor_stats(init_p)
            delta = (p.detach().float() - init_p.detach().float())
            rec["delta_from_init"] = _tensor_stats(delta)
        if p.dim() == 2:
            rec["per_row_norms"] = _per_row_norms_summary(p)
            rec["svd"] = _svd_summary(p)
        out["params"][pname] = rec
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/diag/static_weights")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    logger.info("Building INIT policy (no checkpoint, all memory at init)...")
    init_policy = _build_policy(cfg, checkpoint=None)

    logger.info("Building TRAINED policy (with %s)...", args.checkpoint)
    trained_policy = _build_policy(cfg, checkpoint=args.checkpoint)

    report: dict = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "modules": {},
    }

    # 1. memory_proj — linear: maps retrieved bank features into the residual addition.
    if hasattr(trained_policy, "memory_proj"):
        report["modules"]["memory_proj"] = diag_module(
            "memory_proj", trained_policy.memory_proj, init_policy.memory_proj
        )

    # 2. cross-attention retrieval block
    if hasattr(trained_policy, "retrieval"):
        report["modules"]["retrieval"] = diag_module(
            "retrieval", trained_policy.retrieval, init_policy.retrieval
        )

    # 3. gate (sigmoid MLP if used; for residual, mostly an identity-ish module)
    if hasattr(trained_policy, "gate"):
        report["modules"]["gate"] = diag_module(
            "gate", trained_policy.gate, init_policy.gate
        )

    # 4. compressor (if config used learned compressor)
    if hasattr(trained_policy, "compressor") and trained_policy.compressor is not None:
        report["modules"]["compressor"] = diag_module(
            "compressor", trained_policy.compressor, init_policy.compressor
        )

    # 5. expert finetune drift: how far did the action expert move from base?
    try:
        trained_lm_expert = trained_policy.base_policy.model.vlm_with_expert.lm_expert
        init_lm_expert = init_policy.base_policy.model.vlm_with_expert.lm_expert
        # Just summary — too many params for full breakdown
        deltas = []
        for (name, p_t), (_, p_i) in zip(
            trained_lm_expert.named_parameters(), init_lm_expert.named_parameters()
        ):
            d = (p_t.detach().float() - p_i.detach().float()).norm().item()
            init_norm = p_i.detach().float().norm().item()
            deltas.append({
                "param": name,
                "init_norm": float(init_norm),
                "delta_norm": float(d),
                "rel_change": float(d / init_norm) if init_norm > 0 else None,
            })
        report["expert_drift"] = {
            "n_params_compared": len(deltas),
            "mean_rel_change": sum(
                d["rel_change"] for d in deltas if d["rel_change"] is not None
            ) / max(1, sum(1 for d in deltas if d["rel_change"] is not None)),
            "max_rel_change": max(
                (d["rel_change"] for d in deltas if d["rel_change"] is not None),
                default=0.0,
            ),
            "first_5_params": deltas[:5],
        }
    except AttributeError as e:
        report["expert_drift"] = {"error": str(e)}

    # Print summary to stdout
    print("=" * 70)
    print(f"Static-weight diagnostic: {args.checkpoint}")
    print("=" * 70)
    for mod_name, mod_rec in report["modules"].items():
        print(f"\n[{mod_name}]")
        for pname, rec in mod_rec["params"].items():
            t = rec["trained"]
            print(f"  {pname:<40s}  shape={t['shape']}  ||W||={t['frobenius_norm']:.4f}  max|w|={t['max_abs']:.4f}")
            if "delta_from_init" in rec:
                d = rec["delta_from_init"]
                print(f"  {'':<40s}  ||W - W_init||={d['frobenius_norm']:.4f}  max|delta|={d['max_abs']:.4f}")
            if "svd" in rec and "min_sv" in rec["svd"]:
                s = rec["svd"]
                print(f"  {'':<40s}  sv_max={s['max_sv']:.4f}  sv_min={s['min_sv']:.4e}  rank_1%={s['rank_eff_1pct']}")
    print()
    if "expert_drift" in report and "mean_rel_change" in report["expert_drift"]:
        ed = report["expert_drift"]
        print(f"[lm_expert] {ed['n_params_compared']} params  mean rel-change={ed['mean_rel_change']:.4%}  max rel-change={ed['max_rel_change']:.4%}")
    print()

    # Save full JSON report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.checkpoint).parent.name}_weights.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
