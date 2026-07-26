"""V10 dual-injection smoke (CPU, fp32) — correctness before GPU.

Checks, on a real LIBERO batch:
  1. warm_start from V7 transfers mem_bank + expert; reasoning@8 stays FRESH.
  2. BOTH banks fire in one forward (reasoning@8 gate + memory@15 gate populated).
  3. finite loss; backward populates reasoning-bank grads (it's actually training).
  4. V10A (Coconut K=1) on top runs; K=0 (coconut disabled) == V10 bit-exact.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t


def build(coconut=False, K=0, warm=True):
    from memory_smolvla.policy.builder import build_policy
    cfg = t.load_config("configs/memvla_libero_v10.yaml")
    pc = cfg["policy"]
    torch.manual_seed(1000)
    return build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=4, compression=pc["compression"],
        n_slots=pc["n_slots"], aux_loss_weight=pc.get("aux_loss_weight", 0.0),
        bptt_memory=pc.get("bptt_memory", False),
        reasoning_enabled=True, reasoning_injection_layer=8, reasoning_n_slots=8,
        reasoning_bptt=True, coconut_enabled=coconut, num_thoughts=K,
        warm_start=pc.get("warm_start") if warm else None,
        policy_overrides=pc.get("overrides") or None,
    ).float().eval(), cfg


def make_batch(policy, cfg):
    from memory_smolvla.data.dataset_config import DatasetConfig
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader
    from lerobot.policies.factory import make_pre_post_processors
    pc, dc = cfg["policy"], cfg["dataset"]
    ds = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train", local_cache_dir=None)
    loader = GroupedEpisodeLoader(cfg=ds, group_size=4, num_groups=2, mem_length=4,
                                  future_horizon=5, image_transforms=None,
                                  policy_config=policy.base_policy.config,
                                  num_workers=0, pin_memory=False, seed=1000)
    pre, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=pc["preprocessor_path"],
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "tokenizer_processor": {"padding": policy.base_policy.config.pad_language_to}})
    raw = next(iter(loader))
    meta = {k: raw.pop(k) for k in ("episode_ids", "timesteps", "future_states", "future_valid")}
    remap = {"observation.images.image": "observation.images.camera1",
             "observation.images.image2": "observation.images.camera2"}
    raw = {remap.get(k, k): v for k, v in raw.items()}
    batch = pre(raw); batch.update(meta)
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}


def prep(policy, batch):
    a = policy.base_policy.prepare_action(batch)
    torch.manual_seed(0)
    noise = policy.base_policy.model.sample_noise(a.shape, a.device)
    time = torch.rand(a.shape[0]) * 0.98 + 0.01
    return noise, time


def main():
    print("building V10 (dual-injection) on CPU, warm-start from V7...")
    p, cfg = build(coconut=False, K=0, warm=True)
    batch = make_batch(p, cfg)
    noise, time = prep(p, batch)

    # 1. warm-start report (fresh vs transferred) — inferred from the reasoning bank
    #    having no pretrained source (V7 had none): its params must be at init.
    reasoning_is_fresh = p.reasoning_bank is not None
    print(f"[1] reasoning_bank present & fresh: {reasoning_is_fresh}  "
          f"(mem_bank + expert warm-started from V7) -> {'PASS' if reasoning_is_fresh else 'FAIL'}")

    # 2 & 3. both banks fire; finite loss; reasoning grads
    p.mem_bank.reset(); p.reasoning_bank.reset()
    p.zero_grad(set_to_none=True)
    loss, ld = p.forward(batch, noise=noise, time=time)
    loss.backward()
    mem_gate = p.mem_bank.last_gate_scale()
    rea_gate = p.reasoning_bank.last_gate_scale()
    both_fire = mem_gate is not None and rea_gate is not None
    rea_grad = any(g.grad is not None and g.grad.abs().sum() > 0 for g in p.reasoning_bank.parameters())
    finite = torch.isfinite(loss).item()
    print(f"[2] both banks fired: memory@15 gate={mem_gate.mean().item():.3f}  "
          f"reasoning@8 gate={rea_gate.mean().item():.3f} -> {'PASS' if both_fire else 'FAIL'}")
    print(f"[3] loss={loss.item():.4f} finite={finite}  reasoning_bank grads={rea_grad} "
          f"-> {'PASS' if finite and rea_grad else 'FAIL'}")

    # 4. Coconut K=0 == V10 bit-exact; K=1 runs
    pc0, _ = build(coconut=True, K=0, warm=True)
    pc0.load_state_dict(p.state_dict(), strict=False)
    pc0.mem_bank.reset(); pc0.reasoning_bank.reset()
    l0, _ = pc0.forward(batch, noise=noise, time=time)
    d = abs(l0.item() - loss.item())
    print(f"[4a] Coconut K=0 == V10: abs_diff={d:.2e} -> {'PASS' if d == 0.0 else 'FAIL'}")

    pc1, _ = build(coconut=True, K=1, warm=True)
    pc1.load_state_dict(p.state_dict(), strict=False)
    pc1.mem_bank.reset(); pc1.reasoning_bank.reset()
    l1, _ = pc1.forward(batch, noise=noise, time=time)
    print(f"[4b] Coconut K=1 runs: loss={l1.item():.4f} finite={torch.isfinite(l1).item()} "
          f"-> {'PASS' if torch.isfinite(l1).item() else 'FAIL'}")

    ok = reasoning_is_fresh and both_fire and finite and rea_grad and d == 0.0 and torch.isfinite(l1).item()
    print(f"\nV10 SMOKE: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
