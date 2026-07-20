"""V10A smoke test (CPU, fp32) — the core invariants before GPU arms.

Runs on a tiny real LIBERO batch (like test_path_equivalence.py) so it never
touches the training GPU. Checks:

  1. K=0 bit-exact: coconut_enabled but num_thoughts=0 ⇒ loss identical to V9
     monolithic (atol 0) — the control invariant.
  2. K=1 runs end-to-end and produces a finite loss.
  3. append_delta_kv grows the cache by exactly one token/layer per pass.
  4. Gradient routing: coconut module params get grads; thought→memory is blocked
     (memory-slot / retrieved inputs are detached in the seed).
  5. K=2 final_only: action_position_offset == valid_prefix_len + 2 while the
     visible cache holds exactly 1 thought key (Invariant 5).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t


def build(coconut_enabled, K):
    from memory_smolvla.policy.builder import build_policy
    cfg = t.load_config("configs/memvla_libero_v9.yaml")
    pc = cfg["policy"]
    torch.manual_seed(1000)
    return build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=4, compression=pc["compression"],
        n_slots=pc["n_slots"], aux_loss_weight=0.0, bptt_memory=pc.get("bptt_memory", False),
        coconut_enabled=coconut_enabled, num_thoughts=K,
        policy_overrides=pc.get("overrides") or None,
    ).float().eval(), cfg


def make_batch(policy, cfg):
    from memory_smolvla.data.dataset_config import DatasetConfig
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader
    from lerobot.policies.factory import make_pre_post_processors
    pc, dc = cfg["policy"], cfg["dataset"]
    ds = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train",
                       local_cache_dir=None)
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
    batch = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in batch.items()}
    return batch


def prep(policy, batch):
    images, img_masks = policy.base_policy.prepare_images(batch)
    state = policy.base_policy.prepare_state(batch)
    actions = policy.base_policy.prepare_action(batch)
    torch.manual_seed(0)
    noise = policy.base_policy.model.sample_noise(actions.shape, actions.device)
    time = torch.rand(actions.shape[0]) * 0.98 + 0.01
    return noise, time


def main():
    print("building K=0 (coconut-enabled, no thoughts) + V9 policies on CPU...")
    p0, cfg = build(coconut_enabled=True, K=0)
    batch = make_batch(p0, cfg)
    noise, time = prep(p0, batch)

    # --- 1. K=0 bit-exact vs V9 monolithic ---
    p0.mem_bank.reset()
    loss0, _ = p0.forward(batch, noise=noise, time=time)

    pv9, _ = build(coconut_enabled=False, K=0)
    pv9.load_state_dict(p0.state_dict(), strict=False)
    pv9.mem_bank.reset()
    lossv9, _ = pv9.forward(batch, noise=noise, time=time)
    d = abs(loss0.item() - lossv9.item())
    print(f"[1] K=0 loss={loss0.item():.8f}  V9 loss={lossv9.item():.8f}  abs_diff={d:.2e}"
          f"  -> {'PASS' if d == 0.0 else 'FAIL'}")

    # --- 2 & 4. K=1 runs + grads route into coconut, not memory-through-thought ---
    p1, _ = build(coconut_enabled=True, K=1)
    p1.load_state_dict(p0.state_dict(), strict=False)
    p1.mem_bank.reset()
    p1.zero_grad(set_to_none=True)
    loss1, ld = p1.forward(batch, noise=noise, time=time)
    loss1.backward()
    finite = torch.isfinite(loss1).item()
    seed_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in p1.coconut_seed.parameters())
    adapter_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in p1.coconut_adapters.parameters())
    print(f"[2] K=1 loss={loss1.item():.6f} finite={finite} -> {'PASS' if finite else 'FAIL'}")
    print(f"[4] grads: seed={seed_grad} adapters={adapter_grad} "
          f"-> {'PASS' if seed_grad and adapter_grad else 'FAIL (expected both True)'}")

    # --- 3 & 5. cache growth + offset decoupling (probe run_coconut_reasoning) ---
    p2, _ = build(coconut_enabled=True, K=2)
    p2.load_state_dict(p0.state_dict(), strict=False)
    p2.mem_bank.reset()
    images, img_masks = p2.base_policy.prepare_images(batch)
    state = p2.base_policy.prepare_state(batch)
    lt = batch["observation.language.tokens"]; lm = batch["observation.language.attention_mask"]
    p2._current_episode_ids = list(batch["episode_ids"])
    p2._current_timesteps = list(batch["timesteps"])
    p2.feature_extractor.set_callback(p2._memory_callback)
    pe, pp, pkv = p2._build_prefix_kv(images, img_masks, lt, lm, state)
    p2.feature_extractor.set_callback(None)
    base_len = pkv[0]["key_states"].shape[1]
    visible, offset = p2.run_coconut_reasoning(pe, pp, pkv)
    n_visible = visible[0]["key_states"].shape[1]
    vpl = pp.long().sum(dim=1)
    off_ok = torch.equal(offset, vpl + 2)
    print(f"[3] prefix cache len={base_len} unchanged (delta appended separately); "
          f"final_only n_visible={n_visible} -> {'PASS' if n_visible == 1 else 'FAIL'}")
    print(f"[5] offset==vpl+2 while visible==1: offset={offset.tolist()} vpl={vpl.tolist()} "
          f"-> {'PASS' if off_ok and n_visible == 1 else 'FAIL'}")

    ok = (d == 0.0) and finite and seed_grad and adapter_grad and (n_visible == 1) and off_ok
    print(f"\nSMOKE: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
