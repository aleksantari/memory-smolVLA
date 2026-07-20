"""V10A Step 0.5 (BLOCKING) — cached ≡ monolithic V9 equivalence.

V10A must train through the cached decomposition (prefix-KV build with
fill_kv_cache=True → cached expert forward), but baseline V9 trains monolithically
(one joint forward). Prove/disprove that the two produce the same flow-matching
loss AND gradients for the V9 policy itself, at matched noise/time.

Decision rule (pre-registered, RevB):
  PASS (fp32 rtol 1e-5) -> the running V9 is a valid Arm-1 control.
  FAIL -> V9 disqualified; ALL arms (incl. control) must train on the cached path.

Runs on CPU (fp32) with a tiny real batch so it doesn't touch the training GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

TOL_R, TOL_A = 1e-5, 1e-5


def flow_loss_monolithic(model, ins, noise, time):
    """Stock lerobot training forward → per-element MSE (B, chunk, adim)."""
    return model.forward(*ins, noise, time)


def flow_loss_cached(model, ins, noise, time):
    """Cached decomposition: prefix KV build (fill_kv_cache=True) → denoise_step."""
    images, img_masks, lang_tokens, lang_masks, state, actions = ins
    te = time[:, None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state)
    prefix_att2d = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
    _, past_kv = model.vlm_with_expert.forward(
        attention_mask=prefix_att2d, position_ids=prefix_pos, past_key_values=None,
        inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True)
    v_t = model.denoise_step(prefix_pad_masks=prefix_pad, past_key_values=past_kv,
                             x_t=x_t, timestep=time)
    return F.mse_loss(u_t, v_t, reduction="none")


def run_loss(policy, batch, ins, noise, time, cached: bool):
    """Install the memory callback (so the prefix is memory-augmented in both paths),
    set per-frame episode ids/timesteps, run the chosen path, return the mean loss."""
    policy.zero_grad(set_to_none=True)
    policy.mem_bank.reset()
    policy._current_episode_ids = list(batch["episode_ids"])
    policy._current_timesteps = list(batch["timesteps"])
    policy.feature_extractor.set_callback(policy._memory_callback)
    try:
        model = policy.base_policy.model
        losses = flow_loss_cached(model, ins, noise, time) if cached \
            else flow_loss_monolithic(model, ins, noise, time)
    finally:
        policy.feature_extractor.set_callback(None)
        policy._current_episode_ids = None
        policy._current_timesteps = None
    adim = policy.base_policy.config.max_action_dim
    return losses[:, :, :adim].mean()


def main():
    cfg = t.load_config("configs/memvla_libero_v9.yaml")
    pc, tc, dc = cfg["policy"], cfg["trainer"], cfg["dataset"]
    from memory_smolvla.data.dataset_config import DatasetConfig
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader
    from memory_smolvla.policy.builder import build_policy
    from lerobot.policies.factory import make_pre_post_processors

    print("building V9 policy on CPU (fp32)...")
    policy = build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=4, compression=pc["compression"],
        n_slots=pc["n_slots"], aux_loss_weight=0.0, bptt_memory=pc.get("bptt_memory", False),
        policy_overrides=pc.get("overrides") or None,
    ).float().eval()

    ds = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train", local_cache_dir=None)
    it = t._build_image_transforms({"enable": False})
    loader = GroupedEpisodeLoader(cfg=ds, group_size=4, num_groups=2, mem_length=4,
                                  future_horizon=5, image_transforms=it,
                                  policy_config=policy.base_policy.config,
                                  num_workers=0, pin_memory=False, seed=1000)
    proj_root = Path(__file__).resolve().parents[1]
    pre, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=pc["preprocessor_path"],
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "tokenizer_processor": {"padding": policy.base_policy.config.pad_language_to}})

    raw = next(iter(loader))
    from lerobot.policies.smolvla.modeling_smolvla import ACTION  # noqa
    meta = {k: raw.pop(k) for k in ("episode_ids", "timesteps", "future_states", "future_valid")}
    remap = {"observation.images.image": "observation.images.camera1",
             "observation.images.image2": "observation.images.camera2"}
    raw = {remap.get(k, k): v for k, v in raw.items()}
    batch = pre(raw); batch.update(meta)
    batch = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}

    ins = (*policy.base_policy.prepare_images(batch),
           policy.base_policy.config)  # placeholder, fixed below
    images, img_masks = policy.base_policy.prepare_images(batch)
    state = policy.base_policy.prepare_state(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    actions = policy.base_policy.prepare_action(batch)
    ins = (images, img_masks, lang_tokens, lang_masks, state, actions)

    B = actions.shape[0]
    torch.manual_seed(0)
    noise = policy.base_policy.model.sample_noise(actions.shape, actions.device)
    time = torch.rand(B) * 0.98 + 0.01   # fixed, matched across paths

    print("computing monolithic V9 flow loss + grads...")
    loss_m = run_loss(policy, batch, ins, noise, time, cached=False)
    loss_m.backward()
    grad_m = {n: p.grad.detach().clone() for n, p in policy.named_parameters() if p.grad is not None}

    print("computing cached V9 flow loss + grads...")
    loss_c = run_loss(policy, batch, ins, noise, time, cached=True)
    loss_c.backward()
    grad_c = {n: p.grad.detach().clone() for n, p in policy.named_parameters() if p.grad is not None}

    print("\n=== RESULT ===")
    print(f"loss  monolithic={loss_m.item():.8f}  cached={loss_c.item():.8f}  "
          f"abs_diff={abs(loss_m.item()-loss_c.item()):.2e}")
    loss_ok = torch.allclose(loss_m, loss_c, rtol=TOL_R, atol=TOL_A)

    # gradient equivalence on the shared trainable params (Step 0.5 param list)
    keys = ["lm_expert", "action_out_proj", "retrieval_blocks", "gate_fusion", "reasoning_head"]
    worst = 0.0; checked = 0; grad_ok = True
    for n in grad_m:
        if any(k in n for k in keys) and n in grad_c:
            gm, gc = grad_m[n], grad_c[n]
            d = (gm - gc).abs().max().item(); worst = max(worst, d); checked += 1
            if not torch.allclose(gm, gc, rtol=1e-4, atol=1e-5):
                grad_ok = False
    print(f"gradients checked on {checked} shared params  worst_abs_diff={worst:.2e}  "
          f"({'MATCH' if grad_ok else 'MISMATCH'})")
    verdict = loss_ok and grad_ok
    print(f"\nGATE 0.5: {'PASS — running V9 is a valid control' if verdict else 'FAIL — cached != monolithic; all arms must train cached'}")
    return 0 if verdict else 2


if __name__ == "__main__":
    sys.exit(main())
