"""V9 gate/memory diagnostic — is the retrieved memory informationally inert?

Loads V9 final.pt, runs a contiguous real episode segment through the memory
bank in EVAL mode (streaming, one frame at a time like a rollout), and measures,
per query where the bank is non-empty:
  * gate scale distribution (should be ~0.40, matching training/eval logs)
  * relative L2 and cosine between `retrieved` and `current` (working_mem)
  * how much the FUSED output actually moves vs current: ||fused-current|| / ||current||

If retrieved ≈ current (tiny rel-L2, cosine ~1), then even a 60%-weight blend
barely changes the hidden state → memory is redundant, which explains why
MEM-ON ≈ BYPASS despite a healthy gate. CPU/fp32; does not touch the GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t

CKPT = "checkpoints/memvla_libero_v9/final.pt"


def main():
    from memory_smolvla.policy.builder import build_policy
    from memory_smolvla.data.dataset_config import DatasetConfig
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader
    from lerobot.policies.factory import make_pre_post_processors

    cfg = t.load_config("configs/memvla_libero_v9.yaml")
    pc, dc = cfg["policy"], cfg["dataset"]
    policy = build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=16, compression=pc["compression"],
        n_slots=pc["n_slots"], bptt_memory=pc.get("bptt_memory", False),
        policy_overrides=pc.get("overrides") or None,
    ).float().eval()
    ck = torch.load(CKPT, map_location="cpu")
    missing, unexpected = policy.load_state_dict(ck["policy_state_dict"], strict=False)
    print(f"loaded {CKPT} @ step {ck.get('step')}  (missing={len(missing)} unexpected={len(unexpected)})")

    # One contiguous 16-frame segment of a single episode (group_size=16).
    ds = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train",
                       local_cache_dir=None)
    loader = GroupedEpisodeLoader(cfg=ds, group_size=16, num_groups=1, mem_length=4,
                                  future_horizon=5, image_transforms=None,
                                  policy_config=policy.base_policy.config,
                                  num_workers=0, pin_memory=False, seed=7)
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

    # Instrument the bank: capture per-item (current, retrieved) inside process_batch.
    bank = policy.mem_bank
    cap = {"cur": [], "ret": []}
    orig = bank.gate_fusion.last_scale

    def spy(cur, ret):
        cap["cur"].append(cur.detach())
        cap["ret"].append(ret.detach())
        return orig(cur, ret)

    bank.gate_fusion.last_scale = spy

    # Stream frames one at a time (eval/streaming semantics), same episode id.
    policy.eval()
    bank.reset()
    images, img_masks = policy.base_policy.prepare_images(batch)
    state = policy.base_policy.prepare_state(batch)
    lt = batch["observation.language.tokens"]; lm = batch["observation.language.attention_mask"]
    N = state.shape[0]
    gates, relL2, cos, movefrac = [], [], [], []
    for i in range(N):
        sl = slice(i, i + 1)
        policy._current_episode_ids = [0]
        policy._current_timesteps = [i]
        policy.feature_extractor.set_callback(policy._memory_callback)
        with torch.no_grad():
            policy._build_prefix_kv([im[sl] for im in images], [m[sl] for m in img_masks],
                                    lt[sl], lm[sl], state[sl])
        policy.feature_extractor.set_callback(None)
        cur, ret = cap["cur"][-1], cap["ret"][-1]           # (1, L, D)
        g = bank.last_gate_scale()
        if i == 0:
            continue  # empty bank ⇒ retrieved==current by construction; skip
        gates.append(g.mean().item())
        num = (ret - cur).norm(dim=-1)                       # (1, L)
        den = cur.norm(dim=-1).clamp(min=1e-6)
        relL2.append((num / den).mean().item())
        cos.append(torch.nn.functional.cosine_similarity(ret, cur, dim=-1).mean().item())
        fused = g * cur + (1 - g) * ret
        movefrac.append(((fused - cur).norm(dim=-1) / den).mean().item())

    def stat(x): return f"mean={sum(x)/len(x):.4f} min={min(x):.4f} max={max(x):.4f}"
    print(f"\nqueries with non-empty bank: {len(gates)}")
    print(f"gate scale (1=all current, 0=all retrieved):  {stat(gates)}")
    print(f"retrieved vs current  rel-L2 ||ret-cur||/||cur||: {stat(relL2)}")
    print(f"retrieved vs current  cosine:                     {stat(cos)}")
    print(f"fused moves current  ||fused-cur||/||cur||:       {stat(movefrac)}")
    print("\ninterpretation:")
    print(" - if rel-L2 small (<0.1) & cosine ~1  -> retrieved ≈ current: memory REDUNDANT")
    print(" - if fused-move small (<0.05)         -> blend barely changes hidden state")
    print(" - gate ~0.40 confirms it is NOT collapsed (matches eval log 0.404)")


if __name__ == "__main__":
    main()
