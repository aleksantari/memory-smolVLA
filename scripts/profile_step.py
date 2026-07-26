"""Profile one training step to locate the 9s bottleneck (caching design basis).

Times, on the real GPU with a real batch: data fetch (incl. pyav decode),
embed_prefix (vision tower + tokenize), the 16-layer VLM forward, expert+loss,
and backward. Tells us whether caching prefix-embeddings (kills vision+decode)
or caching the layer-8 residual (also kills layers 0-7) is worth the complexity.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = t.load_config("configs/memvla_libero_v9.yaml")
    pc, dc, tc = cfg["policy"], cfg["dataset"], cfg["trainer"]
    from memory_smolvla.policy.builder import build_policy
    from memory_smolvla.data.dataset_config import DatasetConfig
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader
    from lerobot.policies.factory import make_pre_post_processors

    policy = build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=8, compression=pc["compression"],
        n_slots=pc["n_slots"], aux_loss_weight=pc.get("aux_loss_weight", 0.0),
        bptt_memory=pc.get("bptt_memory", False), policy_overrides=pc.get("overrides") or None,
    ).to(dev)
    policy.train()

    ds = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train", local_cache_dir=None)
    it = t._build_image_transforms(dc.get("image_transforms"))
    loader = GroupedEpisodeLoader(cfg=ds, group_size=8, num_groups=16, mem_length=pc["mem_length"],
                                  future_horizon=5, image_transforms=it,
                                  policy_config=policy.base_policy.config,
                                  num_workers=int(tc.get("num_workers", 8)), pin_memory=False, seed=1)
    pre, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=pc["preprocessor_path"],
        preprocessor_overrides={"device_processor": {"device": dev},
                                "tokenizer_processor": {"padding": policy.base_policy.config.pad_language_to}})
    remap = {"observation.images.image": "observation.images.camera1",
             "observation.images.image2": "observation.images.camera2"}
    model = policy.base_policy.model

    def fetch(it_):
        raw = next(it_)
        meta = {k: raw.pop(k) for k in ("episode_ids", "timesteps", "future_states", "future_valid")}
        raw = {remap.get(k, k): v for k, v in raw.items()}
        b = pre(raw); b.update(meta)
        return b

    di = iter(loader)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    print(f"device={dev}  batch={16*8}  warming up (3 iters)...")
    for _ in range(3):
        b = fetch(di)
        with amp:
            loss, _ = policy.forward(b)
        loss.backward(); policy.zero_grad(set_to_none=True)
    sync()

    N = 5
    T = {k: 0.0 for k in ("fetch", "embed_prefix", "vlm_layers", "expert_loss", "backward", "total")}
    for _ in range(N):
        t0 = time.perf_counter(); b = fetch(di); sync(); t1 = time.perf_counter()
        T["fetch"] += t1 - t0
        # component timing via a manual forward mirroring the monolithic path
        images, img_masks = policy.base_policy.prepare_images(b)
        state = policy.base_policy.prepare_state(b)
        lt = b["observation.language.tokens"]; lm = b["observation.language.attention_mask"]
        actions = policy.base_policy.prepare_action(b)
        policy._current_episode_ids = list(b["episode_ids"]); policy._current_timesteps = list(b["timesteps"])
        policy.feature_extractor.set_callback(policy._memory_callback)
        sync(); a0 = time.perf_counter()
        with amp:
            pe, pp, pa = model.embed_prefix(images, img_masks, lt, lm, state=state)
        sync(); a1 = time.perf_counter()
        # full step for the rest via policy.forward on the same batch (fair total)
        policy.feature_extractor.set_callback(None)
        sync(); s0 = time.perf_counter()
        with amp:
            loss, _ = policy.forward(b)
        sync(); s1 = time.perf_counter()
        loss.backward(); sync(); s2 = time.perf_counter()
        policy.zero_grad(set_to_none=True)
        T["embed_prefix"] += a1 - a0
        T["expert_loss"] += s1 - s0            # full fwd (incl embed+layers+expert)
        T["backward"] += s2 - s1
        T["total"] += (t1 - t0) + (s2 - s0)

    print("\n=== per-step timing (mean of %d) ===" % N)
    for k in ("fetch", "embed_prefix", "expert_loss", "backward", "total"):
        print(f"  {k:14s} {T[k]/N:7.3f}s")
    fwd = T["expert_loss"] / N; emb = T["embed_prefix"] / N
    print(f"\n  embed_prefix (vision+tokenize) is {100*emb/fwd:4.0f}% of the forward")
    print(f"  => caching prefix embeddings removes ~{emb:.2f}s/step + all video decode from fetch ({T['fetch']/N:.2f}s)")


if __name__ == "__main__":
    main()
