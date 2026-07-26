"""Precompute the prefix-embedding cache for a config (fast-iteration lever).

Iterates every LeRobot dataset row once, runs the SAME preprocessing + policy
``embed_prefix`` as training (augmentation OFF, ``pad_language_to=max_length`` so
shapes are fixed), and writes fp16 embeddings + masks to a memmap keyed by row
index. Training then reads these back and skips the SigLIP vision tower (~76% of
a step; profiled). Run once per (dataset, base model, padding) — NOT per arm.

Usage:
    python scripts/precompute_prefix.py --config configs/memvla_libero_v10.yaml \
        --out cache/prefix_libero_maxlen --batch 64 [--limit N] [--verify]

--verify (no write): rebuilds a few rows live and checks bit-agreement with an
existing cache. GPU required (vision tower). Safe to run while nothing else uses
the GPU; do NOT run concurrently with training (contends for the GPU).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t
from memory_smolvla.data.group_loader import _collate, LIBERO_REPO_ID

REMAP = {"observation.images.image": "observation.images.camera1",
         "observation.images.image2": "observation.images.camera2"}


def build(cfg, device):
    from memory_smolvla.policy.builder import build_policy
    from lerobot.policies.factory import make_pre_post_processors
    pc = cfg["policy"]
    ov = dict(pc.get("overrides") or {})
    ov["pad_language_to"] = "max_length"                 # fixed-shape prefix
    policy = build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=pc["injection_layer"],
        inject_before=pc["inject_before"], mem_length=pc["mem_length"],
        retrieval_layers=pc["retrieval_layers"], consolidate_type=pc["consolidate_type"],
        dataloader_type="group", group_size=8, compression=pc.get("compression", "mean_pool"),
        n_slots=pc.get("n_slots", 4), reasoning_enabled=bool(pc.get("reasoning_enabled", False)),
        reasoning_injection_layer=int(pc.get("reasoning_injection_layer", 8)),
        reasoning_n_slots=int(pc.get("reasoning_n_slots", 8)),
        warm_start=None, policy_overrides=ov,
    ).to(device).eval()
    pre, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=pc["preprocessor_path"],
        preprocessor_overrides={"device_processor": {"device": str(device)},
                                "tokenizer_processor": {"padding": "max_length"}})
    return policy, pre


class _RowDataset(torch.utils.data.Dataset):
    """Map-style view over LeRobot rows that returns remapped raw frames, so a
    DataLoader can decode video in parallel workers (the precompute bottleneck)."""

    def __init__(self, repo_id, deltas, n):
        self.repo_id, self.deltas, self.n = repo_id, deltas, n
        self._ds = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._ds is None:  # build per-worker (LeRobotDataset isn't fork-friendly)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self._ds = LeRobotDataset(repo_id=self.repo_id, delta_timestamps=self.deltas,
                                      root=None, image_transforms=None)
        return {REMAP.get(k, k): v for k, v in self._ds[int(i)].items()}


def _embed_collated(policy, pre, batch, device):
    batch = pre(batch)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
        images, img_masks = policy.base_policy.prepare_images(batch)
        state = policy.base_policy.prepare_state(batch)
        lt = batch["observation.language.tokens"]; lm = batch["observation.language.attention_mask"]
        embs, pad, att = policy.base_policy.model.embed_prefix(images, img_masks, lt, lm, state=state)
    return embs.float(), pad, att


def embed_rows(policy, pre, dataset, idxs, device):
    """Live embed_prefix for a list of row indices → (embs, pad, att) tensors."""
    frames = [dataset[int(i)] for i in idxs]
    batch = _collate(frames)
    batch = {REMAP.get(k, k): v for k, v in batch.items()}
    batch = pre(batch)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
        images, img_masks = policy.base_policy.prepare_images(batch)
        state = policy.base_policy.prepare_state(batch)
        lt = batch["observation.language.tokens"]; lm = batch["observation.language.attention_mask"]
        embs, pad, att = policy.base_policy.model.embed_prefix(images, img_masks, lt, lm, state=state)
    return embs.float(), pad, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12, help="parallel decode workers")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows (smoke)")
    ap.add_argument("--verify", action="store_true", help="check an existing cache, don't write")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = t.load_config(args.config)
    policy, pre = build(cfg, device)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from memory_smolvla.data.group_loader import GroupedEpisodeLoader, DatasetConfig  # noqa
    from memory_smolvla.data.dataset_config import DatasetConfig as DCfg
    dc = cfg["dataset"]
    ds_cfg = DCfg(delta_timestamps=dc.get("delta_timestamps", {}), split="train", local_cache_dir=None)
    # resolve deltas exactly like the loader
    deltas = GroupedEpisodeLoader._resolve_deltas(ds_cfg, policy.base_policy.config)
    dataset = LeRobotDataset(repo_id=LIBERO_REPO_ID, delta_timestamps=deltas,
                             root=None, image_transforms=None)
    N = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    # probe L
    e0, p0, a0 = embed_rows(policy, pre, dataset, [0], device)
    L, D = e0.shape[1], e0.shape[2]
    print(f"N={N} rows, prefix L={L}, D={D}, dtype=fp16 -> ~{N*L*D*2/1e9:.1f} GB")

    if args.verify:
        from memory_smolvla.data.prefix_cache import PrefixCache
        cache = PrefixCache(args.out)
        cache.check_compatible(policy)
        import random
        rng = random.Random(0)
        probe = sorted(rng.sample(range(cache.N), k=min(24, cache.N)))
        emb_c, pad_c, att_c = cache.lookup(probe, device, dtype=torch.float32)
        emb_l, pad_l, att_l = embed_rows(policy, pre, dataset, probe, device)
        derr = (emb_c - emb_l).abs().max().item()
        pad_ok = torch.equal(pad_c, pad_l.bool()); att_ok = torch.equal(att_c, att_l.long())
        print(f"VERIFY on {len(probe)} rows: max|emb_cache-emb_live|={derr:.3e} (fp16 eps ~1e-3), "
              f"pad_match={pad_ok} att_match={att_ok}")
        ok = derr < 3e-3 and pad_ok and att_ok
        print("VERIFY:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    from memory_smolvla.data.prefix_cache import open_writer
    meta = {"pad_language_to": "max_length", "image_transforms": False,
            "base_checkpoint": cfg["policy"]["base_checkpoint"], "repo_id": LIBERO_REPO_ID}
    emb_mm, pad_mm, att_mm = open_writer(args.out, N, L, D, meta)

    # Parallel video decode in workers keeps the GPU embed_prefix fed (serial
    # single-process decode starves it — profiled at 5 rows/s vs GPU-bound).
    row_ds = _RowDataset(LIBERO_REPO_ID, deltas, N)
    ctx = torch.multiprocessing.get_context("spawn")  # pyav isn't fork-safe
    loader = torch.utils.data.DataLoader(
        row_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=_collate, prefetch_factor=4 if args.workers else None,
        multiprocessing_context=ctx if args.workers else None,
    )
    t0 = time.perf_counter(); done = 0
    for batch in loader:
        n = batch["observation.state"].shape[0]
        embs, pad, att = _embed_collated(policy, pre, batch, device)
        sl = slice(done, done + n)
        emb_mm[sl] = embs.half().cpu().numpy()
        pad_mm[sl] = pad.to(torch.uint8).cpu().numpy()
        att_mm[sl] = att.to(torch.uint8).cpu().numpy()
        done += n
        if (done // args.batch) % 25 == 0:
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{N} rows  {rate:.0f} rows/s  ETA {(N-done)/max(rate,1e-6)/60:.1f} min", flush=True)
    emb_mm.flush(); pad_mm.flush(); att_mm.flush()
    print(f"DONE: wrote {done} rows to {args.out} in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
