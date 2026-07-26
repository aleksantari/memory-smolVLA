"""Precomputed prefix-embedding cache (fast-iteration lever).

The frozen SigLIP vision tower is ~76% of a training step (profiled: 5.9s of
10.4s). With image augmentation OFF, ``embed_prefix(images, lang, state)`` is a
deterministic function of the frame, so we precompute it once for every dataset
row and read it back at train time — skipping vision + tokenize + assembly.

Layout (keyed by LeRobot dataset row index):
  <dir>/prefix.f16   memmap float16  (N, L, D)   — embed_prefix embeddings
  <dir>/pad.u8       memmap uint8    (N, L)      — prefix_pad_masks
  <dir>/att.u8       memmap uint8    (N, L)      — prefix_att_masks
  <dir>/meta.json    {N, L, D, pad_language_to, image_transforms:false, ...}

Correctness contract: cached (prefix_embs, pad, att) must be *bit-exact* to the
live ``embed_prefix`` output for the same row (verified by
``scripts/precompute_prefix.py --verify`` and the equivalence test). A cache
whose meta does not match the current policy/config must never be used silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class PrefixCache:
    """Memory-mapped reader for precomputed prefix embeddings."""

    def __init__(self, cache_dir: str) -> None:
        self.dir = Path(cache_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No prefix cache meta at {meta_path}")
        self.meta = json.loads(meta_path.read_text())
        N, L, D = self.meta["N"], self.meta["L"], self.meta["D"]
        self.N, self.L, self.D = N, L, D
        self._emb = np.memmap(self.dir / "prefix.f16", dtype=np.float16, mode="r", shape=(N, L, D))
        self._pad = np.memmap(self.dir / "pad.u8", dtype=np.uint8, mode="r", shape=(N, L))
        self._att = np.memmap(self.dir / "att.u8", dtype=np.uint8, mode="r", shape=(N, L))

    def check_compatible(self, policy) -> None:
        """Fail loudly if the cache was built for a different policy/config."""
        cfg = policy.base_policy.config
        want = {
            "D": policy.d_model,
            "pad_language_to": cfg.pad_language_to,
            "image_transforms": False,
        }
        for k, v in want.items():
            if self.meta.get(k) != v:
                raise ValueError(
                    f"prefix cache mismatch on '{k}': cache={self.meta.get(k)!r} "
                    f"policy={v!r}. Rebuild the cache (scripts/precompute_prefix.py)."
                )

    def lookup(self, global_idxs, device, dtype=torch.float32):
        """Return (prefix_embs (B,L,D), pad_masks (B,L) bool, att_masks (B,L)) for
        the given LeRobot row indices, in order."""
        idx = np.asarray([int(i) for i in global_idxs])
        emb = torch.from_numpy(np.ascontiguousarray(self._emb[idx])).to(device=device, dtype=dtype)
        pad = torch.from_numpy(np.ascontiguousarray(self._pad[idx])).to(device=device).bool()
        att = torch.from_numpy(np.ascontiguousarray(self._att[idx])).to(device=device).long()
        return emb, pad, att


def open_writer(cache_dir: str, N: int, L: int, D: int, meta_extra: dict):
    """Create the memmap files + meta.json for writing; returns (emb, pad, att)."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    emb = np.memmap(d / "prefix.f16", dtype=np.float16, mode="w+", shape=(N, L, D))
    pad = np.memmap(d / "pad.u8", dtype=np.uint8, mode="w+", shape=(N, L))
    att = np.memmap(d / "att.u8", dtype=np.uint8, mode="w+", shape=(N, L))
    meta = {"N": N, "L": L, "D": D, **meta_extra}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return emb, pad, att
