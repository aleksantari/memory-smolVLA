"""Causal ablation: zero out memory_proj.weight in a checkpoint and save the
result as a new .pt file. Lets us eval "v5_meanpool_v4hp with memory pathway
neutralized" through the same eval pipeline. If the ablated checkpoint
performs identically to v5_meanpool_v4hp, memory_proj was already effectively
a no-op (already-trained-to-be-tiny). If it performs identically to base
smolvla_libero, the small memory_proj had been actively hurting and zeroing
it recovers baseline.

Usage:
    python scripts/diag_zero_memory_proj.py \\
        --in checkpoints/v5_meanpool_v4hp/final.pt \\
        --out checkpoints/v5_meanpool_v4hp_zeroproj/final.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt["policy_state_dict"]

    target_keys = [k for k in sd.keys() if k.startswith("memory_proj.")]
    if not target_keys:
        raise SystemExit(f"No memory_proj.* keys found in {args.inp}")

    for k in target_keys:
        before = sd[k]
        sd[k] = torch.zeros_like(before)
        logging.info("Zeroed %s  shape=%s  prev||·||=%.4f", k, list(before.shape), before.float().norm().item())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    logging.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
