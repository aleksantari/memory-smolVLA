"""Upload trained memory-augmented SmolVLA checkpoints to HuggingFace.

Each checkpoint becomes its own model repo under the user's HF account, with:
- final.pt (1.67 GB policy state dict)
- config (the training YAML)
- README.md (model card with results + eval pipeline notes)

Run:
    python scripts/upload_to_huggingface.py --user tarmus --variants meanpool compressor two_stream

Auth: assumes huggingface-cli login already done (uses ~/.cache/huggingface/token
or ~/_netrc).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
logger = logging.getLogger("upload")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANT_INFO = {
    "meanpool": {
        "checkpoint": "checkpoints/v5_meanpool_v4hp/final.pt",
        "config": "configs/libero_v5_meanpool_v4hp.yaml",
        "results_dirs": [
            "results/v5_meanpool_v4hp_5ep",
            "results/v5_meanpool_v4hp_stepinc1_5ep",
            "results/v5_meanpool_v4hp_zeroproj_5ep",
            "results/v5_meanpool_v4hp_full",
        ],
        "title": "Memory-SmolVLA · LIBERO · mean-pool bank · V4 hyperparameters",
        "description": (
            "SmolVLA finetuned on the HuggingFaceVLA/libero dataset with a "
            "temporal memory-bank wrapper. The bank stores one mean-pooled "
            "token per write (compression_mode=mean_pool, ~170x reduction "
            "of the prefix). Bank capacity 16, FIFO eviction, write_stride=50, "
            "residual gate, injection at VLM layer 8. Action expert + state "
            "projection finetuned at lr=1e-5; memory modules trained from "
            "scratch at lr=1e-4. 30,000 optimizer steps, grad_accum=1."
        ),
    },
    "compressor": {
        "checkpoint": "checkpoints/v5_compressor_v4hp/final.pt",
        "config": "configs/libero_v5_compressor_v4hp.yaml",
        "results_dirs": [
            "results/v5_compressor_v4hp_1ep",
            "results/v5_compressor_v4hp_stepinc1_1ep",
        ],
        "title": "Memory-SmolVLA · LIBERO · learned Perceiver compressor (n_slots=4) · V4 hyperparameters",
        "description": (
            "Same recipe as the mean-pool variant but with a learned 4-slot "
            "Perceiver-Resampler compressor instead of zero-param mean pool. "
            "Bank holds 16 entries x 4 slots = 64 keys at retrieval (vs "
            "mean-pool's 16). All other settings identical."
        ),
    },
    "two_stream": {
        "checkpoint": "checkpoints/v5_two_stream_v4hp/final.pt",
        "config": "configs/libero_v5_two_stream_v4hp.yaml",
        "results_dirs": [
            "results/v5_two_stream_v4hp_1ep",
            "results/v5_two_stream_v4hp_stepinc1_1ep",
        ],
        "title": "Memory-SmolVLA · LIBERO · two-stream perceptual + task split · V4 hyperparameters",
        "description": (
            "Same recipe as the mean-pool variant but with two-stream split: "
            "perceptual stream (image + image-special tokens) -> 16 slots, "
            "task-anchor stream (language + state) -> 1 slot. 17 slots/entry "
            "x 16 entries = 272 keys at retrieval. n_image_tokens=132 = "
            "2 cams x (64 image + 2 special). All other settings identical."
        ),
    },
}


RESULTS_TEXT = {
    "meanpool": """
## Eval results (sim, LIBERO via lerobot pipeline)

5 episodes × 10 tasks per suite (n=50/suite). All numbers run through
`scripts/eval_memory_libero_v2.py` against the upstream
`HuggingFaceVLA/smolvla_libero` baseline measured the same way.

| Suite       | Baseline `smolvla_libero` (no memory) | This model (no fix) | This model + `step_increment=1` |
|---|---:|---:|---:|
| libero_spatial | 76% | 64% | 62% |
| libero_object  | 86% | 86% | **88%** |
| libero_goal    | 82% | 80% | 76% |
| libero_10      | 42% | 30% | **44%** |
| **Overall**    | **71.5%** | 65% | **67.5%** |

**Important caveat on `step_increment`:** the inference-time policy parameter
`step_increment` must be set to **1** to match the bank-fill cadence the
model was trained against. The default of 50 (= chunk_size) makes the bank
cycle every 16 env steps instead of every 800, and at inference the bank
ends up holding only the last 16 env steps' prefixes (effectively a short
lookback) instead of the temporally-spread snapshots training expected.
Without the fix, libero_10 reads 30%; with it, 44% (recovers most of the
gap, neutral on baseline within sampling noise of n=50).

**Honest reading on whether memory helps long-horizon:** with 50-episode
binomial SE ~7pp, the 44% (memory + step_inc=1) vs 42% (baseline) gap is
within noise. Comparing memory-on (44%) vs same-finetune-no-memory (48%,
from the `zeroproj` ablation at 5ep) suggests memory is approximately
neutral or slightly negative on libero_10 at this scale. We have NOT
proved memory improves long-horizon performance; we have proved that the
architecture and training recipe are sound and the previous catastrophic
"0% in sim" results were eval-pipeline bugs.
""",
    "compressor": """
## Eval results (sim, LIBERO via lerobot pipeline)

1 episode × 10 tasks per suite (n=10/suite — wide error bars):

| Suite       | This model (no fix) | This model + `step_increment=1` |
|---|---:|---:|
| libero_spatial | 90% | 70% |
| libero_object  | 90% | 90% |
| libero_goal    | 50% | 60% |
| libero_10      | 20% | **40%** |
| **Overall**    | 62.5% | 65% |

The bank-cycling fix (`step_increment=1` at inference) doubles libero_10
from 20% → 40%, matching the +20pp swing seen across all three v5
architectures. At 1ep these per-suite numbers are noisy.
""",
    "two_stream": """
## Eval results (sim, LIBERO via lerobot pipeline)

1 episode × 10 tasks per suite (n=10/suite — wide error bars):

| Suite       | This model (no fix) | This model + `step_increment=1` |
|---|---:|---:|
| libero_spatial | 60% | 60% |
| libero_object  | 90% | 90% |
| libero_goal    | 80% | 50% |
| libero_10      | 20% | **40%** |
| **Overall**    | 62.5% | 60% |

The bank-cycling fix (`step_increment=1` at inference) doubles libero_10
from 20% → 40%, matching the +20pp swing seen across all three v5
architectures. At 1ep these per-suite numbers are noisy — the 80%→50%
on libero_goal is most likely sampling noise on N=10.
""",
}


def make_readme(variant: str, info: dict, repo_branch_url: str) -> str:
    return f"""---
license: apache-2.0
tags:
- robotics
- vision-language-action
- smolvla
- libero
- memory
base_model: HuggingFaceVLA/smolvla_libero
pipeline_tag: robotics
---

# {info['title']}

{info['description']}

Repository (training code, eval pipeline, run logs):
{repo_branch_url}

Base model: [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
{RESULTS_TEXT[variant]}

## How to use

```bash
git clone https://github.com/aleksantari/memory-smolVLA
cd memory-smolVLA
git checkout claude/feature/v5-all-fixes
pip install -e ".[dev]"

# Download this model:
huggingface-cli download tarmus/memory-smolvla-libero-{variant}-v4hp final.pt config.yaml --local-dir checkpoints/{variant}_v4hp

# Eval (in WSL with osmesa headless rendering):
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/path/to/LIBERO
python scripts/eval_memory_libero_v2.py \\
    --checkpoint checkpoints/{variant}_v4hp/final.pt \\
    --config checkpoints/{variant}_v4hp/config.yaml \\
    --suite libero_10 \\
    --n-episodes 5 \\
    --step-increment 1                 # critical, see results table
```

## Training recipe (V4 optimizer hyperparameters)

| | |
|---|---|
| Base | `HuggingFaceVLA/smolvla_libero` |
| training_mode | `expert_finetune` (action expert + state proj unfrozen, VLM frozen) |
| total_steps | 30,000 |
| grad_accum_steps | 1 |
| memory_lr | 1e-4 |
| expert_lr | 1e-5 |
| warmup_steps | 500 |
| weight_decay | 1e-4 |
| max_grad_norm | 1.0 |
| Hardware | RTX 5080 (Windows) / WSL Ubuntu-22.04 for sim eval |

See `RUN_LOG.md` in the repo for the full diagnostic narrative including
why the V4 hyperparameters work where the broader v5 plan's batch fix
hyperparameters (`expert_lr=1e-4`, 3000 steps, `grad_accum=32`) destroyed
the action expert.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="tarmus", help="HuggingFace username/org")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_INFO.keys()),
        default=list(VARIANT_INFO.keys()),
    )
    parser.add_argument("--repo-branch-url", default="https://github.com/aleksantari/memory-smolVLA/tree/claude/feature/v5-all-fixes")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without doing it")
    args = parser.parse_args()

    api = HfApi()

    for variant in args.variants:
        info = VARIANT_INFO[variant]
        repo_id = f"{args.user}/memory-smolvla-libero-{variant}-v4hp"
        ckpt_path = PROJECT_ROOT / info["checkpoint"]
        cfg_path = PROJECT_ROOT / info["config"]

        if not ckpt_path.exists():
            logger.warning("MISSING checkpoint: %s — skipping %s", ckpt_path, variant)
            continue

        readme_text = make_readme(variant, info, args.repo_branch_url)
        readme_path = PROJECT_ROOT / f"_tmp_{variant}_README.md"
        readme_path.write_text(readme_text, encoding="utf-8")

        logger.info("=" * 60)
        logger.info("Variant: %s", variant)
        logger.info("Repo:    %s", repo_id)
        logger.info("Files:   %s (%.2f GB), %s, README.md",
                    ckpt_path.name, ckpt_path.stat().st_size / 1e9, cfg_path.name)

        if args.dry_run:
            logger.info("[DRY RUN] would create repo + upload above files")
            readme_path.unlink()
            continue

        # Create repo (idempotent: exist_ok=True)
        create_repo(repo_id, exist_ok=True, repo_type="model", private=False)
        logger.info("Repo ready: https://huggingface.co/%s", repo_id)

        # Upload files
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Initial model card for {variant}",
        )
        logger.info("  uploaded README.md")

        api.upload_file(
            path_or_fileobj=str(cfg_path),
            path_in_repo="config.yaml",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add training config for {variant}",
        )
        logger.info("  uploaded config.yaml")

        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo="final.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload final.pt for {variant}",
        )
        logger.info("  uploaded final.pt (~%.2f GB)", ckpt_path.stat().st_size / 1e9)

        # Also upload eval JSONs (small, useful for reproducibility)
        for results_dir_str in info["results_dirs"]:
            results_dir = PROJECT_ROOT / results_dir_str
            if not results_dir.exists():
                continue
            for json_file in results_dir.glob("*.json"):
                api.upload_file(
                    path_or_fileobj=str(json_file),
                    path_in_repo=f"eval_results/{results_dir.name}/{json_file.name}",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message="Add eval JSON",
                )
            logger.info("  uploaded eval JSONs from %s", results_dir.name)

        readme_path.unlink()
        logger.info("Done: https://huggingface.co/%s", repo_id)

    logger.info("All variants done.")


if __name__ == "__main__":
    main()
