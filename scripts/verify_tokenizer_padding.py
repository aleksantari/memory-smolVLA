"""Verify tokenizer padding mode matches between memvla and baseline_v2.

The memvla training recipe loads its preprocessor pipeline from
``lerobot/smolvla_base`` via ``make_pre_post_processors(pretrained_path=...)``.
That path deserializes a saved ``policy_preprocessor.json`` — any
``pad_language_to`` override on :class:`SmolVLAConfig` is NOT consulted.
If the hub's saved tokenizer step differs from baseline_v2's saved one
(e.g. ``max_length`` vs ``longest``), training silently drifts from the
reference run.

This script runs the exact same loader path ``scripts/train.py`` uses
for memvla — including the preprocessor override dict threaded through —
and compares it against baseline_v2's saved preprocessor. Output is a
verdict per preprocessor: ``longest``, ``max_length``, or ``ambiguous``.

Usage:
    python scripts/verify_tokenizer_padding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import memory_smolvla.data._video_compat  # noqa: F401,E402

from memory_smolvla.policy.builder import build_policy  # noqa: E402

# Prompts span well under tokenizer_max_length=48 so the two modes
# produce distinguishable padded widths.
PROMPTS = [
    "pick up the cup",
    "move the red block onto the table",
    "carefully stack the ceramic bowl on top of the wooden crate near the stove",
    "please carefully and precisely stack the small ceramic bowl on top of "
    "the wooden crate that is sitting near the stove on the left side",
]

TOKENIZER_MAX_LENGTH = 48


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base = load_config(str(cfg_path.parent / cfg.pop("_base_")))
        cfg = _deep_merge(base, cfg)
    return cfg


def _build_test_batch(prompts: list[str], policy_config) -> dict:
    """Minimal batch. Only ``task`` is consumed up to the tokenizer step;
    image/state placeholders exist only so earlier batch-dim and rename
    steps don't short-circuit on an empty observation dict."""
    batch_size = len(prompts)
    return {
        "observation.state": torch.zeros(batch_size, 8),
        "observation.images.image": torch.zeros(batch_size, 3, 256, 256),
        "task": list(prompts),
    }


def _derive_verdict(padded_width: int, lengths: list[int]) -> str:
    longest = max(lengths)
    if padded_width == longest and padded_width < TOKENIZER_MAX_LENGTH:
        return "longest"
    if padded_width == TOKENIZER_MAX_LENGTH and padded_width > longest:
        return "max_length"
    return "ambiguous"


def _run_one(
    label: str,
    policy,
    preprocessor,
    prompts: list[str],
) -> str:
    """Run the preprocessor up to and including the tokenizer step.

    We stop before the normalizer step because its stats are baked in
    with specific state/action dims that differ between the hub and
    baseline_v2 checkpoints — irrelevant to the tokenizer verdict.
    """
    from lerobot.processor.core import TransitionKey

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")

    batch = _build_test_batch(prompts, policy.base_policy.config)

    tokenizer_idx = None
    for i, step in enumerate(preprocessor.steps):
        if step.__class__.__name__ == "TokenizerProcessorStep":
            tokenizer_idx = i
            break
    if tokenizer_idx is None:
        print("ERROR: no TokenizerProcessorStep found in pipeline.")
        return "ambiguous"

    transition = None
    for i, t in enumerate(preprocessor.step_through(batch)):
        transition = t
        if i - 1 == tokenizer_idx:  # step_through yields initial + one-per-step
            break

    obs = transition.get(TransitionKey.OBSERVATION) or {}
    tokens_key = "observation.language.tokens"
    mask_key = "observation.language.attention_mask"
    if tokens_key not in obs or mask_key not in obs:
        print(f"ERROR: expected keys {tokens_key} / {mask_key} missing.")
        print(f"Got keys: {sorted(obs.keys())}")
        return "ambiguous"

    tokens = obs[tokens_key].detach().cpu()
    mask = obs[mask_key].detach().cpu().bool()
    padded_width = tokens.shape[1]
    lengths = mask.sum(dim=1).tolist()

    for i, (prompt, tok, m, length) in enumerate(zip(prompts, tokens, mask, lengths)):
        print(f"\nsample {i}: {prompt!r}")
        print(f"  token_ids    : {tok.tolist()}")
        print(f"  attention_mask: {m.int().tolist()}")
        print(f"  non-pad length: {length}  padded width: {padded_width}")

    verdict = _derive_verdict(padded_width, lengths)
    same_width = all(True for _ in lengths)  # padded width is uniform by construction
    print(f"\npadded_width={padded_width}, per-sample lengths={lengths}, "
          f"max_length_cap={TOKENIZER_MAX_LENGTH}, uniform={same_width}")
    print(f"VERDICT: {verdict}")
    return verdict


def _build_memvla(cfg: dict, apply_override: bool):
    """Build memvla policy + preprocessor the same way scripts/train.py does."""
    from lerobot.policies.factory import make_pre_post_processors

    policy_cfg = cfg.get("policy", {})
    trainer_cfg = cfg.get("trainer", {})

    policy = build_policy(
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        injection_layer=policy_cfg.get("injection_layer", 15),
        inject_before=policy_cfg.get("inject_before", True),
        mem_length=policy_cfg.get("mem_length", 4),
        retrieval_layers=policy_cfg.get("retrieval_layers", 2),
        use_timestep_pe=policy_cfg.get("use_timestep_pe", True),
        consolidate_type=policy_cfg.get("consolidate_type", "tome"),
        update_fused=policy_cfg.get("update_fused", False),
        dataloader_type=policy_cfg.get("dataloader_type", "group"),
        group_size=policy_cfg.get("group_size", 8),
        policy_overrides=policy_cfg.get("overrides") or None,
    )

    pre_overrides = {
        "device_processor": {"device": trainer_cfg.get("device", "cpu")},
    }
    if apply_override:
        pre_overrides["tokenizer_processor"] = {
            "padding": policy.base_policy.config.pad_language_to,
        }

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        preprocessor_overrides=pre_overrides,
    )
    return policy, preprocessor


def _build_baseline(policy):
    """Build baseline_v2 preprocessor (reuse memvla policy for config/features)."""
    from lerobot.policies.factory import make_pre_post_processors

    baseline_ckpt = (
        PROJECT_ROOT
        / "outputs/libero_baseline_v2/checkpoints/last/pretrained_model"
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=str(baseline_ckpt),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return preprocessor


def main():
    cfg_path = PROJECT_ROOT / "configs/memvla_libero.yaml"
    cfg = load_config(str(cfg_path))

    print("=" * 70)
    print("Tokenizer padding verification")
    print("=" * 70)
    print(f"Config: {cfg_path}")
    print(f"Prompts ({len(PROMPTS)}):")
    for i, p in enumerate(PROMPTS):
        print(f"  [{i}] {p!r}")

    # Build once (expensive) and reuse for all three variants below.
    memvla_policy, memvla_pre_nooverride = _build_memvla(cfg, apply_override=False)

    verdict_memvla_nooverride = _run_one(
        "MEMVLA preprocessor (no tokenizer override) — raw hub path",
        memvla_policy,
        memvla_pre_nooverride,
        PROMPTS,
    )

    baseline_pre = _build_baseline(memvla_policy)
    verdict_baseline = _run_one(
        "BASELINE_V2 preprocessor (saved pretrained_model)",
        memvla_policy,
        baseline_pre,
        PROMPTS,
    )

    _, memvla_pre_override = _build_memvla(cfg, apply_override=True)
    verdict_memvla_override = _run_one(
        "MEMVLA preprocessor (tokenizer override applied)",
        memvla_policy,
        memvla_pre_override,
        PROMPTS,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  baseline_v2                       : VERDICT = {verdict_baseline}")
    print(f"  memvla (no tokenizer override)    : VERDICT = {verdict_memvla_nooverride}")
    print(f"  memvla (tokenizer override on)    : VERDICT = {verdict_memvla_override}")
    aligned = verdict_baseline == verdict_memvla_override
    print(f"\nPost-fix alignment (baseline == memvla-with-override): {aligned}")


if __name__ == "__main__":
    main()
