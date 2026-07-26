"""Factory for building :class:`MemorySmolVLAPolicy`.

Single code path: vanilla SmolVLA base checkpoint, action expert trained
from scratch, memory trained from scratch (spec §5.1).
"""

from __future__ import annotations

import logging

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy

logger = logging.getLogger(__name__)


# LIBERO dataset has state=(8,) and action=(7,) — includes the parallel-jaw
# gripper dim SO-100 lacks. The hub ``lerobot/smolvla_base`` config.json bakes
# in SO-100 shapes (6, 6), which causes ``_get_action_chunk`` to trim inference
# output to 6 dims, silently dropping LIBERO's gripper. Mirrors what
# ``lerobot.policies.factory.make_policy(ds_meta=...)`` does when input/output
# features are empty — we apply it unconditionally because the hub config
# populates them with SO-100 values.
_LIBERO_DATASET_REPO_ID = "HuggingFaceVLA/libero"


def build_policy(
    *,
    base_checkpoint: str = "lerobot/smolvla_base",
    num_vlm_layers: int = 16,
    injection_layer: int = 15,
    inject_before: bool = True,
    mem_length: int = 8,
    retrieval_layers: int = 2,
    use_timestep_pe: bool = True,
    consolidate_type: str = "tome",
    update_fused: bool = False,
    dataloader_type: str = "group",
    group_size: int = 8,
    compression: str = "none",
    n_slots: int = 4,
    aux_loss_weight: float = 0.0,
    bptt_memory: bool = False,
    reasoning_enabled: bool = False,
    reasoning_injection_layer: int = 8,
    reasoning_n_slots: int = 8,
    reasoning_bptt: bool = True,
    coconut_enabled: bool = False,
    num_thoughts: int = 0,
    coconut_adapter_layers: tuple = (12, 13, 14, 15),
    coconut_feedback_hidden: int = 1920,
    coconut_expert_visibility: str = "final_only",
    coconut_feedback_gate_init: float = -1.0,
    warm_start: str | None = None,
    policy_overrides: dict | None = None,
) -> MemorySmolVLAPolicy:
    """Build a :class:`MemorySmolVLAPolicy`.

    Loads ``base_checkpoint`` via :meth:`SmolVLAPolicy.from_pretrained`
    (so normalization stats, config, and the VLM backbone are
    initialized correctly), then resets the action expert
    (``lm_expert`` + action projection heads) to a fresh random init
    per spec §5.1. Memory modules are always initialized from scratch.

    ``policy_overrides`` (optional) overrides individual
    :class:`SmolVLAConfig` fields after loading — use it to force parity
    with a reference recipe when the hub's saved ``config.json`` drifts
    from what we want. Architectural fields that affect weight shapes
    (``num_vlm_layers``, ``num_expert_layers``, ``expert_width_multiplier``,
    ``self_attn_every_n_layers``, ``vlm_model_name``) cannot be safely
    overridden after ``from_pretrained`` and will raise if requested.
    """
    logger.info(
        "Building SmolVLA from %s: num_vlm_layers=%d, action expert reinitialized",
        base_checkpoint, num_vlm_layers,
    )
    base_policy = SmolVLAPolicy.from_pretrained(base_checkpoint)
    if base_policy.config.num_vlm_layers != num_vlm_layers:
        raise ValueError(
            f"Checkpoint {base_checkpoint} has num_vlm_layers="
            f"{base_policy.config.num_vlm_layers}, config requested {num_vlm_layers}."
        )

    _apply_libero_feature_shapes(base_policy)

    if policy_overrides:
        _apply_policy_overrides(base_policy, policy_overrides)

    _reinit_action_expert(base_policy)

    policy = MemorySmolVLAPolicy(
        base_policy=base_policy,
        injection_layer=injection_layer,
        inject_before=inject_before,
        mem_length=mem_length,
        retrieval_layers=retrieval_layers,
        use_timestep_pe=use_timestep_pe,
        consolidate_type=consolidate_type,
        update_fused=update_fused,
        dataloader_type=dataloader_type,
        group_size=group_size,
        compression=compression,
        n_slots=n_slots,
        state_dim=base_policy.config.input_features["observation.state"].shape[0],
        aux_loss_weight=aux_loss_weight,
        bptt_memory=bptt_memory,
        reasoning_enabled=reasoning_enabled,
        reasoning_injection_layer=reasoning_injection_layer,
        reasoning_n_slots=reasoning_n_slots,
        reasoning_bptt=reasoning_bptt,
        coconut_enabled=coconut_enabled,
        num_thoughts=num_thoughts,
        coconut_adapter_layers=coconut_adapter_layers,
        coconut_feedback_hidden=coconut_feedback_hidden,
        coconut_expert_visibility=coconut_expert_visibility,
        coconut_feedback_gate_init=coconut_feedback_gate_init,
    )

    if warm_start:
        _warm_start(policy, warm_start)

    n_trainable = sum(p.numel() for p in policy.trainable_parameters())
    n_total = sum(p.numel() for p in policy.parameters())
    n_memory = sum(p.numel() for p in policy.mem_bank.parameters())
    logger.info(
        "Policy built: %d / %d params trainable (%.1f%%); memory=%d",
        n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1), n_memory,
    )
    return policy


def _warm_start(policy: MemorySmolVLAPolicy, checkpoint: str) -> None:
    """Best-effort transfer of shared weights from a prior checkpoint.

    Fast-iteration lever: init the trained-from-scratch parts (action expert +
    working-memory bank) from a prior run so a new architecture fine-tunes in
    ~15-20k steps instead of ~60k. Only *matching* keys transfer (strict=False);
    genuinely new modules (e.g. V10's reasoning@8 bank, Coconut) stay fresh.

    V10 warm-starts from V7 (its mean-pool ``mem_bank`` and SmolVLA expert both
    match key-for-key; ``reasoning_bank`` / coconut are new). Reports exactly
    what transferred vs. stayed fresh so a silent shape/name mismatch is loud.
    """
    import torch

    ck = torch.load(checkpoint, map_location="cpu")
    src = ck.get("policy_state_dict", ck)
    own = policy.state_dict()
    transfer = {k: v for k, v in src.items()
                if k in own and own[k].shape == v.shape}
    fresh = [k for k in own if k not in transfer]
    skipped = [k for k in src if k not in own or (k in own and own[k].shape != src[k].shape)]
    policy.load_state_dict(transfer, strict=False)
    logger.info(
        "warm_start from %s @ step %s: transferred %d/%d tensors; %d stay fresh; "
        "%d source tensors skipped (name/shape mismatch).",
        checkpoint, ck.get("step", "?"), len(transfer), len(own), len(fresh), len(skipped),
    )
    # Surface which trainable *modules* are fresh (the real signal for the user).
    def _fresh_module(prefix):
        return any(k.startswith(prefix) for k in fresh)
    for name in ("reasoning_bank", "coconut_seed", "coconut_feedback",
                 "coconut_adapters", "mem_bank"):
        if hasattr(policy, name) and getattr(policy, name) is not None:
            logger.info("  %-16s : %s", name,
                        "FRESH" if _fresh_module(name + ".") else "warm-started")


_ARCHITECTURAL_POLICY_FIELDS = frozenset({
    "num_vlm_layers",
    "num_expert_layers",
    "expert_width_multiplier",
    "self_attn_every_n_layers",
    "vlm_model_name",
})


def _apply_policy_overrides(base_policy: SmolVLAPolicy, overrides: dict) -> None:
    for key, value in overrides.items():
        if not hasattr(base_policy.config, key):
            raise ValueError(f"Unknown SmolVLAConfig field in policy_overrides: {key!r}")
        if key in _ARCHITECTURAL_POLICY_FIELDS:
            raise ValueError(
                f"policy_overrides[{key!r}] is architectural and cannot be applied "
                f"after from_pretrained (weight shapes would diverge)."
            )
        current = getattr(base_policy.config, key)
        if current != value:
            logger.info("policy_overrides: %s: %r -> %r", key, current, value)
            setattr(base_policy.config, key, value)


def _apply_libero_feature_shapes(base_policy: SmolVLAPolicy) -> None:
    """Rewrite ``cfg.input_features['observation.state']`` and
    ``cfg.output_features['action']`` from the LIBERO dataset's metadata.

    The hub config bakes in SO-100 shapes (state=6, action=6). Model weights
    are shape-invariant (``state_proj`` / ``action_out_proj`` are sized to
    ``max_state_dim`` / ``max_action_dim`` = 32), so rewriting after
    ``from_pretrained`` is safe. The shapes affect inference-time slicing
    (``_get_action_chunk`` trims to ``action_feature.shape[0]``) and
    normalizer feature registration.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    try:
        # lerobot >= 0.5
        from lerobot.datasets.feature_utils import dataset_to_policy_features
    except ImportError:
        # lerobot <= 0.4
        from lerobot.datasets.utils import dataset_to_policy_features

    meta = LeRobotDatasetMetadata(_LIBERO_DATASET_REPO_ID)
    ds_feats = dataset_to_policy_features(meta.features)
    cfg = base_policy.config
    before_state = cfg.input_features["observation.state"].shape
    before_action = cfg.output_features["action"].shape
    cfg.input_features["observation.state"] = ds_feats["observation.state"]
    cfg.output_features["action"] = ds_feats["action"]
    logger.info(
        "Overrode policy config shapes from LIBERO metadata: "
        "observation.state %s -> %s, action %s -> %s",
        before_state, cfg.input_features["observation.state"].shape,
        before_action, cfg.output_features["action"].shape,
    )


def _reinit_action_expert(base_policy: SmolVLAPolicy) -> None:
    """Reinitialize ``lm_expert`` and every ``action_*`` projection head.

    ``SmolVLAPolicy.from_pretrained`` loads a trained expert; spec §5.1
    wants a freshly initialized one. Any leaf module exposing
    ``reset_parameters()`` is re-initialized.
    """
    def _reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    vwe = base_policy.model.vlm_with_expert
    vwe.lm_expert.apply(_reset)
    for name in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        mod = getattr(base_policy.model, name, None)
        if mod is not None:
            mod.apply(_reset)
