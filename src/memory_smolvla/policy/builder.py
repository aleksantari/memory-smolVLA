"""Factory for building :class:`MemorySmolVLAPolicy`.

Single code path: vanilla SmolVLA base checkpoint, action expert trained
from scratch, memory trained from scratch (spec §5.1).
"""

from __future__ import annotations

import logging

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy

logger = logging.getLogger(__name__)


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
) -> MemorySmolVLAPolicy:
    """Build a :class:`MemorySmolVLAPolicy`.

    Loads ``base_checkpoint`` via :meth:`SmolVLAPolicy.from_pretrained`
    (so normalization stats, config, and the VLM backbone are
    initialized correctly), then resets the action expert
    (``lm_expert`` + action projection heads) to a fresh random init
    per spec §5.1. Memory modules are always initialized from scratch.
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
    )

    n_trainable = sum(p.numel() for p in policy.trainable_parameters())
    n_total = sum(p.numel() for p in policy.parameters())
    n_memory = sum(p.numel() for p in policy.mem_bank.parameters())
    logger.info(
        "Policy built: %d / %d params trainable (%.1f%%); memory=%d",
        n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1), n_memory,
    )
    return policy


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
