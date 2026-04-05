"""Factory for building MemorySmolVLAPolicy in the correct training mode.

Handles the two checkpoint strategies:

- ``memory_only`` / ``expert_finetune``: Load full ``lerobot/smolvla_base``
  checkpoint (VLM + expert weights). Only valid when ``num_vlm_layers=16``.

- ``expert_scratch`` / ``expert_only_scratch``: Build from config with
  pretrained VLM backbone but a randomly-initialized action expert.
  Required when ``num_vlm_layers != 16`` (e.g. 8 or 24 layers).
"""

from __future__ import annotations

import logging

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy

logger = logging.getLogger(__name__)


def build_policy(
    training_mode: str,
    *,
    num_vlm_layers: int = 16,
    base_checkpoint: str = "lerobot/smolvla_base",
    injection_layer: int = 8,
    bank_max_size: int = 16,
    retrieval_n_heads: int = 4,
    gate_hidden_dim: int = 256,
    inject_before: bool = False,
    # --- New options ---
    memory_backend: str = "episodic",
    use_compressor: bool = False,
    compressor_n_slots: int = 8,
    use_write_gate: bool = False,
    use_multi_scale: bool = False,
    eviction: str = "fifo",
    alpha_target: float = 0.2,
    alpha_reg_weight: float = 0.0,
    step_increment: int = 1,
    gate_init_bias: float = -5.0,
) -> MemorySmolVLAPolicy:
    """Build a ``MemorySmolVLAPolicy`` for the requested training mode.

    Args:
        training_mode: One of ``"memory_only"``, ``"expert_finetune"``,
            ``"expert_scratch"``, or ``"expert_only_scratch"``.
        num_vlm_layers: Number of VLM transformer layers.
        base_checkpoint: HuggingFace repo ID of the pretrained SmolVLA.
        injection_layer: VLM layer index after which memory fires.
        bank_max_size: Max entries in the memory bank.
        retrieval_n_heads: Attention heads in cross-attention retrieval.
        gate_hidden_dim: Hidden dim of the sigmoid gate MLP.
        inject_before: If True, inject before the layer's attention.
        memory_backend: ``"episodic"`` (bank + cross-attention) or
            ``"working"`` (GRU recurrent state).
        use_compressor: If True, compress prefix before bank writes.
        compressor_n_slots: Number of compressed slots per entry.
        use_write_gate: If True, learn when to write to the bank.
        use_multi_scale: If True, use multi-scale memory bank.
        eviction: Bank eviction strategy — ``"fifo"`` or ``"consolidate"``.
        alpha_target: Target alpha for gate regularization.
        alpha_reg_weight: Weight for gate alpha regularization loss.
        step_increment: Timestamp increment per callback invocation.

    Returns:
        A ``MemorySmolVLAPolicy`` with parameters frozen/unfrozen
        according to ``training_mode``.
    """
    _pretrained_modes = {"memory_only", "expert_finetune"}
    _scratch_modes = {"expert_scratch", "expert_only_scratch"}

    if training_mode in _pretrained_modes and num_vlm_layers != 16:
        raise ValueError(
            f"training_mode={training_mode!r} requires loading the pretrained "
            f"SmolVLA checkpoint, which was trained with num_vlm_layers=16. "
            f"Got num_vlm_layers={num_vlm_layers}. "
            f"Use 'expert_scratch' or 'expert_only_scratch' for variable layer counts."
        )

    if training_mode in _pretrained_modes:
        logger.info(
            "Loading pretrained SmolVLA checkpoint: %s", base_checkpoint
        )
        base_policy = SmolVLAPolicy.from_pretrained(base_checkpoint)
    else:
        logger.info(
            "Building SmolVLA from config: num_vlm_layers=%d, random expert init",
            num_vlm_layers,
        )
        cfg = SmolVLAConfig.from_pretrained(base_checkpoint)
        cfg.num_vlm_layers = num_vlm_layers
        cfg.load_vlm_weights = True
        cfg.train_expert_only = True
        base_policy = SmolVLAPolicy(cfg)

    policy = MemorySmolVLAPolicy(
        base_policy=base_policy,
        injection_layer=injection_layer,
        bank_max_size=bank_max_size,
        retrieval_n_heads=retrieval_n_heads,
        gate_hidden_dim=gate_hidden_dim,
        training_mode=training_mode,
        inject_before=inject_before,
        memory_backend=memory_backend,
        use_compressor=use_compressor,
        compressor_n_slots=compressor_n_slots,
        use_write_gate=use_write_gate,
        use_multi_scale=use_multi_scale,
        eviction=eviction,
        alpha_target=alpha_target,
        alpha_reg_weight=alpha_reg_weight,
        step_increment=step_increment,
        gate_init_bias=gate_init_bias,
    )

    n_trainable = sum(p.numel() for p in policy.trainable_parameters())
    n_total = sum(p.numel() for p in policy.parameters())
    logger.info(
        "Policy built: %d / %d parameters trainable (%.1f%%)",
        n_trainable,
        n_total,
        100.0 * n_trainable / max(n_total, 1),
    )
    return policy
