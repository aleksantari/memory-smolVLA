"""Memory-augmented SmolVLA policy wrapper (MemoryVLA-port design).

Wraps a ``SmolVLAPolicy`` with a :class:`FullSeqMemBank` that stores the
full VLM output sequence per timestep. Memory is injected at a single
layer (default ``injection_layer=15``, the final cross-attn layer under
``self_attn_every_n_layers=2``) with ``inject_before=True`` so the
fusion lands on the residual-stream hidden state just before the
layer's own ``input_layernorm`` runs.

Training mode (single, no dispatch):
  - VLM backbone: frozen.
  - Action expert + ``action_out_proj``: trained from scratch.
  - ``FullSeqMemBank`` (retrieval blocks, timestep embedder, gate fusion):
    trained from scratch.

See ``memory_smolvla_implementation_spec.md`` §3.4 and §5.1.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn

from memory_smolvla.memory.full_seq_bank import FullSeqMemBank
from memory_smolvla.policy.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MemorySmolVLAPolicy(nn.Module):
    """Memory-augmented wrapper around a SmolVLAPolicy.

    Intercepts the VLM prefix residual stream at layer ``injection_layer``
    (``inject_before=True``), passes it through a :class:`FullSeqMemBank`,
    and returns the fused tensor to SmolVLA's own layer stack. Per-batch
    ``episode_ids`` and ``timesteps`` must be attached to the batch dict
    (see :class:`GroupedEpisodeLoader`) at training time; at inference
    time they default to ``[0]`` / the internal timestep counter.

    Args:
        base_policy: A ``SmolVLAPolicy`` instance.
        injection_layer: VLM layer index for memory injection.
        inject_before: Fire before ``input_layernorm`` at that layer.
        mem_length: Memory bank capacity per episode.
        retrieval_layers: Number of stacked ``CrossTransformerBlock`` layers.
        use_timestep_pe: Add sinusoidal temporal PE to bank keys.
        consolidate_type: ``"tome"`` (token-merge) or ``"fifo"``.
        update_fused: If ``True``, write fused tokens to the bank
            instead of raw. Default ``False`` matches spec §3.4.
        dataloader_type: ``"group"`` or ``"stream"``. Must match the
            loader's layout.
        group_size: Number of contiguous frames per group in a batch.
    """

    def __init__(
        self,
        base_policy,
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
        state_dim: int = 8,
        aux_loss_weight: float = 0.0,
        bptt_memory: bool = False,
    ) -> None:
        super().__init__()

        self.base_policy = base_policy
        self.aux_loss_weight = aux_loss_weight

        # Freeze VLM backbone; unfreeze action expert + action_out_proj.
        for param in self.base_policy.parameters():
            param.requires_grad = False
        vlm_with_expert = self.base_policy.model.vlm_with_expert
        for param in vlm_with_expert.lm_expert.parameters():
            param.requires_grad = True
        for param in self.base_policy.model.action_out_proj.parameters():
            param.requires_grad = True

        d_model = vlm_with_expert.config.text_config.hidden_size
        self.d_model = d_model
        self.injection_layer = injection_layer

        self.mem_bank = FullSeqMemBank(
            token_size=d_model,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            consolidate_type=consolidate_type,
            update_fused=update_fused,
            dataloader_type=dataloader_type,
            group_size=group_size,
            compression=compression,
            n_slots=n_slots,
            state_dim=state_dim,
            bptt_memory=bptt_memory,
        )

        self.feature_extractor = FeatureExtractor(
            vlm_with_expert=vlm_with_expert,
            injection_layer=injection_layer,
            inject_before=inject_before,
        )

        # Transient per-batch metadata consumed by ``_memory_callback``.
        # Populated by ``forward`` / ``predict_action_chunk`` / ``select_action``.
        self._current_episode_ids: list[Any] | None = None
        self._current_timesteps: list[int] | None = None
        self._infer_timestep: int = 0

        logger.info(
            "MemorySmolVLAPolicy: d_model=%d, injection_layer=%d, inject_before=%s, "
            "mem_length=%d, retrieval_layers=%d, consolidate=%s, "
            "dataloader_type=%s, group_size=%d",
            d_model, injection_layer, inject_before, mem_length, retrieval_layers,
            consolidate_type, dataloader_type, group_size,
        )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Training forward with memory augmentation.

        The batch must carry ``episode_ids`` and ``timesteps`` (length-B
        Python lists) produced by :class:`GroupedEpisodeLoader`.
        """
        episode_ids = batch.get("episode_ids")
        timesteps = batch.get("timesteps")
        if episode_ids is None or timesteps is None:
            raise KeyError(
                "Batch must contain 'episode_ids' and 'timesteps' "
                "(use GroupedEpisodeLoader)."
            )

        # Strip metadata before handing the batch to SmolVLA (it only
        # understands its own observation/action keys). future_states /
        # future_valid feed the V8 PTP aux loss, not the base policy.
        _META = ("episode_ids", "timesteps", "future_states", "future_valid")
        inner_batch = {k: v for k, v in batch.items() if k not in _META}

        self._current_episode_ids = list(episode_ids)
        self._current_timesteps = list(timesteps)
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            loss, loss_dict = self.base_policy.forward(
                inner_batch, noise=noise, time=time
            )
        finally:
            self.feature_extractor.set_callback(None)
            self._current_episode_ids = None
            self._current_timesteps = None

        scale = self.mem_bank.last_gate_scale()
        if scale is not None:
            loss_dict["gate_value_mean"] = scale.mean().item()
            loss_dict["gate_value_std"] = scale.std().item()

        # V8 PTP auxiliary loss: supervise the reasoning tokens to predict the
        # future proprioceptive state. No-op unless in reasoning mode with a
        # positive weight and the loader supplied future targets.
        if self.aux_loss_weight > 0 and "future_states" in batch:
            aux = self.mem_bank.aux_future_loss(
                batch["future_states"], batch["future_valid"]
            )
            if aux is not None:
                loss = loss + self.aux_loss_weight * aux
                loss_dict["aux_future_state_loss"] = aux.item()

        return loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        episode_id: Any = 0,
        **kwargs,
    ) -> Tensor:
        """Full-chunk inference. Pass ``episode_id`` per rollout."""
        self.eval()
        inner_batch = {k: v for k, v in batch.items() if k not in ("episode_ids", "timesteps")}
        B = _infer_batch_size(inner_batch)
        self._current_episode_ids = [episode_id] * B
        self._current_timesteps = [self._infer_timestep] * B
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            actions = self.base_policy.predict_action_chunk(
                inner_batch, noise=noise, **kwargs
            )
        finally:
            self.feature_extractor.set_callback(None)
            self._current_episode_ids = None
            self._current_timesteps = None
        self._infer_timestep += 1
        return actions

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        episode_id: Any = 0,
        **kwargs,
    ) -> Tensor:
        """Streaming single-action inference. Pass ``episode_id`` per rollout."""
        self.eval()
        inner_batch = {k: v for k, v in batch.items() if k not in ("episode_ids", "timesteps")}
        B = _infer_batch_size(inner_batch)
        self._current_episode_ids = [episode_id] * B
        self._current_timesteps = [self._infer_timestep] * B
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            action = self.base_policy.select_action(
                inner_batch, noise=noise, **kwargs
            )
        finally:
            self.feature_extractor.set_callback(None)
            self._current_episode_ids = None
            self._current_timesteps = None
        self._infer_timestep += 1
        return action

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset action queues and memory for a new episode."""
        self.base_policy.reset()
        self.reset_memory()

    def reset_memory(self) -> None:
        """Clear the memory bank and the inference timestep counter."""
        self.mem_bank.reset()
        self._infer_timestep = 0

    # ------------------------------------------------------------------
    # Memory callback
    # ------------------------------------------------------------------

    def _memory_callback(self, prefix_hidden: Tensor, layer_idx: int) -> Tensor:
        """Fuse the residual-stream prefix with retrieved memory.

        The callback receives the un-normalized ``inputs_embeds[0]`` at
        the start of the injection layer; SmolVLA's own
        ``input_layernorm`` runs on the returned tensor.
        """
        eids = self._current_episode_ids
        tsteps = self._current_timesteps
        if eids is None or tsteps is None:
            # Callback fired outside a forward wrapper (e.g. extra
            # denoising steps with an empty prefix). No-op.
            return prefix_hidden

        B = prefix_hidden.shape[0]
        if len(eids) != B or len(tsteps) != B:
            raise ValueError(
                f"episode_ids/timesteps length ({len(eids)}/{len(tsteps)}) "
                f"does not match prefix batch size ({B})."
            )

        orig_dtype = prefix_hidden.dtype
        fused = self.mem_bank.process_batch(prefix_hidden, eids, tsteps)
        return fused.to(orig_dtype)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters: expert + action_out_proj + memory."""
        vwe = self.base_policy.model.vlm_with_expert
        yield from vwe.lm_expert.parameters()
        yield from self.base_policy.model.action_out_proj.parameters()
        yield from self.mem_bank.parameters()

    def get_gate_statistics(self) -> dict[str, float]:
        """Return the most recent gate scale statistics, or {} if the bank
        has not yet been called (e.g. before the first forward of a rollout).

        Keys mirror training's ``loss_dict``: ``gate_value_mean`` /
        ``gate_value_std``.
        """
        scale = self.mem_bank.last_gate_scale()
        if scale is None:
            return {}
        return {
            "gate_value_mean": scale.mean().item(),
            "gate_value_std": scale.std().item(),
        }

    @property
    def config(self):
        """Proxy to base policy config for compatibility."""
        return self.base_policy.config


def _infer_batch_size(batch: dict) -> int:
    for v in batch.values():
        if isinstance(v, Tensor) and v.ndim >= 1:
            return v.shape[0]
    return 1
