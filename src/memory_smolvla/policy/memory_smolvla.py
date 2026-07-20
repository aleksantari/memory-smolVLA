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


def append_delta_kv(base: dict, delta: dict) -> dict:
    """Concatenate a per-layer thought delta K/V onto the running cache (V10A).

    Explicit growth (the stock ``fill_kv_cache=False`` path cats into locals and
    never writes back — see ``compat_check.py`` findings D/E), so the cache grows
    by exactly ``delta``'s token count at every layer.
    """
    return {
        l: {
            "key_states": torch.cat(
                [base[l]["key_states"], delta[l]["key_states"]], dim=1
            ),
            "value_states": torch.cat(
                [base[l]["value_states"], delta[l]["value_states"]], dim=1
            ),
        }
        for l in base
    }


def merge_deltas(deltas: list[dict]) -> dict:
    """Concatenate a list of per-pass delta K/V dicts into one (all_visible mode)."""
    merged = deltas[0]
    for d in deltas[1:]:
        merged = append_delta_kv(merged, d)
    return merged


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
        coconut_enabled: bool = False,
        num_thoughts: int = 0,
        coconut_adapter_layers: tuple = (12, 13, 14, 15),
        coconut_feedback_hidden: int = 1920,
        coconut_expert_visibility: str = "final_only",
        coconut_feedback_gate_init: float = -1.0,
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

        # V10A — Coconut latent-thought reasoning (RevB). Disabled ⇒ the policy is
        # bit-exact V9. num_thoughts==0 also collapses to V9 regardless of the flag.
        self.coconut_enabled = bool(coconut_enabled)
        self.num_thoughts = int(num_thoughts)
        self.coconut_adapter_layers = tuple(coconut_adapter_layers)
        self.coconut_expert_visibility = str(coconut_expert_visibility)
        if self.coconut_enabled and self.num_thoughts > 0:
            from memory_smolvla.memory.reasoning import (
                CoconutFeedbackAdapter,
                CoconutThoughtSeed,
                LatentModeAdapter,
                MatchInputRMS,
            )

            self.coconut_seed = CoconutThoughtSeed(d=d_model)
            self.coconut_feedback = CoconutFeedbackAdapter(
                d=d_model, hid=coconut_feedback_hidden
            )
            with torch.no_grad():
                self.coconut_feedback.gate.fill_(float(coconut_feedback_gate_init))
            self.coconut_match_rms = MatchInputRMS()
            # RevB: pre-QKV latent adapters, one per adapter layer (zero-init ⇒
            # identity). Keyed by layer index for the thought pass.
            self.coconut_adapters = nn.ModuleDict(
                {str(l): LatentModeAdapter(d=d_model) for l in self.coconut_adapter_layers}
            )
            logger.info(
                "V10A Coconut ENABLED: K=%d, adapter_layers=%s, visibility=%s, "
                "feedback_hidden=%d, gate_init=%.1f",
                self.num_thoughts, self.coconut_adapter_layers,
                self.coconut_expert_visibility, coconut_feedback_hidden,
                coconut_feedback_gate_init,
            )

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

        # V10A: route through the cached Coconut path when thoughts are active.
        # K==0 / disabled falls through to the untouched V9 monolithic forward
        # (Step 0.5 proved cached ≡ monolithic, so the two are interchangeable).
        if self._coconut_active():
            return self._coconut_forward(batch, noise, time)

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
    # V10A — Coconut latent-thought orchestration (RevB Step 3)
    # ------------------------------------------------------------------

    def _coconut_active(self) -> bool:
        """True iff the Coconut thought path should run (K>0 and enabled)."""
        return self.coconut_enabled and self.num_thoughts > 0

    def _build_prefix_kv(self, images, img_masks, lang_tokens, lang_masks, state):
        """V9 memory-augmented prefix → (prefix_embs, prefix_pad, prefix_kv).

        Runs ``embed_prefix`` then the patched VLM forward with
        ``fill_kv_cache=True``; the memory callback fires at the injection layer
        so ``prefix_kv`` is the *augmented* (post-fusion) K/V, and the bank
        exposes ``_last_retrieved`` / ``_last_memory_slots`` for the seed. The
        bank advances exactly once here (one write per query — Invariant 4).
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        model = self.base_policy.model
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        _, prefix_kv = model.vlm_with_expert.forward(
            attention_mask=prefix_att2d, position_ids=prefix_pos,
            past_key_values=None, inputs_embeds=[prefix_embs, None],
            use_cache=True, fill_kv_cache=True,
        )
        return prefix_embs, prefix_pad, prefix_kv

    def run_coconut_reasoning(self, prefix_embs, prefix_pad, prefix_kv):
        """Run K thought passes; return (visible_delta_kv, action_position_offset).

        Thought phase lives in a *detached* world (the running cache is detached
        per-tensor so thought→memory gradients are structurally blocked); the
        returned ``visible`` delta is spliced onto the **live** ``prefix_kv`` by
        the caller so V9's BPTT through the prefix is preserved.
        """
        K = self.num_thoughts
        B = prefix_pad.shape[0]
        valid_prefix_len = prefix_pad.long().sum(dim=1)          # (B,)

        running_cache = {
            l: {k: v.detach() for k, v in d.items()} for l, d in prefix_kv.items()
        }
        memory_slots = self.mem_bank._last_memory_slots           # (B, 1, D)
        retrieved = self.mem_bank._last_retrieved                 # (B, L, D)
        adapters = {int(l): m for l, m in self.coconut_adapters.items()}

        z = self.coconut_seed(memory_slots, retrieved, prefix_pad)   # (B, 1, D)
        z = self.coconut_match_rms(z, prefix_embs, prefix_pad)

        thought_deltas = []
        for k in range(K):
            pos = valid_prefix_len[:, None] + k                   # (B, 1)
            ones = torch.ones(
                (B, k + 1), dtype=prefix_pad.dtype, device=prefix_pad.device
            )
            key_pad = torch.cat([prefix_pad, ones], dim=1)[:, None, :].bool()
            out = self.feature_extractor.vlm_thought_pass(
                token=z, past_key_values=running_cache,
                position_ids=pos, key_padding_mask=key_pad, adapters=adapters,
            )
            thought_deltas.append(out.delta_kv)
            running_cache = append_delta_kv(running_cache, out.delta_kv)
            if k < K - 1:                                         # feedback BETWEEN passes
                z = self.coconut_feedback(out.hidden_state)
                z = self.coconut_match_rms(z, prefix_embs, prefix_pad)

        if self.coconut_expert_visibility == "final_only":
            visible = thought_deltas[-1]
        else:
            visible = merge_deltas(thought_deltas)
        action_position_offset = valid_prefix_len + K            # NOT visible length
        return visible, action_position_offset

    def _coconut_denoise(self, prefix_pad, prefix_kv, visible,
                         action_position_offset, x_t, timestep):
        """One flow-matching denoise step with the thought delta spliced in.

        Mirrors ``VLAFlowMatching.denoise_step`` but (a) attends the thought-
        augmented cache, (b) widens the prefix pad mask by ``n_visible`` valid
        columns, and (c) takes an **explicit** ``action_position_offset`` instead
        of deriving positions from the (widened) cache length (Invariant 5).
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        model = self.base_policy.model
        expert_cache = append_delta_kv(prefix_kv, visible)        # LIVE prefix + thought
        n_visible = visible[next(iter(visible))]["key_states"].shape[1]
        B = prefix_pad.shape[0]
        ones = torch.ones(
            (B, n_visible), dtype=prefix_pad.dtype, device=prefix_pad.device
        )
        prefix_pad_w = torch.cat([prefix_pad, ones], dim=1)      # (B, prefix_len + n_visible)

        suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad.shape[1]
        prefix_len_w = prefix_pad_w.shape[1]
        prefix_pad_2d = prefix_pad_w[:, None, :].expand(B, suffix_len, prefix_len_w)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        position_ids = action_position_offset[:, None] + torch.cumsum(suffix_pad, dim=1) - 1

        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full_att_2d, position_ids=position_ids,
            past_key_values=expert_cache, inputs_embeds=[None, suffix_embs],
            use_cache=self.base_policy.config.use_cache, fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.base_policy.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def _coconut_forward(self, batch, noise, time):
        """Training forward for the Coconut path (flow-matching loss)."""
        import torch.nn.functional as F

        model = self.base_policy.model
        cfg = self.base_policy.config
        _META = ("episode_ids", "timesteps", "future_states", "future_valid")
        inner = {k: v for k, v in batch.items() if k not in _META}

        images, img_masks = self.base_policy.prepare_images(inner)
        state = self.base_policy.prepare_state(inner)
        lang_tokens = inner["observation.language.tokens"]
        lang_masks = inner["observation.language.attention_mask"]
        actions = self.base_policy.prepare_action(inner)
        actions_is_pad = inner.get("action_is_pad")

        if noise is None:
            noise = model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = model.sample_time(actions.shape[0], actions.device)
        te = time[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions

        self._current_episode_ids = list(batch["episode_ids"])
        self._current_timesteps = list(batch["timesteps"])
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            prefix_embs, prefix_pad, prefix_kv = self._build_prefix_kv(
                images, img_masks, lang_tokens, lang_masks, state
            )
        finally:
            self.feature_extractor.set_callback(None)
            self._current_episode_ids = None
            self._current_timesteps = None

        visible, offset = self.run_coconut_reasoning(prefix_embs, prefix_pad, prefix_kv)
        v_t = self._coconut_denoise(prefix_pad, prefix_kv, visible, offset, x_t, time)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        losses = losses[:, :, : cfg.action_feature.shape[0]]
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        losses = losses[:, :, : cfg.max_action_dim]
        loss = losses.mean()

        loss_dict = {"loss": loss.item(), "num_thoughts": float(self.num_thoughts)}
        scale = self.mem_bank.last_gate_scale()
        if scale is not None:
            loss_dict["gate_value_mean"] = scale.mean().item()
            loss_dict["gate_value_std"] = scale.std().item()
        if self.aux_loss_weight > 0 and "future_states" in batch:
            aux = self.mem_bank.aux_future_loss(
                batch["future_states"], batch["future_valid"]
            )
            if aux is not None:
                loss = loss + self.aux_loss_weight * aux
                loss_dict["aux_future_state_loss"] = aux.item()
        return loss, loss_dict

    @torch.no_grad()
    def _coconut_sample_actions(self, images, img_masks, lang_tokens, lang_masks,
                                state, noise=None):
        """Inference denoise loop with the thought delta spliced in (mirrors
        ``VLAFlowMatching.sample_actions``)."""
        model = self.base_policy.model
        cfg = self.base_policy.config
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = model.sample_noise(
                (bsize, cfg.chunk_size, cfg.max_action_dim), device
            )
        prefix_embs, prefix_pad, prefix_kv = self._build_prefix_kv(
            images, img_masks, lang_tokens, lang_masks, state
        )
        visible, offset = self.run_coconut_reasoning(prefix_embs, prefix_pad, prefix_kv)

        num_steps = cfg.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = self._coconut_denoise(prefix_pad, prefix_kv, visible, offset, x_t, t_tensor)
            x_t = x_t + dt * v_t
        return x_t

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
            if self._coconut_active():
                images, img_masks = self.base_policy.prepare_images(inner_batch)
                state = self.base_policy.prepare_state(inner_batch)
                lang_tokens = inner_batch["observation.language.tokens"]
                lang_masks = inner_batch["observation.language.attention_mask"]
                # The bank must advance once for this query; the callback fires
                # inside _build_prefix_kv, so clear it right after so the denoise
                # loop's expert forwards never re-fire it.
                actions = self._coconut_sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state, noise=noise
                )
            else:
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
        if self._coconut_active():
            raise NotImplementedError(
                "V10A Coconut inference runs through predict_action_chunk "
                "(chunk / ensemble eval). select_action's base queue path does "
                "not splice the thought delta — use --ensemble or chunk eval."
            )
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
        # V10A Coconut modules (only present when enabled with K>0).
        for name in ("coconut_seed", "coconut_feedback", "coconut_match_rms",
                     "coconut_adapters"):
            mod = getattr(self, name, None)
            if mod is not None:
                yield from mod.parameters()

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
