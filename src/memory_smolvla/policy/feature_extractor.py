"""Hook-based feature extraction from SmolLM2 intermediate layers.

Monkey-patches ``SmolVLMWithExpertModel.forward()`` to intercept the
layer-by-layer processing loop at a configurable injection point.
This is necessary because SmolVLA's forward loop manually decomposes
each transformer layer (calling ``input_layernorm``, ``q_proj``, etc.
individually) rather than calling ``layer.forward()`` — meaning
standard PyTorch ``register_forward_hook`` on decoder layer modules
will not fire.

The patched forward replicates the original loop exactly, with one
addition: after the injection layer completes, a user-provided
callback is invoked with the VLM prefix hidden states, and the
(optionally modified) prefix replaces the original for all
subsequent layers.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ForwardMode(enum.Enum):
    """Which computation the patched VLM forward is running (V10A / RevB).

    PREFIX          — the V9 memory-augmented prefix build. Memory callback ON,
                      bank write ON, timestep advance ON, latent adapters OFF.
    LATENT_THOUGHT  — a Coconut thought token pass. Memory callback OFF, no write,
                      no timestep advance, latent adapters ON. Forced onto the VLM
                      *self-attention* path at every layer (never cross-attn / the
                      expert), reading the merged (detached prefix + prior thoughts)
                      cache and returning this token's per-layer delta K/V.
    FLOW_ACTION     — the action-expert denoise forward. Everything OFF.
    """

    PREFIX = "prefix"
    LATENT_THOUGHT = "thought"
    FLOW_ACTION = "flow"


class FeatureExtractor:
    """Intercepts VLM intermediate representations via forward-loop patching.

    Replaces ``vlm_with_expert.forward()`` with a version that invokes
    a callback after the specified injection layer. The callback
    receives the VLM prefix hidden states and returns (optionally
    modified) hidden states that flow into all upper layers.

    When no callback is set, the patched forward produces bit-identical
    output to the original.

    Args:
        vlm_with_expert: The ``SmolVLMWithExpertModel`` instance to patch.
        injection_layer: Transformer layer index (0-based) after which
            the callback fires. For the default 16-layer SmolVLA:
            N/4=4, N/2=8, 3N/4=12, N-1=15.
        inject_before: If ``True``, the callback fires *before* the
            injection layer's attention instead of after its MLP.
            Use this for the last layer (N-1) so that the augmented
            VLM features enter the KV cache and are visible to the
            action expert during denoising cross-attention.
    """

    def __init__(
        self,
        vlm_with_expert,
        injection_layer: int,
        inject_before: bool = False,
    ) -> None:
        self._vwe = vlm_with_expert
        self._injection_layer = injection_layer
        self._inject_before = inject_before
        self._callback: Callable[[Tensor, int], Tensor] | None = None

        # Save original forward so we can restore it
        self._original_forward = vlm_with_expert.forward

        # Bind the patched forward to the model instance.
        # Note: functools.wraps cannot be used here because
        # _patched_forward is a bound method, and setattr for
        # __module__/__qualname__ fails on method objects.
        vlm_with_expert.forward = self._patched_forward

        logger.info(
            "FeatureExtractor installed on %s at injection_layer=%d (inject_before=%s)",
            type(vlm_with_expert).__name__,
            injection_layer,
            inject_before,
        )

    def set_callback(
        self,
        callback: Callable[[Tensor, int], Tensor] | None,
    ) -> None:
        """Set or clear the injection callback.

        Args:
            callback: Function ``(prefix_hidden [B, L, D], layer_idx) -> [B, L, D]``,
                or ``None`` to disable interception. When ``None``, the
                patched forward behaves identically to the original.
        """
        self._callback = callback

    def remove(self) -> None:
        """Restore the original forward method and clear state."""
        self._vwe.forward = self._original_forward
        self._callback = None
        logger.info("FeatureExtractor removed, original forward restored.")

    # ------------------------------------------------------------------
    # Patched forward — replicates SmolVLMWithExpertModel.forward()
    # from smolvlm_with_expert.py lines 404-499, with injection point.
    # ------------------------------------------------------------------

    def _patched_forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        """Forward pass with optional callback at the injection layer.

        This is an exact replication of
        ``SmolVLMWithExpertModel.forward()`` with one addition:
        after the injection layer's post-attention processing, the
        callback (if set) is invoked on ``inputs_embeds[0]`` (the
        VLM prefix stream).
        """
        vwe = self._vwe

        # --- Setup (original lines 413-425) ---
        models = [vwe.get_vlm_model().text_model, vwe.lm_expert]
        model_layers = vwe.get_model_layers(models)
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = vwe.num_vlm_layers
        head_dim = vwe.vlm.config.text_config.head_dim

        # --- Layer loop (original lines 426-489) ---
        for layer_idx in range(num_layers):
            # === PRE-LAYER INJECTION POINT ===
            if (
                self._inject_before
                and layer_idx == self._injection_layer
                and self._callback is not None
                and inputs_embeds[0] is not None
            ):
                inputs_embeds[0] = self._callback(
                    inputs_embeds[0], layer_idx
                )

            # Attention dispatch
            if (
                fill_kv_cache
                or "cross" not in vwe.attention_mode
                or (
                    vwe.self_attn_every_n_layers > 0
                    and layer_idx % vwe.self_attn_every_n_layers == 0
                )
            ):
                att_outputs, past_key_values = vwe.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = vwe.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            # Post-attention processing
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(
                            layer.self_attn.o_proj.weight.dtype
                        )
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

            # === POST-LAYER INJECTION POINT ===
            if (
                not self._inject_before
                and layer_idx == self._injection_layer
                and self._callback is not None
                and inputs_embeds[0] is not None
            ):
                inputs_embeds[0] = self._callback(
                    inputs_embeds[0], layer_idx
                )

        # --- Final norm (original lines 491-499) ---
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    # ------------------------------------------------------------------
    # V10A — LATENT_THOUGHT forced self-attention pass (RevB).
    # ------------------------------------------------------------------

    def vlm_thought_pass(
        self,
        token: Tensor,
        past_key_values: dict,
        position_ids: torch.LongTensor,
        key_padding_mask: Tensor,
        adapters: dict[int, Callable[[Tensor], Tensor]] | None = None,
    ):
        """Push one thought token through all VLM layers via self-attention only.

        This is the ``ForwardMode.LATENT_THOUGHT`` forced dispatch: it *never*
        enters ``forward_cross_attn_layer`` and *never* touches the action
        expert (which is why ``[z, None]`` through the stock forward would break
        — see ``compat_check.py`` finding F). It reads the merged detached cache
        (prefix + prior thoughts) and returns *this* token's per-layer delta K/V,
        exactly one token per layer, ready to append to the running cache.

        Args:
            token: ``(B, 1, D)`` current thought embedding.
            past_key_values: running cache ``{layer: {"key_states","value_states"}}``
                already containing prefix (+ any prior thoughts). Not mutated.
            position_ids: ``(B, 1)`` RoPE position of the thought (per example).
            key_padding_mask: ``(B, 1, S+1)`` bool — which of the ``S`` cached keys
                plus this token the thought may attend (True = attend).
            adapters: optional ``{layer_idx: LatentModeAdapter}`` applied to the
                thought row *before* that layer's ``input_layernorm``/QKV
                (RevB pre-QKV placement, layers 12-15). Zero-init ⇒ identity.

        Returns:
            ``ThoughtPassOutput`` (imported lazily to avoid a cycle) with the
            last-layer hidden state ``(B, 1, D)`` and ``delta_kv``.
        """
        from memory_smolvla.memory.reasoning import ThoughtPassOutput
        from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

        vwe = self._vwe
        text_model = vwe.get_vlm_model().text_model
        layers = vwe.get_model_layers([text_model, vwe.lm_expert])[0]
        num_layers = vwe.num_vlm_layers
        head_dim = vwe.vlm.config.text_config.head_dim
        attention_interface = vwe.get_attention_interface()
        adapters = adapters or {}

        z = token
        batch_size = z.shape[0]
        delta_kv: dict[int, dict[str, Tensor]] = {}

        for layer_idx in range(num_layers):
            layer = layers[layer_idx]

            # RevB: latent adapter on the thought row, pre-QKV (identity at init).
            z_in = adapters[layer_idx](z) if layer_idx in adapters else z

            hidden = layer.input_layernorm(z_in)
            hidden = hidden.to(dtype=layer.self_attn.q_proj.weight.dtype)
            hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hidden).view(hidden_shape)
            k = layer.self_attn.k_proj(hidden).view(hidden_shape)
            v = layer.self_attn.v_proj(hidden).view(hidden_shape)

            q = apply_rope(q, position_ids)
            k = apply_rope(k, position_ids)

            # This token's own K/V = the delta to append to the cache.
            delta_kv[layer_idx] = {"key_states": k, "value_states": v}

            cached = past_key_values[layer_idx]
            k_full = torch.cat([cached["key_states"], k], dim=1)
            v_full = torch.cat([cached["value_states"], v], dim=1)

            att_output = attention_interface(
                key_padding_mask, batch_size, head_dim, q, k_full, v_full
            )
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

            out = layer.self_attn.o_proj(att_output)
            out = out + z_in
            residual = out.clone()
            out = layer.post_attention_layernorm(out)
            out = layer.mlp(out)
            z = out + residual

        return ThoughtPassOutput(hidden_state=z, delta_kv=delta_kv)
