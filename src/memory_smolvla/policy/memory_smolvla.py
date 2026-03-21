"""Memory-augmented SmolVLA policy wrapper.

Wraps a frozen ``SmolVLAPolicy`` with a temporal memory system that
intercepts VLM intermediate representations at a configurable layer,
writes them to an episodic memory bank, retrieves relevant past
context via cross-attention, and fuses the result back into the
representation stream via a learned sigmoid gate.

The action expert (flow-matching head) receives the same
representation shape it was trained on, just memory-augmented —
upper VLM layers process the augmented prefix, so the expert's
cross-attention sees enriched context.

Only the memory modules (retrieval, gate, projection) have trainable
parameters. The base SmolVLA policy is fully frozen.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from memory_smolvla.memory.bank import ConsolidatingMemoryBank
from memory_smolvla.memory.gating import SigmoidGate
from memory_smolvla.memory.retrieval import CrossAttentionRetrieval
from memory_smolvla.policy.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


_VALID_TRAINING_MODES = frozenset(
    {"memory_only", "expert_scratch", "expert_finetune", "expert_only_scratch"}
)


class MemorySmolVLAPolicy(nn.Module):
    """Memory-augmented wrapper around a frozen SmolVLAPolicy.

    Intercepts VLM hidden states at layer ``injection_layer``, writes
    them to an episodic memory bank, retrieves via cross-attention
    with temporal positional encoding, and fuses via a sigmoid gate.
    The augmented prefix flows through upper VLM layers so the action
    expert sees enriched context.

    Args:
        base_policy: A ``SmolVLAPolicy`` instance. The VLM backbone is
            always frozen. Which other parameters are trainable depends
            on ``training_mode``.
        injection_layer: VLM layer index for memory injection.
            Default 8 (= N/2 for 16-layer SmolVLA).
        bank_max_size: Maximum memory bank entries before consolidation.
        retrieval_n_heads: Number of attention heads in cross-attention
            retrieval.
        gate_hidden_dim: Hidden dimension of the sigmoid gate MLP.
        training_mode: Controls which parameters are trainable.

            - ``"memory_only"`` — freeze everything in base_policy,
              only memory modules train. Requires pretrained SmolVLA
              checkpoint (16-layer VLM).
            - ``"expert_finetune"`` — fine-tune pretrained action expert
              + memory modules. Requires pretrained SmolVLA checkpoint.
            - ``"expert_scratch"`` — train action expert + memory from
              scratch (random expert init). Use when ``num_vlm_layers``
              differs from 16.
            - ``"expert_only_scratch"`` — train action expert only, no
              memory modules. Ablation baseline for variable-layer VLM.
    """

    def __init__(
        self,
        base_policy,
        injection_layer: int = 8,
        bank_max_size: int = 16,
        retrieval_n_heads: int = 4,
        gate_hidden_dim: int = 256,
        training_mode: str = "memory_only",
    ) -> None:
        super().__init__()

        if training_mode not in _VALID_TRAINING_MODES:
            raise ValueError(
                f"Invalid training_mode {training_mode!r}. "
                f"Must be one of {sorted(_VALID_TRAINING_MODES)}."
            )
        self._training_mode = training_mode

        # Register base policy as submodule so .to(device) propagates.
        # Start by freezing everything, then selectively unfreeze below.
        self.base_policy = base_policy
        for param in self.base_policy.parameters():
            param.requires_grad = False

        # Selectively unfreeze action expert + output projection when the
        # training mode requires it.
        if training_mode in ("expert_scratch", "expert_finetune", "expert_only_scratch"):
            vwe = self.base_policy.model.vlm_with_expert
            for param in vwe.lm_expert.parameters():
                param.requires_grad = True
            for param in self.base_policy.model.action_out_proj.parameters():
                param.requires_grad = True

        # Detect VLM hidden size from the base policy
        vlm_with_expert = self.base_policy.model.vlm_with_expert
        d_model = vlm_with_expert.config.text_config.hidden_size
        self.d_model = d_model
        self.injection_layer = injection_layer

        # --- Memory modules (trainable) ---
        self.memory_bank = ConsolidatingMemoryBank(max_size=bank_max_size)

        self.retrieval = CrossAttentionRetrieval(
            d_model=d_model,
            n_heads=retrieval_n_heads,
        )

        self.gate = SigmoidGate(
            d_model=d_model,
            hidden_dim=gate_hidden_dim,
        )

        # Zero-initialized projection so model starts as vanilla SmolVLA
        self.memory_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.memory_proj.weight)

        # Initialize gate for near-zero alpha at startup:
        # gate_mlp = Sequential(Linear, SiLU, Linear, Sigmoid)
        # gate_mlp[2] is the Linear(hidden, 1) before Sigmoid.
        # Zero weights + large negative bias → alpha ≈ sigmoid(-5) ≈ 0.007
        with torch.no_grad():
            final_linear = self.gate.gate_mlp[2]
            nn.init.zeros_(final_linear.weight)
            nn.init.constant_(final_linear.bias, -5.0)

        # Freeze memory modules in expert_only_scratch mode — they should
        # not receive gradients even if a caller uses policy.parameters().
        if training_mode == "expert_only_scratch":
            for param in self.retrieval.parameters():
                param.requires_grad = False
            for param in self.gate.parameters():
                param.requires_grad = False
            self.memory_proj.weight.requires_grad = False

        # --- Install feature extractor ---
        self.feature_extractor = FeatureExtractor(
            vlm_with_expert=vlm_with_expert,
            injection_layer=injection_layer,
        )

        # Episode state
        self._timestamp: int = 0
        self._last_gate_alpha: Tensor | None = None

        logger.info(
            "MemorySmolVLAPolicy initialized: d_model=%d, injection_layer=%d, "
            "bank_max_size=%d, retrieval_n_heads=%d, gate_hidden_dim=%d, "
            "training_mode=%s",
            d_model,
            injection_layer,
            bank_max_size,
            retrieval_n_heads,
            gate_hidden_dim,
            training_mode,
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
        """Training forward pass with memory augmentation.

        Installs the memory callback on the feature extractor, delegates
        to the base policy's forward (which triggers the callback at the
        injection layer), then removes the callback.

        Args:
            batch: Training batch from the data loader.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.

        Returns:
            Tuple of ``(loss, loss_dict)`` matching the
            ``SmolVLAPolicy.forward()`` interface.
        """
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            loss, loss_dict = self.base_policy.forward(
                batch, noise=noise, time=time
            )
        finally:
            self.feature_extractor.set_callback(None)

        # Append gate statistics to loss_dict
        if self._last_gate_alpha is not None:
            loss_dict["gate_alpha_mean"] = self._last_gate_alpha.mean().item()
            loss_dict["gate_alpha_std"] = self._last_gate_alpha.std().item()

        return loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Inference forward pass returning a full action chunk.

        The memory callback fires during the KV-cache-building phase
        (when the prefix is processed) and is a no-op during denoise
        steps (when only the suffix is processed with cached KV).
        """
        self.eval()
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            actions = self.base_policy.predict_action_chunk(
                batch, noise=noise, **kwargs
            )
        finally:
            self.feature_extractor.set_callback(None)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Streaming inference returning a single action."""
        self.eval()
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            action = self.base_policy.select_action(
                batch, noise=noise, **kwargs
            )
        finally:
            self.feature_extractor.set_callback(None)
        return action

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset action queues and memory for a new episode."""
        self.base_policy.reset()
        self.reset_memory()

    def reset_memory(self) -> None:
        """Clear the memory bank and reset the timestamp counter.

        Call at episode boundaries during both training and evaluation.
        """
        self.memory_bank.reset()
        self._timestamp = 0
        self._last_gate_alpha = None

    # ------------------------------------------------------------------
    # Memory callback — the core logic
    # ------------------------------------------------------------------

    def _memory_callback(
        self, prefix_hidden: Tensor, layer_idx: int
    ) -> Tensor:
        """Invoked by the feature extractor at the injection layer.

        Reads past context from the memory bank, fuses it with the
        current prefix via cross-attention and a sigmoid gate, then
        writes the current features to the bank for future timesteps.

        The write happens *after* retrieval so the current frame is
        never in its own memory context (prevents self-retrieval
        shortcuts).

        Args:
            prefix_hidden: VLM prefix hidden states after the injection
                layer, shape ``[B, L_prefix, d_model]``.
            layer_idx: Which layer just completed (for logging).

        Returns:
            Augmented prefix hidden states, same shape as input.
        """
        B, L, D = prefix_hidden.shape

        if B > 1:
            raise NotImplementedError(
                "Memory callback currently supports batch_size=1 only. "
                "Episode-sequential training requires B=1."
            )

        device = prefix_hidden.device
        current_time = self._timestamp

        # Step 1: Retrieve from past memories (if any exist)
        if len(self.memory_bank) > 0:
            memories, timestamps = self.memory_bank.read_all(device=device)
            # memories: [K, N_tokens, d_model], timestamps: [K]

            K, N_tok, D_mem = memories.shape

            # Compute time deltas
            time_deltas = (current_time - timestamps).float()  # [K]

            # Flatten: [K, N_tok, D] → [K*N_tok, D] → [B, K*N_tok, D]
            memory_flat = memories.reshape(K * N_tok, D_mem)
            memory_batch = memory_flat.unsqueeze(0).expand(B, -1, -1)

            # Expand time deltas: [K] → repeat each N_tok times → [K*N_tok] → [B, K*N_tok]
            time_deltas_expanded = time_deltas.repeat_interleave(N_tok)
            time_deltas_batch = time_deltas_expanded.unsqueeze(0).expand(B, -1)

            # Cross-attention retrieval
            retrieved = self.retrieval(
                current_tokens=prefix_hidden,
                memory_tokens=memory_batch,
                time_deltas=time_deltas_batch,
            )  # [B, L, D]

            # Apply zero-initialized projection
            retrieved = self.memory_proj(retrieved)  # [B, L, D]

            # Gated fusion
            fused, alpha = self.gate(
                current=prefix_hidden,
                retrieved=retrieved,
            )  # fused: [B, L, D], alpha: [B, L, 1]

            # Store gate values for logging
            self._last_gate_alpha = alpha.detach()
        else:
            # First timestep: no past memories, pass through unchanged
            fused = prefix_hidden

        # Step 2: Write current prefix to bank for future timesteps
        for b in range(B):
            self.memory_bank.write(
                tokens=prefix_hidden[b],  # [L_prefix, d_model]
                timestamp=current_time,
            )

        # Step 3: Advance timestamp
        self._timestamp += 1

        return fused

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters according to ``training_mode``.

        Memory modules are included in all modes except
        ``"expert_only_scratch"``. The action expert and output
        projection are included when the mode requires expert training.
        The VLM backbone is never included.
        """
        if self._training_mode != "expert_only_scratch":
            yield from self.retrieval.parameters()
            yield from self.gate.parameters()
            yield from self.memory_proj.parameters()

        if self._training_mode in ("expert_scratch", "expert_finetune", "expert_only_scratch"):
            vwe = self.base_policy.model.vlm_with_expert
            yield from vwe.lm_expert.parameters()
            yield from self.base_policy.model.action_out_proj.parameters()

    def get_gate_statistics(self) -> dict[str, float]:
        """Return gate activation statistics for logging.

        Returns an empty dict if no forward pass has been performed
        since the last reset.
        """
        if self._last_gate_alpha is None:
            return {}
        alpha = self._last_gate_alpha
        return {
            "gate_alpha_mean": alpha.mean().item(),
            "gate_alpha_std": alpha.std().item(),
            "gate_alpha_min": alpha.min().item(),
            "gate_alpha_max": alpha.max().item(),
        }

    @property
    def training_mode(self) -> str:
        """The active training mode string."""
        return self._training_mode

    @property
    def config(self):
        """Proxy to base policy config for compatibility."""
        return self.base_policy.config
