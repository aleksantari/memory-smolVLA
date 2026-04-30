"""Memory-augmented SmolVLA policy wrapper.

Wraps a frozen ``SmolVLAPolicy`` with a temporal memory system that
intercepts VLM intermediate representations at a configurable layer,
writes them to an episodic memory bank, retrieves relevant past
context via cross-attention, and fuses the result back into the
representation stream via a learned sigmoid gate.

Supports two memory backends:
- **Episodic memory** (default): Explicit memory bank with cross-attention
  retrieval and optional learned compressor, write gate, and multi-scale
  storage.
- **Working memory**: GRU-based recurrent state — simpler, constant cost,
  no train/inference temporal mismatch.

Only the memory modules (retrieval, gate, projection, compressor, write
gate, working memory) have trainable parameters. The base SmolVLA policy
is fully frozen.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from memory_smolvla.memory.bank import MemoryBank
from memory_smolvla.memory.compressor import MemoryCompressor
from memory_smolvla.memory.gating import ResidualGate, SigmoidGate
from memory_smolvla.memory.multi_scale_bank import MultiScaleMemoryBank
from memory_smolvla.memory.retrieval import CrossAttentionRetrieval
from memory_smolvla.memory.working_memory import WorkingMemory
from memory_smolvla.memory.write_gate import WriteGate
from memory_smolvla.policy.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


_VALID_TRAINING_MODES = frozenset(
    {"memory_only", "expert_scratch", "expert_finetune", "expert_only_scratch"}
)

_VALID_MEMORY_BACKENDS = frozenset({"episodic", "working"})

_VALID_COMPRESSION_MODES = frozenset({"none", "mean_pool"})


class MemorySmolVLAPolicy(nn.Module):
    """Memory-augmented wrapper around a frozen SmolVLAPolicy.

    Intercepts VLM hidden states at layer ``injection_layer``, writes
    them to an episodic memory bank (or updates a working memory GRU),
    retrieves via cross-attention with temporal positional encoding (or
    reads the GRU state), and fuses via a sigmoid gate.

    Args:
        base_policy: A ``SmolVLAPolicy`` instance. The VLM backbone is
            always frozen.
        injection_layer: VLM layer index for memory injection.
        bank_max_size: Maximum memory bank entries before eviction.
        retrieval_n_heads: Number of attention heads in cross-attention.
        gate_hidden_dim: Hidden dimension of the sigmoid gate MLP.
        training_mode: Controls which parameters are trainable.
        inject_before: If ``True``, memory fires *before* the injection
            layer's attention rather than after its MLP.
        memory_backend: ``"episodic"`` (bank + cross-attention) or
            ``"working"`` (GRU recurrent state).
        use_compressor: If ``True``, compress prefix before writing to
            the bank. Reduces storage and retrieval cost.
        compressor_n_slots: Number of compressed slots per memory entry.
        compression_mode: Cheap (zero-param) compression applied to the
            prefix before writing to the bank. ``"none"`` stores the full
            prefix (legacy). ``"mean_pool"`` stores a single token that
            is the mean of the prefix along the token axis. Note: this
            is independent of ``use_compressor`` (which adds a learned
            Perceiver compressor); compression_mode runs *after* the
            learned compressor if both are enabled.
        use_write_gate: If ``True``, learn when to write to the bank.
        use_multi_scale: If ``True``, use multi-scale memory bank instead
            of a single bank.
        eviction: Bank eviction strategy — ``"fifo"`` or ``"consolidate"``.
        alpha_target: Target alpha for gate regularization.
        alpha_reg_weight: Weight for gate alpha regularization loss.
        step_increment: Timestamp increment per callback invocation.
            Set to 1 for frame-by-frame training. Set to ``chunk_size``
            (e.g. 50) during inference so temporal PE sees the same
            scale as during training.
    """

    def __init__(
        self,
        base_policy,
        injection_layer: int = 8,
        bank_max_size: int = 16,
        retrieval_n_heads: int = 4,
        gate_hidden_dim: int = 256,
        training_mode: str = "memory_only",
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
        gate_type: str = "sigmoid",
        compression_mode: str = "none",
    ) -> None:
        super().__init__()

        if training_mode not in _VALID_TRAINING_MODES:
            raise ValueError(
                f"Invalid training_mode {training_mode!r}. "
                f"Must be one of {sorted(_VALID_TRAINING_MODES)}."
            )
        if memory_backend not in _VALID_MEMORY_BACKENDS:
            raise ValueError(
                f"Invalid memory_backend {memory_backend!r}. "
                f"Must be one of {sorted(_VALID_MEMORY_BACKENDS)}."
            )
        if compression_mode not in _VALID_COMPRESSION_MODES:
            raise ValueError(
                f"Invalid compression_mode {compression_mode!r}. "
                f"Must be one of {sorted(_VALID_COMPRESSION_MODES)}."
            )
        self._compression_mode = compression_mode

        self._training_mode = training_mode
        self._memory_backend = memory_backend

        # Register base policy as submodule so .to(device) propagates.
        self.base_policy = base_policy
        for param in self.base_policy.parameters():
            param.requires_grad = False

        # Selectively unfreeze action expert when needed.
        if training_mode in ("expert_scratch", "expert_finetune", "expert_only_scratch"):
            vwe = self.base_policy.model.vlm_with_expert
            for param in vwe.lm_expert.parameters():
                param.requires_grad = True
            for param in self.base_policy.model.action_out_proj.parameters():
                param.requires_grad = True

        # Detect VLM hidden size
        vlm_with_expert = self.base_policy.model.vlm_with_expert
        d_model = vlm_with_expert.config.text_config.hidden_size
        self.d_model = d_model
        self.injection_layer = injection_layer

        # --- Memory modules ---
        if memory_backend == "episodic":
            # Memory bank (single or multi-scale)
            self._use_multi_scale = use_multi_scale
            if use_multi_scale:
                self.memory_bank = MultiScaleMemoryBank(
                    short_size=max(1, bank_max_size // 4),
                    medium_size=max(1, bank_max_size // 2),
                    long_size=bank_max_size,
                    eviction=eviction,
                )
            else:
                self.memory_bank = MemoryBank(
                    max_size=bank_max_size, eviction=eviction
                )

            # Optional compressor (learned write head)
            self._use_compressor = use_compressor
            if use_compressor:
                self.compressor = MemoryCompressor(
                    d_model=d_model,
                    n_slots=compressor_n_slots,
                    n_heads=retrieval_n_heads,
                )
            else:
                self.compressor = None

            # Optional write gate (selective writing)
            self._use_write_gate = use_write_gate
            if use_write_gate:
                self.write_gate = WriteGate(d_model=d_model)
            else:
                self.write_gate = None

            # Cross-attention retrieval
            self.retrieval = CrossAttentionRetrieval(
                d_model=d_model,
                n_heads=retrieval_n_heads,
            )

            # Working memory placeholder (not used in episodic mode)
            self.working_mem = None

        else:  # working memory
            self._use_multi_scale = False
            self._use_compressor = False
            self._use_write_gate = False
            self.memory_bank = None
            self.compressor = None
            self.write_gate = None
            self.retrieval = None

            self.working_mem = WorkingMemory(
                d_model=d_model,
                state_dim=d_model,
            )

        # --- Shared modules (both backends) ---
        self._gate_type = gate_type
        if gate_type == "residual":
            self.gate = ResidualGate()
        else:
            self.gate = SigmoidGate(
                d_model=d_model,
                hidden_dim=gate_hidden_dim,
                alpha_target=alpha_target,
                alpha_reg_weight=alpha_reg_weight,
            )

        # Zero-initialized projection so model starts as vanilla SmolVLA
        self.memory_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.memory_proj.weight)

        # Initialize sigmoid gate bias (only for sigmoid gate)
        if gate_type == "sigmoid":
            with torch.no_grad():
                final_linear = self.gate.gate_mlp[2]
                nn.init.zeros_(final_linear.weight)
                nn.init.constant_(final_linear.bias, gate_init_bias)

        # Freeze memory modules in expert_only_scratch mode
        if training_mode == "expert_only_scratch":
            for mod in self._memory_modules():
                for param in mod.parameters():
                    param.requires_grad = False
            self.memory_proj.weight.requires_grad = False

        # --- Install feature extractor ---
        self.feature_extractor = FeatureExtractor(
            vlm_with_expert=vlm_with_expert,
            injection_layer=injection_layer,
            inject_before=inject_before,
        )

        # Episode state
        self._timestamp: int = 0
        self._step_increment = step_increment
        self._last_gate_alpha: Tensor | None = None
        self._last_write_prob: Tensor | None = None

        # Cache for trainable parameter list (avoid re-collecting each step)
        self._trainable_params_cache: list[nn.Parameter] | None = None

        logger.info(
            "MemorySmolVLAPolicy initialized: d_model=%d, injection_layer=%d, "
            "bank_max_size=%d, memory_backend=%s, use_compressor=%s, "
            "use_write_gate=%s, use_multi_scale=%s, eviction=%s, "
            "alpha_reg_weight=%.4f, gate_type=%s, gate_init_bias=%.1f, "
            "compression_mode=%s, training_mode=%s",
            d_model, injection_layer, bank_max_size, memory_backend,
            use_compressor, use_write_gate, use_multi_scale, eviction,
            alpha_reg_weight, gate_type, gate_init_bias,
            compression_mode, training_mode,
        )

    def _memory_modules(self) -> list[nn.Module]:
        """Return all memory-related nn.Modules (for freezing/unfreezing)."""
        modules = [self.gate]
        if self.retrieval is not None:
            modules.append(self.retrieval)
        if self.compressor is not None:
            modules.append(self.compressor)
        if self.write_gate is not None:
            modules.append(self.write_gate)
        if self.working_mem is not None:
            modules.append(self.working_mem)
        return modules

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
        to the base policy's forward, then removes the callback.

        Returns:
            Tuple of ``(loss, loss_dict)``.
        """
        self.feature_extractor.set_callback(self._memory_callback)
        try:
            loss, loss_dict = self.base_policy.forward(
                batch, noise=noise, time=time
            )
        finally:
            self.feature_extractor.set_callback(None)

        # Gate regularization
        if self._last_gate_alpha is not None:
            reg_loss = self.gate.regularization_loss(self._last_gate_alpha)
            if reg_loss.item() > 0:
                loss = loss + reg_loss
                loss_dict["gate_reg_loss"] = reg_loss.item()

        # Append gate statistics to loss_dict
        if self._last_gate_alpha is not None:
            loss_dict["gate_alpha_mean"] = self._last_gate_alpha.mean().item()
            loss_dict["gate_alpha_std"] = self._last_gate_alpha.std().item()

        if self._last_write_prob is not None:
            loss_dict["write_prob_mean"] = self._last_write_prob.mean().item()

        return loss, loss_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Inference forward pass returning a full action chunk."""
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
        """Clear the memory bank/state and reset the timestamp counter."""
        if self.memory_bank is not None:
            self.memory_bank.reset()
        if self.working_mem is not None:
            self.working_mem.reset()
        self._timestamp = 0
        self._last_gate_alpha = None
        self._last_write_prob = None

    @property
    def step_increment(self) -> int:
        """Timestamp increment per callback invocation."""
        return self._step_increment

    @step_increment.setter
    def step_increment(self, value: int) -> None:
        """Set timestamp increment (use chunk_size at inference time)."""
        self._step_increment = value

    # ------------------------------------------------------------------
    # Memory callback — the core logic
    # ------------------------------------------------------------------

    def _memory_callback(
        self, prefix_hidden: Tensor, layer_idx: int
    ) -> Tensor:
        """Invoked by the feature extractor at the injection layer.

        Dispatches to the appropriate memory backend.
        """
        B, L, D = prefix_hidden.shape

        if B > 1:
            raise NotImplementedError(
                "Memory callback currently supports batch_size=1 only. "
                "Episode-sequential training requires B=1."
            )

        if self._memory_backend == "episodic":
            return self._episodic_callback(prefix_hidden, layer_idx)
        else:
            return self._working_memory_callback(prefix_hidden, layer_idx)

    def _episodic_callback(
        self, prefix_hidden: Tensor, layer_idx: int
    ) -> Tensor:
        """Episodic memory: bank read → cross-attention → gate → bank write."""
        B, L, D = prefix_hidden.shape
        device = prefix_hidden.device
        orig_dtype = prefix_hidden.dtype
        compute_dtype = self.memory_proj.weight.dtype
        current_time = self._timestamp

        prefix_compute = prefix_hidden.to(compute_dtype)

        # Step 1: Retrieve from past memories (if any exist)
        if len(self.memory_bank) > 0:
            memories, timestamps = self.memory_bank.read_all(device=device)
            K, N_tok, D_mem = memories.shape

            time_deltas = (current_time - timestamps).float()

            memory_flat = memories.reshape(K * N_tok, D_mem).to(compute_dtype)
            memory_batch = memory_flat.unsqueeze(0).expand(B, -1, -1)

            time_deltas_expanded = time_deltas.repeat_interleave(N_tok)
            time_deltas_batch = time_deltas_expanded.unsqueeze(0).expand(B, -1)

            retrieved = self.retrieval(
                current_tokens=prefix_compute,
                memory_tokens=memory_batch,
                time_deltas=time_deltas_batch,
            )

            retrieved = self.memory_proj(retrieved)

            fused, alpha = self.gate(
                current=prefix_compute,
                retrieved=retrieved,
            )

            self._last_gate_alpha = alpha.detach()
        else:
            # First timestep: pass zeros through gate for graph connectivity
            retrieved = self.memory_proj(torch.zeros_like(prefix_compute))
            fused, alpha = self.gate(
                current=prefix_compute,
                retrieved=retrieved,
            )
            self._last_gate_alpha = alpha.detach()

        fused = fused.to(orig_dtype)

        # Step 2: Write current prefix to bank
        for b in range(B):
            tokens_to_store = prefix_hidden[b]  # [L_prefix, d_model]

            # Optionally compress before storing
            if self._use_compressor:
                tokens_to_store = self.compressor(
                    tokens_to_store.unsqueeze(0).to(compute_dtype)
                ).squeeze(0).to(orig_dtype)

            # Cheap zero-param compression (mean-pool the whole prefix
            # to one token). Applied after the learned compressor so
            # both can be combined; the typical use is to set one or
            # the other.
            if self._compression_mode == "mean_pool":
                tokens_to_store = tokens_to_store.mean(dim=0, keepdim=True)

            # Optionally gate the write
            if self._use_write_gate:
                recent_memory = None
                if len(self.memory_bank) > 0:
                    memories, _ = self.memory_bank.read_all(device=device)
                    recent_memory = memories.to(compute_dtype).mean(dim=(0, 1)).unsqueeze(0)

                should_write, write_prob = self.write_gate.should_write(
                    tokens_to_store.unsqueeze(0).to(compute_dtype),
                    recent_memory,
                )
                self._last_write_prob = write_prob.detach()

                if not should_write and not self.training:
                    # Skip write during inference if gate says no.
                    # During training, always write so gradients flow
                    # through the write gate.
                    pass
                else:
                    self.memory_bank.write(
                        tokens=tokens_to_store,
                        timestamp=current_time,
                    )
            else:
                self.memory_bank.write(
                    tokens=tokens_to_store,
                    timestamp=current_time,
                )

        # Step 3: Advance timestamp
        self._timestamp += self._step_increment

        return fused

    def _working_memory_callback(
        self, prefix_hidden: Tensor, layer_idx: int
    ) -> Tensor:
        """Working memory: GRU update → project → gate."""
        B, L, D = prefix_hidden.shape
        orig_dtype = prefix_hidden.dtype
        compute_dtype = self.memory_proj.weight.dtype

        prefix_compute = prefix_hidden.to(compute_dtype)

        # GRU produces memory signal
        memory_signal = self.working_mem(prefix_compute)

        # Project through zero-init linear
        retrieved = self.memory_proj(memory_signal)

        # Gated fusion
        fused, alpha = self.gate(
            current=prefix_compute,
            retrieved=retrieved,
        )

        self._last_gate_alpha = alpha.detach()

        # Detach GRU state to prevent full BPTT across entire episode.
        # Gradients flow through the current step only (same as episodic).
        self.working_mem.detach_state()

        # Advance timestamp (for logging consistency)
        self._timestamp += self._step_increment

        return fused.to(orig_dtype)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield trainable parameters according to ``training_mode``.

        Uses a cached parameter list to avoid re-collecting each step.
        """
        if self._trainable_params_cache is not None:
            yield from self._trainable_params_cache
            return

        params = []

        if self._training_mode != "expert_only_scratch":
            # Shared modules
            params.extend(self.gate.parameters())
            params.append(self.memory_proj.weight)

            # Backend-specific modules
            if self.retrieval is not None:
                params.extend(self.retrieval.parameters())
            if self.compressor is not None:
                params.extend(self.compressor.parameters())
            if self.write_gate is not None:
                params.extend(self.write_gate.parameters())
            if self.working_mem is not None:
                params.extend(self.working_mem.parameters())

        if self._training_mode in ("expert_scratch", "expert_finetune", "expert_only_scratch"):
            vwe = self.base_policy.model.vlm_with_expert
            params.extend(vwe.lm_expert.parameters())
            params.extend(self.base_policy.model.action_out_proj.parameters())

        self._trainable_params_cache = params
        yield from params

    def get_gate_statistics(self) -> dict[str, float]:
        """Return gate activation statistics for logging."""
        if self._last_gate_alpha is None:
            return {}
        alpha = self._last_gate_alpha
        stats = {
            "gate_alpha_mean": alpha.mean().item(),
            "gate_alpha_std": alpha.std().item(),
            "gate_alpha_min": alpha.min().item(),
            "gate_alpha_max": alpha.max().item(),
        }
        if self._last_write_prob is not None:
            stats["write_prob_mean"] = self._last_write_prob.mean().item()
        return stats

    @property
    def training_mode(self) -> str:
        """The active training mode string."""
        return self._training_mode

    @property
    def memory_backend(self) -> str:
        """The active memory backend string."""
        return self._memory_backend

    @property
    def config(self):
        """Proxy to base policy config for compatibility."""
        return self.base_policy.config
