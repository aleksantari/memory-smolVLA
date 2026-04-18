"""Training configuration dataclass for MemorySmolVLATrainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Hyperparameters and settings for MemorySmolVLATrainer.

    Args:
        total_steps: Total gradient update steps.
        memory_lr: Learning rate for :class:`FullSeqMemBank` parameters.
        expert_lr: Learning rate for action-expert parameters.
        weight_decay: AdamW weight decay applied to all parameter groups.
        max_grad_norm: Maximum gradient norm for clipping. 0 to disable.
        warmup_steps: Number of linear warmup steps.
        grad_accum_steps: Forward passes to accumulate before an
            optimizer step.
        checkpoint_every: Save a checkpoint every this many steps.
        log_every: Log metrics every this many steps.
        gate_hist_every: Log a wandb histogram of gate values every N
            steps. Set to 0 to disable.
        checkpoint_dir: Directory where checkpoints are saved.
        wandb_project: Weights & Biases project name. ``None`` to disable.
        wandb_run_name: Optional run name override for wandb.
        device: PyTorch device string.
        num_groups: Number of episode groups per batch (``G``).
        group_size: Contiguous frames per group. ``batch_size = G * group_size``.
        num_workers: DataLoader worker processes used by
            :class:`GroupedEpisodeLoader`. ``0`` keeps batch construction
            on the main thread.
        prefetch_factor: Per-worker prefetch depth.
        pin_memory: Whether the DataLoader pins host memory for faster
            H→D transfers.
        use_amp: Wrap forward in ``torch.autocast(dtype=amp_dtype)`` to
            match baseline v2's mixed-precision regime. Backward runs in
            the same autocast context; no GradScaler (bfloat16 doesn't
            need one).
        amp_dtype: ``"bfloat16"`` or ``"float16"``. Default bfloat16.
        adam_betas: AdamW ``(beta1, beta2)`` tuple. Defaults to baseline
            v2's ``(0.9, 0.95)``.
        adam_eps: AdamW epsilon. Defaults to PyTorch's ``1e-8`` (matches
            baseline v2).
        seed: Random seed applied to Python ``random``, NumPy, and
            PyTorch (CPU + CUDA) before building the model and loader.
            Matches baseline v2's ``seed=1000``.
    """

    total_steps: int = 100_000
    memory_lr: float = 1e-4
    expert_lr: float = 1e-4
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    warmup_steps: int = 1_000
    grad_accum_steps: int = 1
    checkpoint_every: int = 5_000
    log_every: int = 100
    gate_hist_every: int = 500
    checkpoint_dir: str = "checkpoints"
    wandb_project: str | None = "memory-smolvla"
    wandb_run_name: str | None = None
    device: str = "cuda"
    num_groups: int = 8
    group_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    seed: int = 1000
