"""Training configuration dataclass for MemorySmolVLATrainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Hyperparameters and settings for MemorySmolVLATrainer.

    Args:
        training_mode: One of ``"memory_only"``, ``"expert_finetune"``,
            ``"expert_scratch"``, or ``"expert_only_scratch"``.
        total_steps: Total gradient update steps.
        memory_lr: Learning rate for memory module parameters (retrieval,
            gate, compressor, write gate, working memory).
        memory_proj_lr: Learning rate for the zero-initialized memory
            projection. Defaults to ``memory_lr / 10`` if ``None``.
            A lower LR prevents the projection from growing too fast
            and overwhelming the gate.
        expert_lr: Learning rate for action expert parameters.
        weight_decay: AdamW weight decay applied to all parameter groups.
        max_grad_norm: Maximum gradient norm for clipping. 0 to disable.
        warmup_steps: Number of linear warmup steps.
        grad_accum_steps: Forward passes to accumulate before an
            optimizer step.
        checkpoint_every: Save a checkpoint every this many steps.
        log_every: Log metrics every this many steps.
        checkpoint_dir: Directory where checkpoints are saved.
        wandb_project: Weights & Biases project name. ``None`` to disable.
        wandb_run_name: Optional run name override for wandb.
        device: PyTorch device string.
        batch_size: Batch size for ``expert_only_scratch`` random mode.
        num_workers: DataLoader worker count for random loading mode.
    """

    training_mode: str = "memory_only"
    total_steps: int = 100_000
    memory_lr: float = 1e-4
    memory_proj_lr: float | None = None
    expert_lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1_000
    grad_accum_steps: int = 1
    checkpoint_every: int = 5_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    wandb_project: str | None = "memory-smolvla"
    wandb_run_name: str | None = None
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
