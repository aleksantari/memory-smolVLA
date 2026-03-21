"""Training configuration dataclass for MemorySmolVLATrainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Hyperparameters and settings for MemorySmolVLATrainer.

    Args:
        training_mode: One of ``"memory_only"``, ``"expert_finetune"``,
            ``"expert_scratch"``, or ``"expert_only_scratch"``. Must
            match the mode used to build the policy.
        total_steps: Total gradient update steps.
        memory_lr: Learning rate for memory module parameters (retrieval,
            gate, memory_proj).
        expert_lr: Learning rate for action expert parameters
            (``lm_expert`` + ``action_out_proj``). Only used in modes
            that train the expert.
        weight_decay: AdamW weight decay applied to all parameter groups.
        max_grad_norm: Maximum gradient norm for clipping. Set to 0 to
            disable clipping.
        warmup_steps: Number of linear warmup steps at the start of
            training.
        grad_accum_steps: Number of forward passes to accumulate before
            an optimizer step. Effective batch size is
            ``grad_accum_steps`` episodes (sequential mode) or
            ``grad_accum_steps * batch_size`` frames (random mode).
        checkpoint_every: Save a checkpoint every this many steps.
        log_every: Log metrics every this many steps.
        checkpoint_dir: Directory where checkpoints are saved.
        wandb_project: Weights & Biases project name. Set to ``None``
            to disable wandb logging.
        wandb_run_name: Optional run name override for wandb.
        device: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
        batch_size: Batch size for ``expert_only_scratch`` random
            loading mode (ignored in episode-sequential modes).
        num_workers: DataLoader worker count for random loading mode.
    """

    training_mode: str = "memory_only"
    total_steps: int = 100_000
    memory_lr: float = 1e-4
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
