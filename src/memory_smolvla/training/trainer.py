"""Training loop for MemorySmolVLAPolicy across all training modes.

Supports two loop strategies:
- **Episode-sequential** (all memory-involved modes): frames arrive in
  temporal order; the memory bank accumulates within each episode; an
  ``EpisodeBoundary`` sentinel triggers a memory reset between episodes.
- **Standard batch** (``expert_only_scratch``): random-sampled batches,
  no memory state, conventional flow-matching training loop.

Gradient accumulation, cosine LR schedule with linear warmup,
gradient clipping, and wandb/checkpoint logging are all supported.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from memory_smolvla.data.episode_loader import EpisodeBoundary
from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy
from memory_smolvla.training.config import TrainerConfig

logger = logging.getLogger(__name__)

# Training modes that require episode-sequential data loading
_SEQUENTIAL_MODES = frozenset(
    {"memory_only", "expert_scratch", "expert_finetune"}
)


class MemorySmolVLATrainer:
    """Trains a :class:`MemorySmolVLAPolicy` according to ``cfg.training_mode``.

    Args:
        policy: The policy to train.
        cfg: Training hyperparameters and settings.
        train_loader: An :class:`EpisodeSequentialLoader` for sequential
            modes, or a ``DataLoader`` for ``expert_only_scratch`` mode.
    """

    def __init__(
        self,
        policy: MemorySmolVLAPolicy,
        cfg: TrainerConfig,
        train_loader,
        preprocessor=None,
        feature_map: dict[str, str] | None = None,
    ) -> None:
        if policy.training_mode != cfg.training_mode:
            raise ValueError(
                f"Policy training_mode={policy.training_mode!r} does not match "
                f"TrainerConfig training_mode={cfg.training_mode!r}."
            )

        self.policy = policy
        self.cfg = cfg
        self.train_loader = train_loader
        self.preprocessor = preprocessor
        self.feature_map = feature_map or {}
        self.device = torch.device(cfg.device)
        self._step = 0

        policy.to(self.device)

        # --- Optimizer with separate LR groups ---
        param_groups = self._build_param_groups()
        self.optimizer = AdamW(
            param_groups, weight_decay=cfg.weight_decay
        )
        self.scheduler = self._build_scheduler()

        # Cache trainable params as a flat list for grad clipping
        self._all_trainable: list[nn.Parameter] = [
            p for g in param_groups for p in g["params"]
        ]

        # --- Logging ---
        self._wandb = None
        if cfg.wandb_project:
            self._init_wandb()

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Trainer ready: mode=%s, total_steps=%d, device=%s",
            cfg.training_mode,
            cfg.total_steps,
            cfg.device,
        )

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run training until ``cfg.total_steps`` gradient updates."""
        self.policy.train()
        if self.cfg.training_mode in _SEQUENTIAL_MODES:
            self._train_sequential()
        else:
            self._train_batch()

    # ------------------------------------------------------------------
    # Episode-sequential loop
    # ------------------------------------------------------------------

    def _train_sequential(self) -> None:
        self.optimizer.zero_grad()
        accum_count = 0

        for item in self.train_loader:
            if self._step >= self.cfg.total_steps:
                break

            if isinstance(item, EpisodeBoundary):
                if accum_count > 0:
                    self._optimizer_step()
                    self._step += 1
                    accum_count = 0
                self.policy.reset_memory()
                continue

            batch = self._to_device(item)
            loss, loss_dict = self.policy(batch)
            (loss / self.cfg.grad_accum_steps).backward()
            accum_count += 1

            if accum_count >= self.cfg.grad_accum_steps:
                self._optimizer_step()
                accum_count = 0
                self._step += 1
                self._log(loss_dict)
                self._maybe_checkpoint()

                if self._step >= self.cfg.total_steps:
                    break

        # Final checkpoint
        self._save_checkpoint("final")

    # ------------------------------------------------------------------
    # Standard batch loop (expert_only_scratch)
    # ------------------------------------------------------------------

    def _train_batch(self) -> None:
        self.optimizer.zero_grad()
        accum_count = 0

        while self._step < self.cfg.total_steps:
            for batch in self.train_loader:
                if self._step >= self.cfg.total_steps:
                    break

                batch = self._to_device(batch)
                loss, loss_dict = self.policy(batch)
                (loss / self.cfg.grad_accum_steps).backward()
                accum_count += 1

                if accum_count >= self.cfg.grad_accum_steps:
                    self._optimizer_step()
                    accum_count = 0
                    self._step += 1
                    self._log(loss_dict)
                    self._maybe_checkpoint()

                    if self._step >= self.cfg.total_steps:
                        break

        self._save_checkpoint("final")

    # ------------------------------------------------------------------
    # Optimizer / scheduler helpers
    # ------------------------------------------------------------------

    def _build_param_groups(self) -> list[dict]:
        """Build AdamW parameter groups with per-component LRs.

        Three possible groups:
        1. Memory modules (retrieval, gate, compressor, write gate, working memory)
        2. Memory projection (separate, lower LR to prevent saturation)
        3. Expert parameters (action expert + output projection)
        """
        groups = []
        mode = self.cfg.training_mode
        vwe = self.policy.base_policy.model.vlm_with_expert

        if mode != "expert_only_scratch":
            # Group 1: Memory modules (excluding memory_proj)
            memory_params = []
            if self.policy.retrieval is not None:
                memory_params.extend(self.policy.retrieval.parameters())
            memory_params.extend(self.policy.gate.parameters())
            if self.policy.compressor is not None:
                memory_params.extend(self.policy.compressor.parameters())
            if self.policy.write_gate is not None:
                memory_params.extend(self.policy.write_gate.parameters())
            if self.policy.working_mem is not None:
                memory_params.extend(self.policy.working_mem.parameters())

            if memory_params:
                groups.append({
                    "params": memory_params,
                    "lr": self.cfg.memory_lr,
                })

            # Group 2: Memory projection (lower LR)
            proj_lr = self.cfg.memory_proj_lr
            if proj_lr is None:
                proj_lr = self.cfg.memory_lr / 10.0
            groups.append({
                "params": [self.policy.memory_proj.weight],
                "lr": proj_lr,
            })

        if mode in ("expert_scratch", "expert_finetune", "expert_only_scratch"):
            # Group 3: Expert parameters
            expert_params = list(vwe.lm_expert.parameters()) + \
                            list(self.policy.base_policy.model.action_out_proj.parameters())
            groups.append({
                "params": expert_params,
                "lr": self.cfg.expert_lr,
            })

        if not groups:
            raise RuntimeError("No trainable parameters found for the optimizer.")

        total = sum(sum(p.numel() for p in g["params"]) for g in groups)
        logger.info(
            "Optimizer: %d trainable parameters, %d groups (LRs: %s)",
            total, len(groups),
            [g["lr"] for g in groups],
        )
        return groups

    def _build_scheduler(self) -> LambdaLR:
        """Cosine decay with linear warmup."""
        warmup = self.cfg.warmup_steps
        total = self.cfg.total_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _optimizer_step(self) -> None:
        if self.cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self._all_trainable,
                self.cfg.max_grad_norm,
            )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _remap_features(self, batch: dict) -> dict:
        """Remap dataset feature keys to policy-expected keys."""
        if not self.feature_map:
            return batch
        remapped = {}
        for k, v in batch.items():
            remapped[self.feature_map.get(k, k)] = v
        return remapped

    def _to_device(self, batch: dict) -> dict:
        batch = self._remap_features(batch)
        if self.preprocessor is not None:
            return self.preprocessor(batch)
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _log(self, loss_dict: dict) -> None:
        if self._step % self.cfg.log_every != 0:
            return

        gate_stats = self.policy.get_gate_statistics()
        metrics = {
            **{f"train/{k}": v for k, v in loss_dict.items()},
            **{f"train/{k}": v for k, v in gate_stats.items()},
            "train/step": self._step,
            **{f"train/lr_group_{i}": pg["lr"] * self.scheduler.get_last_lr()[0] / max(self.scheduler.get_last_lr()[0], 1e-12)
               for i, pg in enumerate(self.optimizer.param_groups)},
        }

        # Add memory bank size if applicable
        if self.policy.memory_bank is not None:
            metrics["train/memory_bank_size"] = len(self.policy.memory_bank)

        logger.info(
            "step=%d loss=%.4f gate_alpha=%.4f",
            self._step,
            loss_dict.get("loss", float("nan")),
            gate_stats.get("gate_alpha_mean", 0.0),
        )

        if self._wandb is not None:
            self._wandb.log(metrics, step=self._step)

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Restore full training state from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._step = ckpt["step"]
        logger.info("Resumed from checkpoint: %s (step %d)", path, self._step)

    def _maybe_checkpoint(self) -> None:
        if self._step > 0 and self._step % self.cfg.checkpoint_every == 0:
            self._save_checkpoint(f"step_{self._step:07d}")

    def _save_checkpoint(self, tag: str) -> None:
        path = Path(self.cfg.checkpoint_dir) / f"{tag}.pt"
        torch.save(
            {
                "step": self._step,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "training_mode": self.cfg.training_mode,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def _init_wandb(self) -> None:
        try:
            import wandb
            self._wandb = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=vars(self.cfg),
                resume="allow",
            )
        except ImportError:
            logger.warning("wandb not installed; logging to stdout only.")
