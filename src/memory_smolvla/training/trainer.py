"""Training loop for :class:`MemorySmolVLAPolicy`.

Single loop over a :class:`GroupedEpisodeLoader`; two optimizer groups
(action expert at ``expert_lr``, memory bank at ``memory_lr``); gate
value statistics logged to stdout and wandb. The grouped loader handles
episode-boundary isolation internally (via ``FullSeqMemBank.process_batch``
in group mode), so no sentinel plumbing is needed here.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from memory_smolvla.policy.memory_smolvla import MemorySmolVLAPolicy
from memory_smolvla.training.config import TrainerConfig

logger = logging.getLogger(__name__)


# LIBERO's dataset keys → SmolVLA policy keys. The dataset stores
# ``observation.images.{image,image2}`` (agentview + wrist); the SmolVLA
# policy expects ``camera1/camera2`` (``camera3`` is auto-padded).
_LIBERO_FEATURE_MAP: dict[str, str] = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


class MemorySmolVLATrainer:
    """Trains a :class:`MemorySmolVLAPolicy` on LIBERO.

    Args:
        policy: The policy to train.
        cfg: Training hyperparameters and settings.
        train_loader: A :class:`GroupedEpisodeLoader` yielding batches
            annotated with ``episode_ids`` and ``timesteps``.
        preprocessor: Optional SmolVLA preprocessor pipeline.
    """

    def __init__(
        self,
        policy: MemorySmolVLAPolicy,
        cfg: TrainerConfig,
        train_loader,
        preprocessor=None,
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.train_loader = train_loader
        self.preprocessor = preprocessor
        self.feature_map = _LIBERO_FEATURE_MAP
        self.device = torch.device(cfg.device)
        self._step = 0

        policy.to(self.device)

        if cfg.use_amp:
            self._amp_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[cfg.amp_dtype]
        else:
            self._amp_dtype = None

        param_groups = self._build_param_groups()
        self.optimizer = AdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
        )
        self.scheduler = self._build_scheduler()
        self._all_trainable: list[nn.Parameter] = [
            p for g in param_groups for p in g["params"]
        ]

        self._wandb = None
        if cfg.wandb_project:
            self._init_wandb()

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Trainer ready: total_steps=%d, device=%s, batch_size=%d",
            cfg.total_steps, cfg.device, cfg.num_groups * cfg.group_size,
        )

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run training until ``cfg.total_steps`` gradient updates."""
        self.policy.train()
        self.optimizer.zero_grad()
        accum_count = 0

        for batch in self.train_loader:
            if self._step >= self.cfg.total_steps:
                break

            batch = self._to_device(batch)
            if self._amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype):
                    loss, loss_dict = self.policy(batch)
            else:
                loss, loss_dict = self.policy(batch)
            (loss / self.cfg.grad_accum_steps).backward()
            accum_count += 1

            if accum_count >= self.cfg.grad_accum_steps:
                self._optimizer_step()
                accum_count = 0
                self._step += 1
                self._log(loss_dict)
                self._maybe_checkpoint()

        self._save_checkpoint("final")

    # ------------------------------------------------------------------

    def _build_param_groups(self) -> list[dict]:
        """Two groups: action expert at ``expert_lr``, memory at ``memory_lr``."""
        vwe = self.policy.base_policy.model.vlm_with_expert

        expert_params = list(vwe.lm_expert.parameters()) + list(
            self.policy.base_policy.model.action_out_proj.parameters()
        )
        memory_params = list(self.policy.mem_bank.parameters())

        groups: list[dict] = []
        if expert_params:
            groups.append({"params": expert_params, "lr": self.cfg.expert_lr})
        if memory_params:
            groups.append({"params": memory_params, "lr": self.cfg.memory_lr})

        if not groups:
            raise RuntimeError("No trainable parameters found for the optimizer.")

        total = sum(sum(p.numel() for p in g["params"]) for g in groups)
        logger.info(
            "Optimizer: %d trainable params across %d groups (LRs: %s)",
            total, len(groups), [g["lr"] for g in groups],
        )
        return groups

    def _build_scheduler(self) -> LambdaLR:
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
            nn.utils.clip_grad_norm_(self._all_trainable, self.cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------

    def _remap_features(self, batch: dict) -> dict:
        if not self.feature_map:
            return batch
        return {self.feature_map.get(k, k): v for k, v in batch.items()}

    def _to_device(self, batch: dict) -> dict:
        batch = self._remap_features(batch)

        meta = {
            k: batch.pop(k) for k in list(batch.keys())
            if k in ("episode_ids", "timesteps")
        }

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        else:
            batch = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

        batch.update(meta)
        return batch

    def _log(self, loss_dict: dict) -> None:
        if self._step % self.cfg.log_every != 0:
            return

        metrics = {
            **{f"train/{k}": v for k, v in loss_dict.items()},
            "train/step": self._step,
        }
        for i, pg in enumerate(self.optimizer.param_groups):
            metrics[f"train/lr_group_{i}"] = pg["lr"]

        logger.info(
            "step=%d loss=%.4f gate_mean=%.4f gate_std=%.4f",
            self._step,
            loss_dict.get("loss", float("nan")),
            loss_dict.get("gate_value_mean", float("nan")),
            loss_dict.get("gate_value_std", float("nan")),
        )

        if self._wandb is not None:
            self._wandb.log(metrics, step=self._step)
            if (
                self.cfg.gate_hist_every > 0
                and self._step % self.cfg.gate_hist_every == 0
            ):
                scale = self.policy.mem_bank.last_gate_scale()
                if scale is not None:
                    import wandb
                    self._wandb.log(
                        {"train/gate_value_hist": wandb.Histogram(scale.detach().float().cpu().numpy())},
                        step=self._step,
                    )

    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, path: str | Path) -> None:
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
