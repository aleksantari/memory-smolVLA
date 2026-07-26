"""Training loop for :class:`MemorySmolVLAPolicy`.

Single loop over a :class:`GroupedEpisodeLoader`; two optimizer groups
(action expert at ``expert_lr``, memory bank at ``memory_lr``); gate
value statistics logged to stdout and wandb. The grouped loader handles
episode-boundary isolation internally (via ``FullSeqMemBank.process_batch``
in group mode), so no sentinel plumbing is needed here.
"""

from __future__ import annotations

import dataclasses
import json
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
        self._tokenizer_padding_verdict = self._verify_tokenizer_padding()
        self._dump_resolved_policy_config()
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
        # V10: the separate reasoning bank trains at the memory LR too.
        if getattr(self.policy, "reasoning_bank", None) is not None:
            memory_params += list(self.policy.reasoning_bank.parameters())
        # V10A: the Coconut thought modules live on the policy (not the bank);
        # group them with the memory LR so they actually train. Absent ⇒ no-op.
        for name in ("coconut_seed", "coconut_feedback", "coconut_match_rms",
                     "coconut_adapters"):
            mod = getattr(self.policy, name, None)
            if mod is not None:
                memory_params += list(mod.parameters())

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
            if k in ("episode_ids", "timesteps", "global_idxs", "future_states", "future_valid")
        }

        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        else:
            batch = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

        # Tensor-valued meta (V8 future_states / future_valid) bypasses the
        # preprocessor but still needs to land on-device for the aux loss.
        batch.update(
            {k: (v.to(self.device) if isinstance(v, Tensor) else v) for k, v in meta.items()}
        )
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
            "step=%d loss=%.4f gate_mean=%.4f gate_std=%.4f aux=%.4f",
            self._step,
            loss_dict.get("loss", float("nan")),
            loss_dict.get("gate_value_mean", float("nan")),
            loss_dict.get("gate_value_std", float("nan")),
            loss_dict.get("aux_future_state_loss", float("nan")),
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

    def _count_params_by_submodule(self) -> dict[str, dict[str, int]]:
        """Return parameter counts broken down by major submodule.

        Keys: ``vlm_backbone`` (text + vision, frozen),
        ``action_expert`` (lm_expert + action_out_proj, trainable from
        scratch), ``memory`` (FullSeqMemBank, trainable from scratch),
        ``other`` (catch-all for anything else attached to base_policy).
        Each value has ``total`` and ``trainable`` subtotals.
        """
        vwe = self.policy.base_policy.model.vlm_with_expert

        def _sum(mod: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            return total, trainable

        vlm_total, vlm_train = 0, 0
        for name, child in vwe.named_children():
            if name == "lm_expert":
                continue
            t, tr = _sum(child)
            vlm_total += t
            vlm_train += tr

        expert_total, expert_train = _sum(vwe.lm_expert)
        aout = getattr(self.policy.base_policy.model, "action_out_proj", None)
        if aout is not None:
            t, tr = _sum(aout)
            expert_total += t
            expert_train += tr

        mem_total, mem_train = _sum(self.policy.mem_bank)

        total_all = sum(p.numel() for p in self.policy.parameters())
        train_all = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        other_total = total_all - vlm_total - expert_total - mem_total
        other_train = train_all - vlm_train - expert_train - mem_train

        return {
            "vlm_backbone_frozen": {"total": vlm_total, "trainable": vlm_train},
            "action_expert_scratch": {"total": expert_total, "trainable": expert_train},
            "memory_scratch": {"total": mem_total, "trainable": mem_train},
            "other": {"total": other_total, "trainable": other_train},
            "all": {"total": total_all, "trainable": train_all},
        }

    def _dump_resolved_policy_config(self) -> None:
        """Serialize the resolved SmolVLA policy config + memory block + param counts.

        Writes ``<checkpoint_dir>/../resolved_policy_config.json`` so it
        can be diffed directly against the baseline's ``train_config.json``
        "policy" block. Silent drift from the baseline's hub config is the
        failure mode this guards against.
        """
        policy_cfg = dataclasses.asdict(self.policy.base_policy.config)

        mem = self.policy.mem_bank
        memory_block = {
            "injection_layer": self.policy.injection_layer,
            "inject_before": self.policy.feature_extractor._inject_before,
            "mem_length": mem.mem_length,
            "retrieval_layers": mem.retrieval_layers,
            "use_timestep_pe": mem.use_timestep_pe,
            "consolidate_type": mem.consolidate_type,
            "update_fused": mem.update_fused,
            "dataloader_type": mem.dataloader_type,
            "group_size": mem.group_size,
            "compression": mem.compression,
            "n_slots": mem.n_slots,
        }

        resolved = {
            "policy": policy_cfg,
            "_memory": memory_block,
            "_param_counts": self._count_params_by_submodule(),
            "tokenizer_padding_verdict": self._tokenizer_padding_verdict,
        }

        out_path = Path(self.cfg.checkpoint_dir) / "resolved_policy_config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(resolved, indent=2, default=str))
        logger.info(
            "Resolved policy config written to %s:\n%s",
            out_path,
            json.dumps(resolved, indent=2, default=str),
        )

    def _verify_tokenizer_padding(self) -> str:
        """Run the preprocessor on a 2-sample mini-batch and report the
        actual tokenizer padding mode. Writes the verdict to stdout and,
        if wandb is active, to the run config. Returns ``"unknown"`` when
        no preprocessor is attached (e.g. unit-test harnesses)."""
        if self.preprocessor is None:
            return "unknown"

        try:
            from lerobot.processor import TransitionKey  # lerobot >= 0.5
        except ImportError:
            from lerobot.processor.core import TransitionKey  # lerobot <= 0.4

        max_len = self.policy.base_policy.config.tokenizer_max_length
        prompts = ["pick up the cup", "please carefully stack the small bowl on the wooden crate"]
        batch = {
            "observation.state": torch.zeros(len(prompts), 8),
            "observation.images.image": torch.zeros(len(prompts), 3, 256, 256),
            "task": list(prompts),
        }
        tokenizer_idx = next(
            (i for i, s in enumerate(self.preprocessor.steps)
             if s.__class__.__name__ == "TokenizerProcessorStep"),
            None,
        )
        if tokenizer_idx is None:
            logger.warning("Trainer preprocessor has no TokenizerProcessorStep; skipping verdict.")
            return "unknown"

        transition = None
        for i, t in enumerate(self.preprocessor.step_through(batch)):
            transition = t
            if i - 1 == tokenizer_idx:
                break
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        mask = obs.get("observation.language.attention_mask")
        if mask is None:
            logger.warning("Trainer preprocessor produced no attention mask; skipping verdict.")
            return "unknown"

        mask = mask.detach().cpu().bool()
        padded_width = mask.shape[1]
        lengths = mask.sum(dim=1).tolist()
        longest = max(lengths)
        if padded_width == longest and padded_width < max_len:
            verdict = "longest"
        elif padded_width == max_len and padded_width > longest:
            verdict = "max_length"
        else:
            verdict = "ambiguous"
        logger.info(
            "tokenizer_padding_verdict=%s (padded_width=%d, lengths=%s, cap=%d)",
            verdict, padded_width, lengths, max_len,
        )
        if self._wandb is not None:
            self._wandb.config.update({"tokenizer_padding_verdict": verdict}, allow_val_change=True)
        return verdict

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
