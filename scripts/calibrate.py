"""Phase-0 hardware calibration for V7 on the DGX Spark (GB10).

Builds the exact training stack for a given config, runs a handful of real
optimizer steps, and reports median seconds/step and peak CUDA memory
(``torch.cuda.max_memory_allocated`` — nvidia-smi reports N/A for GB10's
unified memory, so in-process measurement is the only reliable signal).

Usage:
    python scripts/calibrate.py --config configs/memvla_libero_v7.yaml --steps 25
    python scripts/calibrate.py --config configs/memvla_libero_v7.yaml --steps 25 --compile

Use it to find the largest (num_groups, group_size) that fits with headroom,
and whether the 100k-step run clears the ~6 s/step / week budget.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as train_mod  # reuse config loading + image-transform builder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import memory_smolvla.data._video_compat  # noqa: F401

from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.data.group_loader import GroupedEpisodeLoader
from memory_smolvla.policy.builder import build_policy
from memory_smolvla.training.config import TrainerConfig
from memory_smolvla.training.trainer import MemorySmolVLATrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5, help="steps to discard before timing")
    ap.add_argument("--compile", action="store_true", help="torch.compile the base forward")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = train_mod.load_config(args.config)
    policy_cfg = cfg.get("policy", {})
    trainer_cfg_dict = cfg.get("trainer", {})
    dataset_cfg_dict = cfg.get("dataset", {})

    # never log calibration to wandb, never checkpoint
    trainer_cfg_dict = dict(trainer_cfg_dict)
    trainer_cfg_dict["wandb_project"] = None
    trainer_cfg_dict["checkpoint_every"] = 10**9
    trainer_cfg_dict["total_steps"] = args.steps
    trainer_cfg_dict["log_every"] = 10**9
    ckpt_dir = trainer_cfg_dict.get("checkpoint_dir", "checkpoints")
    if not Path(ckpt_dir).is_absolute():
        trainer_cfg_dict["checkpoint_dir"] = str(project_root / "checkpoints/_calib")

    group_size = policy_cfg.get("group_size", trainer_cfg_dict.get("group_size", 8))
    num_groups = trainer_cfg_dict.get("num_groups", 8)
    mem_length = policy_cfg.get("mem_length", 8)
    batch = num_groups * group_size

    torch.manual_seed(int(trainer_cfg_dict.get("seed", 1000)))

    policy = build_policy(
        base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
        num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
        injection_layer=policy_cfg.get("injection_layer", 15),
        inject_before=policy_cfg.get("inject_before", True),
        mem_length=mem_length,
        retrieval_layers=policy_cfg.get("retrieval_layers", 2),
        use_timestep_pe=policy_cfg.get("use_timestep_pe", True),
        consolidate_type=policy_cfg.get("consolidate_type", "tome"),
        update_fused=policy_cfg.get("update_fused", False),
        dataloader_type=policy_cfg.get("dataloader_type", "group"),
        group_size=group_size,
        compression=policy_cfg.get("compression", "none"),
        n_slots=policy_cfg.get("n_slots", 4),
        policy_overrides=policy_cfg.get("overrides") or None,
    )

    if args.compile:
        policy.base_policy.model = torch.compile(policy.base_policy.model)

    dataset_cfg = DatasetConfig(
        delta_timestamps=dataset_cfg_dict.get("delta_timestamps", {}),
        split=dataset_cfg_dict.get("split", "train"),
        local_cache_dir=dataset_cfg_dict.get("local_cache_dir"),
    )
    image_transforms = train_mod._build_image_transforms(dataset_cfg_dict.get("image_transforms"))
    train_loader = GroupedEpisodeLoader(
        cfg=dataset_cfg, group_size=group_size, num_groups=num_groups, mem_length=mem_length,
        image_transforms=image_transforms, policy_config=policy.base_policy.config,
        num_workers=int(trainer_cfg_dict.get("num_workers", 0)),
        prefetch_factor=int(trainer_cfg_dict.get("prefetch_factor", 4)),
        pin_memory=bool(trainer_cfg_dict.get("pin_memory", False)),
        seed=int(trainer_cfg_dict.get("seed", 1000)),
    )

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor_path = policy_cfg.get(
        "preprocessor_path",
        str(project_root / "outputs/libero_baseline_v2/checkpoints/last/pretrained_model"),
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.base_policy.config,
        pretrained_path=preprocessor_path,
        preprocessor_overrides={
            "device_processor": {"device": trainer_cfg_dict.get("device", "cuda")},
            "tokenizer_processor": {"padding": policy.base_policy.config.pad_language_to},
        },
    )

    tcfg = TrainerConfig(**{k: v for k, v in trainer_cfg_dict.items()
                            if k in TrainerConfig.__dataclass_fields__})
    trainer = MemorySmolVLATrainer(policy=policy, cfg=tcfg, train_loader=train_loader,
                                   preprocessor=preprocessor)

    policy.train()
    trainer.optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    dev = trainer.device
    dtype = trainer._amp_dtype

    times: list[float] = []
    it = iter(train_loader)
    print(f"CALIB config={args.config} batch={batch} (num_groups={num_groups} x "
          f"group_size={group_size}) mem_length={mem_length} "
          f"compression={policy_cfg.get('compression','none')} compile={args.compile}")
    for i in range(args.steps):
        t0 = time.perf_counter()
        batch_data = trainer._to_device(next(it))
        if dtype is not None:
            with torch.autocast(device_type=dev.type, dtype=dtype):
                loss, _ = policy(batch_data)
        else:
            loss, _ = policy(batch_data)
        loss.backward()
        trainer._optimizer_step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)
        print(f"  step {i:3d}  {dt:6.2f}s  loss={loss.item():.4f}", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    med = statistics.median(times) if times else float("nan")
    steps_100k_days = med * 100_000 / 86400
    print("=" * 60)
    print(f"RESULT batch={batch} median_s_per_step={med:.2f} peak_cuda_mem_GB={peak_gb:.1f}")
    print(f"       est 100k-step wall-clock = {steps_100k_days:.1f} days "
          f"({'OK <7d' if steps_100k_days < 7 else 'TOO SLOW — reduce batch or steps'})")
    print("=" * 60)


if __name__ == "__main__":
    main()
