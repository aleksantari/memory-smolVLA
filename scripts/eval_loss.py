"""Held-out loss evaluation for memory-augmented SmolVLA.

Computes flow-matching loss on validation episodes for each checkpoint,
enabling fair comparison between memory-augmented and baseline models
trained on the same SO100 dataset.

Usage:
    python scripts/eval_loss.py \
        --checkpoints checkpoints/final.pt checkpoints/injection_quarter/final.pt \
        --configs configs/memory_only.yaml configs/memory_injection_quarter.yaml \
        --labels "layer8_memory" "layer4_memory" \
        --val-ratio 0.2 \
        --output results/loss_comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import memory_smolvla.data._video_compat  # noqa: F401  # patch pyav for torchvision >= 0.26

from memory_smolvla.data.dataset_config import DatasetConfig
from memory_smolvla.data.episode_loader import EpisodeBoundary, EpisodeSequentialLoader
from memory_smolvla.policy.builder import build_policy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = cfg_path.parent / cfg.pop("_base_")
        base = load_config(str(base_path))
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_val_episode_indices(dataset_cfg: DatasetConfig, val_ratio: float):
    """Get episode indices for validation split.

    Takes the last val_ratio fraction of episodes as validation.
    Returns (train_indices, val_indices) as lists of (ds_idx, ep_idx).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    all_episodes = []
    for ds_idx, repo_id in enumerate(dataset_cfg.repo_ids):
        ds = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=dataset_cfg.delta_timestamps or None,
            root=dataset_cfg.local_cache_dir,
        )
        n_ep = ds.num_episodes
        all_episodes.append((ds_idx, repo_id, n_ep))
        logger.info("Dataset %s: %d episodes", repo_id, n_ep)

    total_eps = sum(n for _, _, n in all_episodes)
    n_val = max(1, int(total_eps * val_ratio))
    n_train = total_eps - n_val

    logger.info("Total episodes: %d, train: %d, val: %d", total_eps, n_train, n_val)
    return n_train, n_val, total_eps


class ValEpisodeLoader:
    """Loads only validation episodes (last N episodes of the dataset)."""

    def __init__(self, dataset_cfg: DatasetConfig, n_train: int, n_total: int):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._datasets = []
        for repo_id in dataset_cfg.repo_ids:
            ds = LeRobotDataset(
                repo_id=repo_id,
                delta_timestamps=dataset_cfg.delta_timestamps or None,
                root=dataset_cfg.local_cache_dir,
            )
            self._datasets.append(ds)

        self._n_train = n_train
        self._n_total = n_total

    def __iter__(self):
        """Yield frames from validation episodes only."""
        global_ep = 0
        for ds_idx, ds in enumerate(self._datasets):
            for ep_idx in range(ds.num_episodes):
                if global_ep < self._n_train:
                    global_ep += 1
                    continue

                # This is a validation episode
                yield EpisodeBoundary(episode_index=ep_idx, dataset_index=ds_idx)

                ep_meta = ds.meta.episodes[ep_idx]
                start = int(ep_meta["dataset_from_index"])
                end = int(ep_meta["dataset_to_index"])

                for frame_idx in range(start, end):
                    frame = ds[frame_idx]
                    # Add batch dim
                    batch = {}
                    for k, v in frame.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.unsqueeze(0)
                        else:
                            batch[k] = v
                    yield batch

                global_ep += 1


def evaluate_model(policy, val_loader, feature_map, preprocessor, device):
    """Run forward pass on all validation episodes, return avg loss and stats."""
    policy.eval()

    total_loss = 0.0
    n_frames = 0
    n_episodes = 0
    episode_losses = []
    current_ep_loss = 0.0
    current_ep_frames = 0
    gate_alphas = []

    for item in val_loader:
        if isinstance(item, EpisodeBoundary):
            # Save previous episode stats
            if current_ep_frames > 0:
                episode_losses.append(current_ep_loss / current_ep_frames)
                n_episodes += 1

            policy.reset_memory()
            current_ep_loss = 0.0
            current_ep_frames = 0
            continue

        # Apply feature map
        batch = {}
        for k, v in item.items():
            mapped_k = feature_map.get(k, k)
            batch[mapped_k] = v

        # Apply preprocessor (tokenizes language, normalizes images, moves to device)
        batch = preprocessor(batch)

        # Enable grad temporarily for forward pass (flow matching needs it internally)
        with torch.enable_grad():
            loss, loss_dict = policy(batch)

        loss_val = loss.item()
        total_loss += loss_val
        current_ep_loss += loss_val
        current_ep_frames += 1
        n_frames += 1

        if "gate_alpha_mean" in loss_dict:
            gate_alphas.append(loss_dict["gate_alpha_mean"])

        if n_frames % 50 == 0:
            logger.info("  frame %d, running_avg_loss=%.6f", n_frames, total_loss / n_frames)

    # Don't forget last episode
    if current_ep_frames > 0:
        episode_losses.append(current_ep_loss / current_ep_frames)
        n_episodes += 1

    avg_loss = total_loss / max(n_frames, 1)
    avg_gate = sum(gate_alphas) / max(len(gate_alphas), 1) if gate_alphas else None

    return {
        "avg_loss": avg_loss,
        "n_frames": n_frames,
        "n_episodes": n_episodes,
        "per_episode_loss": episode_losses,
        "avg_gate_alpha": avg_gate,
    }


def main():
    parser = argparse.ArgumentParser(description="Held-out loss evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--configs", nargs="+", required=True, help="Config paths (one per checkpoint)")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each model")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of episodes for validation")
    parser.add_argument("--output", default="results/loss_comparison.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--include-base", action="store_true",
                        help="Also evaluate raw SmolVLA base (no memory, no training)")
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.configs) == len(args.labels), \
        "Must provide same number of checkpoints, configs, and labels"

    # Load first config for dataset info
    cfg0 = load_config(args.configs[0])
    dataset_cfg_dict = cfg0.get("dataset", {})
    dataset_cfg = DatasetConfig(
        repo_ids=dataset_cfg_dict.get("repo_ids", []),
        delta_timestamps=dataset_cfg_dict.get("delta_timestamps", {}),
        split=dataset_cfg_dict.get("split", "train"),
        local_cache_dir=dataset_cfg_dict.get("local_cache_dir"),
    )
    feature_map = dataset_cfg_dict.get("feature_map", {})

    # Determine val split
    n_train, n_val, n_total = get_val_episode_indices(dataset_cfg, args.val_ratio)

    results = {"val_episodes": n_val, "val_ratio": args.val_ratio, "models": {}}

    # Evaluate each model
    for ckpt_path, cfg_path, label in zip(args.checkpoints, args.configs, args.labels):
        logger.info("=" * 60)
        logger.info("Evaluating: %s", label)
        logger.info("  checkpoint: %s", ckpt_path)
        logger.info("  config: %s", cfg_path)

        cfg = load_config(cfg_path)
        policy_cfg = cfg.get("policy", {})
        training_mode = policy_cfg.get(
            "training_mode", cfg.get("trainer", {}).get("training_mode", "memory_only")
        )

        policy = build_policy(
            training_mode=training_mode,
            num_vlm_layers=policy_cfg.get("num_vlm_layers", 16),
            base_checkpoint=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
            injection_layer=policy_cfg.get("injection_layer", 8),
            bank_max_size=policy_cfg.get("bank_max_size", 16),
            retrieval_n_heads=policy_cfg.get("retrieval_n_heads", 4),
            gate_hidden_dim=policy_cfg.get("gate_hidden_dim", 256),
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy = policy.to(args.device)
        logger.info("  loaded step %d", ckpt.get("step", -1))

        # Build preprocessor (tokenizes language, normalizes, moves to device)
        from lerobot.policies.factory import make_pre_post_processors
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.base_policy.config,
            pretrained_path=policy_cfg.get("base_checkpoint", "lerobot/smolvla_base"),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )

        val_loader = ValEpisodeLoader(dataset_cfg, n_train, n_total)
        model_results = evaluate_model(policy, val_loader, feature_map, preprocessor, args.device)
        results["models"][label] = model_results

        logger.info("  avg_loss=%.6f  n_frames=%d  n_episodes=%d  gate_alpha=%s",
                     model_results["avg_loss"],
                     model_results["n_frames"],
                     model_results["n_episodes"],
                     f"{model_results['avg_gate_alpha']:.4f}" if model_results["avg_gate_alpha"] is not None else "N/A")

        # Free GPU memory
        del policy, ckpt
        torch.cuda.empty_cache()

    # Evaluate raw base SmolVLA (no memory, no training) as baseline
    if args.include_base:
        logger.info("=" * 60)
        logger.info("Evaluating: raw_smolvla_base (no memory, no fine-tuning)")

        policy = build_policy(
            training_mode="memory_only",
            num_vlm_layers=16,
            base_checkpoint="lerobot/smolvla_base",
            injection_layer=8,
            bank_max_size=16,
            retrieval_n_heads=4,
            gate_hidden_dim=256,
        )
        # Don't load any checkpoint — use the model as-is (gate near zero = vanilla SmolVLA)
        policy = policy.to(args.device)

        from lerobot.policies.factory import make_pre_post_processors
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.base_policy.config,
            pretrained_path="lerobot/smolvla_base",
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )

        val_loader = ValEpisodeLoader(dataset_cfg, n_train, n_total)
        base_results = evaluate_model(policy, val_loader, feature_map, preprocessor, args.device)
        results["models"]["raw_smolvla_base"] = base_results

        logger.info("  avg_loss=%.6f  n_frames=%d", base_results["avg_loss"], base_results["n_frames"])

        del policy
        torch.cuda.empty_cache()

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("-" * 60)
    for label, res in results["models"].items():
        gate_str = f"  gate_alpha={res['avg_gate_alpha']:.4f}" if res.get("avg_gate_alpha") is not None else ""
        logger.info("  %-30s  loss=%.6f  (%d frames, %d eps)%s",
                     label, res["avg_loss"], res["n_frames"], res["n_episodes"], gate_str)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
