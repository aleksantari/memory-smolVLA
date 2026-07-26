"""Cached-forward equivalence: same batch, cache ON vs OFF → same flow loss.

Confirms the global_idxs → PrefixCache.lookup → _build_prefix_kv wiring produces
the same training loss as the live vision path (to fp16 precision). CPU/fp32; a
tiny hand-built batch over cached rows so it needs no GPU (won't touch training).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_smolvla.data._video_compat  # noqa
import train as t
from memory_smolvla.data.group_loader import _collate, LIBERO_REPO_ID, GroupedEpisodeLoader
from memory_smolvla.data.dataset_config import DatasetConfig

CACHE = "/tmp/claude-1000/-home-tarmus/e8da783b-9f3d-4f51-be00-f445165a1330/scratchpad/prefix_test"
REMAP = {"observation.images.image": "observation.images.camera1",
         "observation.images.image2": "observation.images.camera2"}


def main():
    from memory_smolvla.policy.builder import build_policy
    from memory_smolvla.data.prefix_cache import PrefixCache
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    cfg = t.load_config("configs/memvla_libero_v10.yaml"); pc = cfg["policy"]
    ov = dict(pc["overrides"]); ov["pad_language_to"] = "max_length"
    torch.manual_seed(0)
    policy = build_policy(
        base_checkpoint=pc["base_checkpoint"], injection_layer=15, inject_before=True,
        mem_length=4, retrieval_layers=2, consolidate_type="tome", dataloader_type="group",
        group_size=8, compression="mean_pool", n_slots=4, aux_loss_weight=0.1,
        reasoning_enabled=True, reasoning_injection_layer=8, reasoning_n_slots=8,
        warm_start=None, policy_overrides=ov,
    ).float().eval()

    dc = cfg["dataset"]
    ds_cfg = DatasetConfig(delta_timestamps=dc.get("delta_timestamps", {}), split="train", local_cache_dir=None)
    deltas = GroupedEpisodeLoader._resolve_deltas(ds_cfg, policy.base_policy.config)
    ds = LeRobotDataset(repo_id=LIBERO_REPO_ID, delta_timestamps=deltas, root=None, image_transforms=None)
    pre, _ = make_pre_post_processors(
        policy.base_policy.config, pretrained_path=pc["preprocessor_path"],
        preprocessor_overrides={"device_processor": {"device": "cpu"},
                                "tokenizer_processor": {"padding": "max_length"}})

    # hand-build a batch over cached rows 0..15 (2 groups of 8)
    idxs = list(range(16))
    frames = [ds[i] for i in idxs]
    batch = _collate(frames); batch = {REMAP.get(k, k): v for k, v in batch.items()}
    batch = pre(batch)
    batch = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}
    batch["episode_ids"] = [0] * 8 + [1] * 8
    batch["timesteps"] = list(range(8)) + list(range(8))
    batch["global_idxs"] = idxs
    batch["future_states"] = torch.zeros(16, policy.mem_bank.future_predictor.mlp[-1].out_features
                                         if getattr(policy.mem_bank, "future_predictor", None) else 8)
    batch["future_valid"] = torch.zeros(16)

    actions = policy.base_policy.prepare_action(batch)
    torch.manual_seed(1)
    noise = policy.base_policy.model.sample_noise(actions.shape, actions.device)
    time = torch.rand(actions.shape[0]) * 0.98 + 0.01

    cache = PrefixCache(CACHE); cache.check_compatible(policy)

    def run(use_cache):
        policy.attach_prefix_cache(cache if use_cache else None)
        policy.mem_bank.reset(); policy.reasoning_bank.reset()
        with torch.no_grad():
            loss, _ = policy.forward(batch, noise=noise, time=time)
        return loss.item()

    l_live = run(False)
    l_cache = run(True)
    rel = abs(l_cache - l_live) / max(abs(l_live), 1e-8)
    print(f"loss live={l_live:.6f}  cached={l_cache:.6f}  rel_diff={rel:.2e}")
    ok = rel < 5e-3   # fp16 prefix vs fp32 live; well below bf16 training noise
    print("CACHE-FORWARD:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
