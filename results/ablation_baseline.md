# SmolVLA LIBERO Baseline (v2)

**Date:** 2026-04-17
**Dataset:** `HuggingFaceVLA/libero` (all 4 suites combined)
**Run:** `libero_baseline_v2`
**Wandb run id:** `3vgb65vy` (two segments: `run-20260416_004440` steps 1–60K, `run-20260416_135006` steps 60K–100K after a clean pause/resume)
**Checkpoint used for final numbers:** `outputs/libero_baseline_v2/checkpoints/100000/pretrained_model/`
**Final per-suite JSONs:** `outputs/libero_baseline_v2_eval_per_suite/<suite>/eval_info.json`

This is the reference baseline for all memory-augmented ablations that follow. Future memory runs should use the same eval config (suites, episode count, `n_action_steps`) and report deltas against the numbers below.

## Configuration

**Policy — `SmolVLAPolicy` (base SmolVLA, loaded from HuggingFaceTB SmolVLM2-500M-Video-Instruct weights):**
- `freeze_vision_encoder=true`, `train_expert_only=true` — VLM frozen, only the action expert trains
- `use_amp=true` (mixed precision)
- `chunk_size=50`, `n_action_steps=10` — policy predicts 50-step chunks, executes 10 before replanning
- `num_steps=10` (flow-matching denoising steps)
- `num_vlm_layers=16`, `self_attn_every_n_layers=2`, `expert_width_multiplier=0.75`
- `attention_mode=cross_attn`, `prefix_length=-1`, `pad_language_to=longest`

**Optimizer / schedule:**
- AdamW, `lr=1e-4`, `betas=(0.9, 0.95)`, `weight_decay=1e-10`, `grad_clip_norm=10.0`
- Cosine decay with warmup: 1,000 warmup steps → decay over 100,000 steps → floor `lr=2.5e-6`

**Data / training:**
- Image transforms enabled: ColorJitter (brightness/contrast/saturation/hue), SharpnessJitter, RandomAffine (±5°, 5% translate); up to 3 tfs per sample
- `batch_size=32`, `num_workers=8`, `seed=1000`
- 100,000 steps (~11.7 epochs, ~3.2M samples)
- `save_freq=20,000`, `eval_freq=10,000`

**Evaluation:**
- All 4 LIBERO suites — `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`
- 10 tasks × 10 episodes = 100 episodes per suite, 400 episodes per checkpoint
- `eval.batch_size=1`, `init_states=true`, MUJOCO EGL backend
- Inference uses `n_action_steps=10` (matches paper's Table 13 best-case config)

## Final Results (step 100K)

Per-suite numbers below come from a dedicated post-training eval pass: `lerobot-eval` run separately per suite against the 100K checkpoint (`n_action_steps=10`, 100 episodes/suite). Raw JSONs at `outputs/libero_baseline_v2_eval_per_suite/<suite>/eval_info.json`.

| Suite | Ours (100K, n=10) | Paper (n=10, Table 13) | Δ |
|-------|------------------:|-----------------------:|----:|
| libero_spatial | **84.0** | 89 | −5.0 |
| libero_object  | **99.0** | 94 | +5.0 |
| libero_goal    | **96.0** | 91 | +5.0 |
| libero_10 (long) | **72.0** | 57 | +15.0 |
| **Overall (avg of 4 suites)** | **87.75** | **82.8** | **+4.95** |

Overall from the post-training per-suite eval: `(84 + 99 + 96 + 72) / 4 = 87.75`. Training-time eval at the same checkpoint reported `eval/pc_success = 87.25` — the 0.5pp drift is standard rollout noise at 100 episodes/suite. The per-suite numbers above are the canonical reference for future comparisons.

## Training Trajectory (per checkpoint)

| Step | Overall | Spatial | Object | goal+long (derived avg) |
|-----:|--------:|--------:|-------:|------------------------:|
| 10K  | 41.25 | 42.0 | 53.0 | 35.0 |
| 20K  | 62.75 | 68.0 | 85.0 | 49.0 |
| 30K  | 72.50 | 61.0 | 93.0 | 68.0 |
| 40K  | 79.75 | 79.0 | 96.0 | 72.0 |
| 50K  | 78.25 | 73.0 | 92.0 | 74.0 |
| 60K  | 83.00 | 82.0 | 96.0 | 77.0 |
| 70K  | 84.75 | 83.0 | 96.0 | 80.0 |
| 80K  | 86.25 | 85.0 | 97.0 | 81.5 |
| 90K  | 86.25 | 85.0 | 95.0 | 82.5 |
| 100K | 87.25 | 84.0 | 98.0 | 83.5 |

Overall improves monotonically after 50K (small dip at 50K is within eval noise with 100 episodes/suite). Spatial plateaus around 84–85 after 80K; object saturates to 95–98 after 60K.

## Comparison to v1 Baseline

| Run | `n_action_steps` | Batch | Overall | Notes |
|-----|-----------------:|------:|--------:|-------|
| v1  | 50 (SmolVLA default) | 8 | 30.5 | Same architecture, but inference at `n=50` matches the paper's worst-case row in Table 13 (`n=50 → 51.8%`); additional gap likely from small batch. |
| **v2** | **10** | **32** | **87.25** | Matches paper's reported-best inference config. |

The jump from 30.5 → 87.25 is primarily attributable to switching `n_action_steps` from 50 → 10 at inference, consistent with SmolVLA paper Table 13 (51.8% at n=50 vs 82.8% at n=10).

## Caveats

- **Per-step goal / long (trajectory table)**: during training, the combined `Suite per_group aggregated: {...}` log line is truncated by interleaved INFO lines before the `libero_goal` / `libero_10` entries, so only spatial + object are recoverable per-checkpoint. The goal+long column in the trajectory is derived as `overall × 4 − spatial − object`. Final-checkpoint per-suite numbers (table above) were backfilled by a dedicated per-suite eval pass.
- **Pause/resume**: training was paused cleanly at step 60K (GPU reclaimed) and resumed via `--config_path=.../train_config.json --resume=true`; the resumed segment continues the cosine schedule without discontinuity. Both wandb segments share `run_id=3vgb65vy`.
- **Eval stochasticity**: 10 episodes/task × 10 tasks = 100 episodes/suite. ±2–3pp per-suite noise is expected (e.g. object 98 during training → 99 on re-eval).

## Takeaways

1. **v2 exceeds paper numbers** (+4.95pp overall) under the paper's recommended inference config. Sound reference baseline.
2. **`n_action_steps=10` is essential.** The v1 result of 30.5% was a measurement artifact of the SmolVLA default `n=50`, not a training failure.
3. **Biggest gain is on `libero_10` (long-horizon): +15pp over paper (72 vs 57).** Goal and object also beat paper by +5pp each; spatial is the only suite slightly under paper (−5pp).
4. **Object is near-saturated (99%).** Remaining headroom sits in spatial and `libero_10` — the latter being exactly the kind of task a temporal memory module should further improve.

## Next Step

The memory-augmented runs (to be benchmarked next) must use this baseline's inference config verbatim: `n_action_steps=10`, all 4 LIBERO suites, 10 episodes/task, same `chunk_size`, same `num_steps`. Per earlier discussion, memory runs should also consider **unfreezing the action expert** rather than training only memory modules against a frozen expert.
