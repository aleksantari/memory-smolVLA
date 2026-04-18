# Claude Code task: Audit and fix memvla training config drift from baseline

## Context

The memvla training run must replicate the baseline SmolVLA LIBERO fine-tuning recipe exactly, except for (a) memory-module additions and (b) the contiguous-episode group dataloader required for memory training. A review of the current `configs/memvla_libero.yaml` against the reference baseline config (`outputs/libero_baseline_v2/checkpoints/last/pretrained_model/train_config.json`) found that several SmolVLA policy fields have been dropped from the memvla config and now rely on code defaults, which may or may not match the baseline. This must be fixed before training.

The reference baseline config is attached below. Use it as the source of truth for every SmolVLA policy hyperparameter that is not memory-specific.

## Tasks — do them in this order

### Task 1. Surface the fully-resolved SmolVLA policy config the memvla trainer actually uses

Before changing anything, add a startup log to the memvla trainer that **prints the full resolved SmolVLA policy config** as a single dict/JSON blob — not just the memory-specific fields, the whole thing, including every field inherited from the base checkpoint, code defaults, or the yaml.

Print this once at trainer init, after the model is constructed but before the first training step. Target: `logger.info` at INFO level, and also write to a file `outputs/<run_name>/resolved_policy_config.json` so we can diff it against the baseline's `train_config.json`.

The fields I specifically want to see in this log:
- `chunk_size`
- `n_action_steps`
- `n_obs_steps`
- `resize_imgs_with_padding`
- `tokenizer_max_length`
- `max_state_dim`, `max_action_dim`
- `min_period`, `max_period`
- `attention_mode`
- `num_vlm_layers`, `num_expert_layers`, `self_attn_every_n_layers`, `expert_width_multiplier`
- `freeze_vision_encoder`, `train_expert_only`, `train_state_proj`, `load_vlm_weights`
- `vlm_model_name`
- `num_steps` (flow-matching steps)
- `use_cache`
- `normalization_mapping`
- `empty_cameras`, `adapt_to_pi_aloha`, `use_delta_joint_actions_aloha`
- `add_image_special_tokens`
- `prefix_length`, `pad_language_to`

Do not proceed to Task 2 until this log has been added and you have visually compared the output against the baseline `train_config.json` "policy" block.

### Task 2. Align the memvla policy config with the baseline

After Task 1 reveals any mismatches, fix them. The target values are the ones from the baseline `train_config.json` "policy" block. Key fields to verify or set explicitly:

| Field | Baseline value | Action |
|---|---|---|
| `chunk_size` | 50 | verify |
| `n_action_steps` | 10 | verify |
| `n_obs_steps` | 1 | verify |
| `resize_imgs_with_padding` | `[512, 512]` | **verify — this is the most likely silent mismatch** |
| `tokenizer_max_length` | 48 | verify |
| `max_state_dim`, `max_action_dim` | 32, 32 | verify |
| `min_period`, `max_period` | 4e-3, 4.0 | verify |
| `attention_mode` | `"cross_attn"` | verify |
| `num_vlm_layers` | 16 | verify |
| `num_expert_layers` | -1 | verify |
| `self_attn_every_n_layers` | 2 | verify |
| `expert_width_multiplier` | 0.75 | verify |
| `freeze_vision_encoder` | true | verify |
| `train_expert_only` | true | verify |
| `train_state_proj` | true | verify |
| `load_vlm_weights` | true | verify |
| `vlm_model_name` | `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"` | verify |
| `num_steps` | 10 | verify |
| `use_cache` | true | verify |
| `normalization_mapping` | `{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}` | verify |

**Decision point:** Whether to encode these explicitly in `memvla_libero.yaml` or inherit them from the base checkpoint is a design call. Prefer: inherit from the base checkpoint where it's guaranteed to match, but if inheritance is not clean (e.g., certain fields overridden by yaml defaults, or yaml omissions falling through to Python defaults rather than checkpoint values), make them explicit in the yaml. **When in doubt, be explicit in the yaml — silent drift is the failure mode we are trying to prevent.**

### Task 3. Confirm the base checkpoint is correctly set

**Resolved by human:** baseline_v2 was initialized from `lerobot/smolvla_base` (community-pretrained VLM + randomly-initialized action expert) and fine-tuned on `HuggingFaceVLA/libero` for 100k steps. This matches the SmolVLA paper's from-scratch action-expert ablation protocol.

The memvla config currently has:
```yaml
base_checkpoint: lerobot/smolvla_base
```

This is correct — it matches the baseline's initialization. **Verify, do not change:**

1. Confirm that `base_checkpoint: lerobot/smolvla_base` in the memvla yaml resolves to the same HuggingFace repo path that baseline_v2 was initialized from. If the memvla loader interprets this string differently than the baseline loader did, fix it so both resolve identically.
2. Confirm that loading this checkpoint produces: pretrained VLM weights (SigLIP + SmolLM2) + randomly-initialized action expert. Specifically, check that the action expert parameters are random, not loaded from `lerobot/smolvla_base`'s action expert weights if any exist.
3. Report: number of parameters loaded as pretrained vs. randomly initialized, broken down by component (VLM, action expert, memory module).

Report these three in the config alignment report. Do NOT change the `base_checkpoint` field.

### Task 4. Adjust `mem_length` and `group_size` for consolidation training

Currently:
```yaml
mem_length: 8
group_size: 8
```

With this setting, the memory bank fills exactly to capacity at the end of each episode-group but consolidation never triggers. At inference time, episodes exceed 8 timesteps and consolidation does kick in — so the model would encounter a distribution of bank states during eval that it never saw during training.

**Change to:**
```yaml
mem_length: 4
group_size: 8
```

This triggers 4 consolidation events per episode-group during training, so the token-merge consolidation path is actually exercised. This matches the spirit of MemoryVLA's training regime.

Other settings consistent with this change:
- `num_groups: 4`, `group_size: 8` → batch size 32 (matches baseline's `batch_size: 32`).
- Keep `consolidate_type: tome`.

### Task 5. Verify delta_timestamps spacing matches baseline

The memvla yaml uses 0.1-second spacing for action timestamps (50 entries from 0.0 to 4.9). This implies 10 Hz action prediction.

Look at how the baseline trainer built its `delta_timestamps`. If it used 0.1-second spacing, no change needed. If it used different spacing (LIBERO's native fps or some other value), align the memvla config.

Specifically, check whether the baseline used:
- The dataset's native fps (whatever `HuggingFaceVLA/libero` stores — likely 20 Hz for LIBERO)
- A fixed 10 Hz or other subrate
- Something derived from `env.fps: 30` in the baseline config (note: this is the env fps, not the dataset fps)

Report the baseline's delta_timestamps spacing. If it differs from 0.1 seconds, update the memvla yaml to match.

### Task 6. Verify eval rollout calls `mem_bank.reset()`

Confirm — by reading the code — that the LIBERO eval loop for the memvla model calls `model.mem_bank.reset()` (or equivalent) at the start of every rollout, before the first action is predicted.

If it does not, this is a critical correctness bug: memory from rollout N will bleed into rollout N+1 and eval numbers will be garbage. Add the reset call at the top of the rollout loop.

Also confirm that the memory bank is in `eval()` mode during rollout, which changes behavior in `FullSeqMemBank.process_batch` (the `if self.training:` branches for batch-boundary episode clearing should not fire during eval).

### Task 7. Add an eval config block to memvla yaml

The baseline has:
```json
"eval": { "n_episodes": 10, "batch_size": 1, "use_async_envs": false }
```

Add an equivalent block to `memvla_libero.yaml`:
```yaml
eval:
  n_episodes: 10
  batch_size: 1
  use_async_envs: false
```

### Task 8. Produce a diff report

After tasks 1–7, write a short markdown report `outputs/<run_name>/config_alignment_report.md` that contains:

1. A table of every SmolVLA policy field, showing baseline value vs. memvla resolved value. Flag any mismatches in bold.
2. Confirmation of the base checkpoint verification (Task 3): which components load pretrained weights, which are randomly initialized.
3. Confirmation of the `mem_length` / `group_size` / consolidation training change (Task 4).
4. Confirmation of delta_timestamps spacing match (Task 5).
5. Confirmation that eval calls mem_bank.reset() (Task 6).
6. Baseline sanity check results (Task 9).
7. Any fields where alignment was not possible (e.g., because the baseline value is ambiguous or not stored in the config). Flag these for human review.

### Task 9. Baseline sanity check — is baseline_v2 actually a good baseline?

Before running memvla, we need to know the baseline is sound. The baseline's configured `steps: 100000` and `eval_freq: 10000` and `save_freq: 20000` imply evaluations at 10k, 20k, ..., 100k. Verify:

1. **Did baseline_v2 actually complete 100k steps?** Look at the baseline's final checkpoint's step count or the training logs. If it stopped early (e.g., crashed at 60k and was never resumed), the memvla run should match whatever step count the baseline actually reached — do not run memvla longer than baseline was trained.

2. **What success rates did baseline_v2 achieve on LIBERO?** Look up the eval results logged during baseline_v2 training (wandb run `3vgb65vy` in the `smolvla-libero-baseline` project, or local eval logs). Report the final eval success rates broken out per suite:
   - LIBERO-Spatial
   - LIBERO-Object
   - LIBERO-Goal
   - LIBERO-10 (long-horizon)

3. **Sanity check against published numbers.** The SmolVLA paper reports LIBERO performance in the 87-96% range depending on suite. If baseline_v2 is wildly below this (e.g., <60% on any suite), flag it — there is likely a training or eval configuration issue that must be resolved before running memvla.

Report all three in the config alignment report. Do not launch memvla training until we have reviewed these numbers.

### Task 10. Per-suite eval reporting for memvla

The baseline was trained jointly across all four LIBERO suites (`env.task: "libero_spatial,libero_object,libero_goal,libero_10"`). This means the gradient signal during training averages across tasks where memory should help (LIBERO-10 / Long, non-Markovian) and tasks where memory should be roughly neutral (Spatial, Object, Goal — approximately Markovian).

At eval time, the memvla effect will concentrate on LIBERO-10. If we report a single averaged success rate across all four suites, the Long-suite improvement gets diluted by three no-change results and the memory contribution becomes invisible.

**Requirement:** the memvla eval must report per-suite success rates, not just an average. Specifically:

1. Ensure the eval loop evaluates on all four suites separately and logs each number to wandb as a distinct metric (`eval/libero_spatial/success_rate`, `eval/libero_object/success_rate`, `eval/libero_goal/success_rate`, `eval/libero_10/success_rate`).
2. Also log the four-suite average (`eval/libero_avg/success_rate`) for compatibility with baseline comparisons.
3. In addition to terminal success rate (binary task completion), if the eval infrastructure supports it, log per-step rollout length and failure-mode categorization for LIBERO-10 — the primary suite of interest. If this requires new eval instrumentation, flag it but do not implement unless trivial.
4. Confirm that the baseline's eval results (from Task 9) are also reported per-suite, so memvla vs. baseline can be compared suite-by-suite. If baseline only has averaged numbers, re-run baseline eval from its final checkpoint to get per-suite numbers.

## Non-goals — do NOT do these

- Do not attempt to "improve" any hyperparameter. The point of this task is fidelity to the baseline recipe, not optimization.
- Do not change batch size (`num_groups * group_size` must equal 32).
- Do not change the memory-specific fields (`injection_layer`, `retrieval_layers`, `use_timestep_pe`, `update_fused`, `dataloader_type`, `inject_before`). These are the intentional design choices, not drift.
- Do not modify the baseline config or baseline checkpoint. Only the memvla side changes.
- Do not touch the image augmentation block — it already matches baseline.

## Deliverables

1. Modified `configs/memvla_libero.yaml` with any Task 2/4/5/7 changes applied.
2. New startup logging in the trainer producing `resolved_policy_config.json` (Task 1).
3. `outputs/<run_name>/config_alignment_report.md` (Task 8), including baseline sanity check results (Task 9).
4. Confirmation or fix of eval reset (Task 6) — either a code change or a screenshot/quote of the existing reset call.
5. Confirmation that eval reports per-suite success rates (Task 10), with an example log snippet showing the wandb metric names.
6. Base checkpoint verification (Task 3): parameter counts broken down by pretrained vs. randomly-initialized.

Do not kick off any training runs. After these changes, I will review the alignment report (including baseline sanity check numbers from Task 9) and approve the training launch separately.
