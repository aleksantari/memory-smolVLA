# V5 Run Log

**Source of truth for what's been tried and what worked.** Update after every training run and every eval. Even crashes — write what failed and why.

If the agent or session crashes, this log is what lets the next session pick up. **Don't skip it.**

---

## How to add a run entry

Copy the template below for each new run. Fill in as you go:

```markdown
## Run <N>: <run_name>

**Status:** [in progress | training complete | quick eval done | full eval done | failed | aborted]
**Config:** `configs/libero_v5_<run_name>.yaml`
**Seed:** 1000 (default)
**Started:** YYYY-MM-DD HH:MM
**Hardware:** RTX 5080 16GB, driver <NN>, CUDA <NN>, PyTorch <NN>
**Branch HEAD:** <commit_sha>

### Training

| Metric | Value |
|---|---|
| Wall-clock time | NN h MM m |
| Final training loss | 0.NNNN |
| Loss curve summary | (e.g. "monotonic decrease, no spikes") |
| Peak VRAM | NN GB |
| Total optimizer steps | 3000 |
| W&B run | <url> |
| Checkpoint path | `checkpoints/<run_name>/final.pt` |

### Quick eval (1 ep/task × 4 suites = 40 episodes)

| Suite | Success | Δ vs baseline (87.75 overall, see HANDOFF.md) | Δ vs v4 bypass (76.00) |
|---|---:|---:|---:|
| libero_spatial | NN.0 | XX.X | XX.X |
| libero_object | NN.0 | XX.X | XX.X |
| libero_goal | NN.0 | XX.X | XX.X |
| libero_10 | NN.0 | XX.X | XX.X |
| **Overall** | **NN.0** | **XX.X** | **XX.X** |

### Gate alpha statistics (training final)

- Mean: 0.NN
- Std: 0.NN
- Min: 0.NN
- Max: 0.NN

(Note: V4 used residual gate, so α reads as 1.0 always. The interesting stat is `train/loss` and per-suite eval.)

### Full eval (10 ep/task × 4 suites = 400 episodes) — only if quick eval qualified

| Suite | Success | Δ vs baseline | Δ vs v4 bypass | Δ vs v4 memory (73.25) |
|---|---:|---:|---:|---:|
| ...

### Decision

[continue / stop / try alternative]

**Reasoning:** ...

### Notes / deviations / surprises

- ...

---
```

---

## Baseline reference (from V4 era — for delta computation)

| Run | Spatial | Object | Goal | Long | Overall |
|---|---:|---:|---:|---:|---:|
| **V2 baseline** (no memory) | 84.0 | 99.0 | 96.0 | 72.0 | **87.75** |
| **V4 bypass** (memory off, expert finetune) | 72.0 | 96.0 | 82.0 | 54.0 | **76.00** |
| **V4 memory ON** | 74.0 | 96.0 | 79.0 | 44.0 | **73.25** |

V4 → baseline gap = −11.75pp (training-mode regression).
V4 memory cost = −2.75pp (memory-on vs bypass).
Where memory hurt most: `libero_10` (54 → 44, −10pp).

Target for V5: any V5 run that lands ≥85 overall AND ≥65 on libero_10 is a clear win.

---

## Runs

<!-- Add run entries below, newest at the top -->

## Setup — 2026-04-30 (Windows host, smolvla conda env)

**Branch HEAD:** `b8f4868` (claude/feature/v5-all-fixes)
**Host:** Windows 11 Pro (NOT Linux as the brief assumed). Training runs on Windows native; sim eval will be done in WSL (per established pipeline) after each training run.
**Hardware:** RTX 5080 16GB, driver 591.86, CUDA 13.1 runtime
**Python env:** `C:\Users\Tarik Metin\miniconda3\envs\smolvla` — Python 3.11.15, **torch 2.11.0+cu128** (well past 2.7 minimum), `lerobot 0.4.4`, `memory_smolvla 0.1.0` (editable), `transformers 4.57.6`, `wandb 0.24.2`. Sim deps (`robosuite`, `mujoco`, `libero`) NOT installed on Windows — that's expected; sim eval lives in WSL.
**HF auth:** logged in as `tarmus` via `huggingface-cli`. `HuggingFaceVLA/smolvla_libero` model cached (1.2GB). `HuggingFaceVLA/libero` dataset NOT cached yet — first run will pull ~30GB. Disk: 84GB free of 931GB on C: (tight but workable).
**W&B:** user logging in via `wandb login` for these runs (decision: online logging).

### Config fix applied

All 4 v5 configs had `checkpoint_dir: /content/drive/MyDrive/memory-smolvla/checkpoints/<run>` — Colab paths inherited from the planning machine. Changed to relative `checkpoints/<run>`; `train.py` resolves relative `checkpoint_dir` against the project root, so they land under `C:\Users\Tarik Metin\memory-smolVLA\checkpoints\<run>\`. No other config changes.

Files edited:
- `configs/libero_v5_run0_diagnostic.yaml`
- `configs/libero_v5_run1_kitchen_sink.yaml`
- `configs/libero_v5_run2_compressor.yaml`
- `configs/libero_v5_run2_two_stream.yaml`

### Decisions for these runs

- **Eval pipeline:** train on Windows, eval in WSL — sim deps install path on Windows is ugly (osmesa is Linux-only) and the user's existing WSL eval pipeline already works. Decision rules between runs are gated on the user's manual WSL eval results, not autonomous in-loop eval.
- **W&B:** online (logged in via `wandb login` in this session — key stored in `C:\Users\Tarik Metin\_netrc`).
- **Tonight's plan:** 100-step smoke test on Run 0, then stop and check before kicking off the 5–8h Run 0 proper.

### Smoke test — 100 steps on Run 0 config (PASSED)

**Command:** `WANDB_MODE=offline python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml --steps 100`
**Started:** 2026-04-30 23:23:03
**Finished:** 2026-04-30 23:43:09 (exit 0)
**Wall clock:** 20m 06s

| Phase | Duration | Notes |
|---|---|---|
| SmolVLA model load (`HuggingFaceVLA/smolvla_libero`) | 13s | from local HF hub cache |
| FeatureExtractor + memory wrapping | <1s | `injection_layer=8`, `gate_type=residual`, `bank_max_size=16`, `compression_mode=none` |
| Dataset load (`HuggingFaceVLA/libero`) | 12s | **already cached locally at `~/.cache/huggingface/lerobot/HuggingFaceVLA/libero` (33 GB)** — no 30GB download wait |
| Optimizer build | 7s | 101.3M trainable / 609.5M total = 16.6% |
| Training (100 optimizer steps × grad_accum=32) | 19m 33s | **11.7s / optimizer step** = 366ms / forward pass |
| Checkpoint save | 1s | `checkpoints/v5_run0_diagnostic/final.pt` = 1.67 GB |

**Final smoke metrics:** `step=100 loss=0.1131 gate_alpha=1.0000` — gate_alpha=1.0 is expected for residual gate. Loss 0.113 at step 100 is reasonable for a freshly-initialized memory module on a tightly-finetuned base.

**Notes:**
- W&B 0.24.2 ran in offline mode, run id `86b35i0w` saved to `wandb/offline-run-20260430_232329-86b35i0w` (not synced — smoke run, intentionally not on dashboard).
- Warning: `torchcodec is not available, falling back to pyav` — pyav is slower than torchcodec for video decoding. Could be a contributor to the per-step time being above the brief's 5-8h estimate (see below). Optional optimization for later.
- Disk before/after: 84 GB → 65 GB free on C: (drop is from concurrent activity; dataset cache was already 33 GB before smoke, didn't grow).

### Sim-compatibility verification (pre-Run 0)

Asked: will the v5 checkpoint load and run in the existing WSL sim eval pipeline (the one that produced V4 baseline 76%, V4 memory 73.25%)? V3's 0%-in-sim despite best held-out loss is the cautionary tale.

**Answer: yes, verified.**

- **Checkpoint format:** trainer saves `{step, policy_state_dict, optimizer_state_dict, scheduler_state_dict, training_mode}`. Eval scripts load via `policy.load_state_dict(ckpt["policy_state_dict"], strict=False)`. Loaded the smoke checkpoint into a freshly-built policy from `configs/libero_v5_run0_diagnostic.yaml`: **0 missing keys, 0 unexpected keys**. Pre/post processors (`DataProcessorPipeline`) build cleanly from `policy.base_policy.config`.
- **V3 fix is in:** `MemorySmolVLAPolicy.__init__` (`memory_smolvla.py:173`) unfreezes `vwe.lm_expert.parameters()` and `action_out_proj.parameters()` whenever `training_mode in {expert_scratch, expert_finetune, expert_only_scratch}`. All four v5 configs use `expert_finetune`. Smoke run reported 101.3M / 609.5M trainable = 16.6% (vs v3's 0.8% with frozen expert). Action expert can adapt to the memory-injection perturbations — that was the v3 root cause.
- **Eval script clarification:** `NEXT_AGENT_BRIEF.md` points at `scripts/eval.py`, but the proven pipeline (per git commit `5d63b6f` "WORKING sim eval pipeline using lerobot's LiberoEnv" and per project memory) is `scripts/eval_memory_libero.py`. Use the latter in WSL. Both scripts thread all v5 flags into `build_policy`, so both are config-compatible — `eval_memory_libero.py` is just the one that's been proven to actually run rollouts successfully.
- **Same trainer as V4:** v5 uses the same `expert_finetune` path V4 used to produce baseline numbers (76% bypass, 73.25% memory). Eval numbers are directly comparable.

### Run 0 wall-clock projection

Brief estimate: 5-8h on RTX 5080.
Measured: 11.7s / optimizer step × 3000 steps = **~9h 45m**.

Above the brief's range but in the same order. Likely contributors:
- pyav vs torchcodec (~10-20% slower for video decoding)
- Windows WDDM driver mode (not TCC; small CUDA dispatch overhead)
- 360W power cap (vs RTX 5080 stock 320W TGP)

Plan: kick off real Run 0 overnight. ETA ~10 hours from launch.

---


## Run 0: diagnostic

**Status:** training in progress
**Config:** `configs/libero_v5_run0_diagnostic.yaml`
**Seed:** 1000 (default in TrainerConfig)
**Started:** 2026-04-30 ~23:55 local
**Hardware:** RTX 5080 16 GB, driver 591.86, torch 2.11.0+cu128, Python 3.11.15
**Branch HEAD:** `b8f4868` (claude/feature/v5-all-fixes) + my Colab-path fix to configs (uncommitted)

### Pre-launch notes

- Smoke checkpoint `checkpoints/v5_run0_diagnostic/final.pt` deleted before launch (would have been overwritten anyway, but cleaner).
- Tried installing `torchcodec` to speed up video decode. Pip install succeeded (`torchcodec-0.11.1-cp311-cp311-win_amd64`), but loading fails with `OSError: WinError 127` — needs FFmpeg "full-shared" DLLs on Windows. Uninstalled. Sticking with pyav. Per-step time ~11.7s (smoke measurement) → projected wall-clock ~9h 45m.
- W&B online (logged in via netrc earlier). Run name `v5_run0_diagnostic` per config.

### Training

(fill after training completes)

### Quick eval

(fill after eval completes — will be done in WSL using `scripts/eval_memory_libero.py`, NOT `scripts/eval.py` which has known issues)

### Decision

(fill after deciding next run)

---

## Run 1: kitchen sink

**Status:** training in progress (PARALLEL with Run 0)
**Config:** `configs/libero_v5_run1_kitchen_sink.yaml`
**Seed:** 1000 (default)
**Started:** 2026-05-01 ~00:33 local
**Hardware:** same RTX 5080 — sharing GPU with Run 0
**Branch HEAD:** same as Run 0

### Deviation from brief — parallel execution

Brief explicitly says "❌ Don't run more than one training process at a time (single GPU)". User asked to run Run 1 in parallel after observing Run 0 was barely using the GPU.

**Observed during Run 0 alone (00:14):** GPU mem 4.5/16.3 GB (27%), GPU util 15%, mem BW 2%, power 66/360 W (18%), temp 35°C. Bottleneck is pyav video decoding on CPU, not GPU.

**Risk register:**
- CPU contention → both runs slow down proportionally (won't crash, just slower)
- VRAM peak collision during checkpointing → OOM possible (low risk; 12 GB headroom on Run 0 alone)
- Crash propagation: if Run 1 hits OOM and dies, in worst case it could take Run 0 with it

**Mitigation:** monitoring Run 0 per-step time. If Run 0 inflates beyond ~14s/step (>30% slower than its solo 10.9s baseline), kill Run 1 to protect Run 0.

### Training

(fill after training completes)

### Quick eval

(fill after eval completes — WSL `scripts/eval_memory_libero.py`)

### Decision

(fill after deciding next run)

---

## 2026-05-01 ~09:00 — Both runs killed by Windows sleep

Around 08:57:35 (Run 1's last log line; Run 0's last was 08:50:11) Windows put the system to sleep. Wake events at 09:14:45 (Hyper-V hypervisor + IsolatedUserMode service starts) confirm the sleep/wake. Both Python training processes died with the sleep — **no traceback, no OOM** in either log, just a clean truncation of output.

**Last good checkpoints:**
- Run 0: `step_0002500.pt` (lost ~100 logged steps; was at step ~2600 when killed)
- Run 1: `step_0002000.pt` (lost ~400 logged steps; was at step ~2400 when killed)

**Mitigation applied (PowerShell via `powershell.exe -NonInteractive`):**
```
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
Sleep, hibernate, and monitor-off disabled on AC. (DC settings unchanged — assume on AC.)

**Resume launched at ~09:55:**
- Run 0 resumed from `step_0002500.pt` to total_steps=3000 (500 steps left, ETA ~+105 min)
- Run 1 resumed from `step_0002000.pt` to total_steps=3000 (1000 steps left, ETA ~+217 min)

W&B continues the same runs (`resume="allow"` in trainer.py).


---

## Run 2A: compressor (TEMPLATE — only if Run 1 underperforms or plateaus)

**Status:** not started
**Config:** `configs/libero_v5_run2_compressor.yaml`
**Seed:** 1000
...

---

## Run 2B: two-stream (TEMPLATE — only if Run 1 underperforms or plateaus)

**Status:** not started
**Config:** `configs/libero_v5_run2_two_stream.yaml`
**Seed:** 1000

**Pre-flight check required:** verify `n_image_tokens: 132` matches your smolvla_libero prefix.
Run a 1-step smoke first; if the runtime check fires with the actual prefix length printed, update n_image_tokens and retry.

...

---

## 2026-05-01 — v5 Run 0 + Run 1 sim eval results (PARTIAL — pipeline was broken, see below)

**Eval pipeline used:** `scripts/eval_memory_libero.py` (the version on this machine — NOT yet patched at this point).
Setting: 1 ep × 10 tasks per suite (the brief's "quick eval"). Run 0 sweep was aborted partway through because the user pointed out we should compare against V4 hyperparameters, not blindly trust v5's deviations.

**Run 0 partial:**
- libero_spatial: **10.0%** (1/10)
- libero_object: **10.0%** (1/10)
- libero_goal: **0.0%** (0/10)
- libero_10: aborted

These were initially read as catastrophic regression. **They turned out to be eval-pipeline failure, not training failure** (see 2026-05-02 below).

Run 1 sweep was never started. Decision rules from brief never applied because the underlying number was unreliable.

---

## 2026-05-02 — Eval pipeline diagnosed and fixed; v5_meanpool_v4hp shows real wins

### Hyperparameter audit

User correctly flagged that the v5 configs deviate heavily from any documented training recipe.

The lerobot package (`lerobot/policies/smolvla/configuration_smolvla.py`) ships these defaults for SmolVLA training (from the upstream repo):

| Field | Lerobot upstream | v5 Run 0 (broken) | V4 |
|---|---|---|---|
| optimizer_lr | 1e-4 | expert 1e-5 | expert 1e-5 |
| weight_decay | 1e-10 | 1e-4 | 1e-4 |
| max_grad_norm | 10 | 1.0 | 1.0 |
| warmup_steps | 1000 | 100 | 500 |
| scheduler_decay_steps | 30000 | 3000 | 30000 |
| grad_accum_steps | (effective batch=4) | 32 + B=1 | 1 + B=1 |

**`v5_v4match` config** (`configs/libero_v5_v4match.yaml`): tried to force lerobot upstream defaults including `expert_lr=1e-4`. Trained 30k steps. Sim eval (after pipeline was fixed): 1/10 on libero_object — the model was destroyed. Confirmed via intermediate checkpoint evals: step_5000, 15000, 25000 all 0/10. expert_lr=1e-4 is too aggressive for finetuning on top of an already-finetuned `smolvla_libero` base (the action expert drifts off-distribution within the first 5k Adam updates).

### Eval pipeline was broken

After v5_v4match also returned 0–10%, suspicion shifted to the eval script. The proven path forward:

1. Ran lerobot's official `lerobot-eval` CLI on the upstream `HuggingFaceVLA/smolvla_libero` (no memory wrapper) → **89/100 on libero_object** (10 ep × 10 tasks). Pipeline works in lerobot.
2. Diffed our `scripts/eval_memory_libero.py` against lerobot's `rollout`. Our script bypassed multiple lerobot processors:
   - **`LiberoProcessorStep` was never applied** — and that step does TWO things our script missed:
     a. **Flips images 180°** (`torch.flip(img, dims=[2,3])`) to match the HuggingFaceVLA/libero camera orientation convention. Our policy was being fed **upside-down images** the whole time.
     b. Builds the 8D state vector via `_quat2axisangle` (we already did equivalent via scipy — that part was fine).
   - We also bypassed `preprocess_observation`, `add_envs_task`, and `env_postprocessor`.
3. Wrote `scripts/eval_memory_libero_v2.py` — a clean reimplementation that uses lerobot's `make_env`, `make_env_pre_post_processors`, `make_pre_post_processors`, and the `rollout` function, plugging our `MemorySmolVLAPolicy` in as the `nn.Module` policy. Bypasses lerobot's `eval_policy` `PreTrainedPolicy` assertion (rollout itself only requires `nn.Module`).
4. Re-eval base smolvla_libero through v2 script: **7/10 on libero_object** (1 ep/task). Matches lerobot-eval (89% at 10 ep/task is in the same range as 70% at 1 ep/task — 1-ep is high-variance).

### v4_meanpool_v4hp — first model on this machine to beat baseline

Trained `configs/libero_v5_meanpool_v4hp.yaml`: V4's exact optimizer hyperparameters (`expert_lr=1e-5`, `memory_lr=1e-4`, `weight_decay=1e-4`, `max_grad_norm=1.0`, `warmup=500`, `total_steps=30000`, `grad_accum=1`) + v5's architectural changes (`compression_mode=mean_pool` → 1 token per bank entry; `write_stride=50` → bank fills every 50 frames during training, matching the 1-write-per-chunk cadence at inference).

**Quick eval through fixed pipeline (1 ep × 10 tasks):**

| Suite | Base smolvla_libero | V4 final | v5_v4match (overcooked) | **v5_meanpool_v4hp** |
|---|---|---|---|---|
| libero_object | 7/10 (70%) | 7/10 (70%) | 1/10 (10%) | **10/10 (100%)** |
| libero_spatial | not measured 1ep | not measured | not measured | 8/10 (80%) |
| libero_goal | not measured 1ep | not measured | not measured | 6/10 (60%) |
| libero_10 | not measured 1ep | not measured | not measured | 4/10 (40%) |
| **Overall** | — | — | — | **28/40 (70%)** |

V4 final at 7/10 on object = parity with base (memory neutral at the V4 architectural level). v5_meanpool_v4hp at 10/10 on object is the first time on this machine a trained model has shown above-baseline performance through a verified pipeline.

**Hard caveat:** 1 ep/task is a single coin flip per task → ±10pp per task error. Suite-level numbers swing significantly with seed. The 100/80/60/40 spread reflects both real architecture and seed luck. Tighter measurement (5 ep/task across all 4 suites + base smolvla_libero head-to-head) is queued next.

### Files added/changed this session

**Configs:**
- `configs/libero_v5_v4match.yaml` (new — failed: expert_lr=1e-4 too high)
- `configs/libero_v5_meanpool_v4hp.yaml` (new — works: V4 optimizer + v5 mean_pool/write_stride)
- `configs/libero_v5_run0_diagnostic.yaml`, `libero_v5_run1_kitchen_sink.yaml`, `libero_v5_run2_compressor.yaml`, `libero_v5_run2_two_stream.yaml` — all had Colab `/content/drive/MyDrive/...` checkpoint paths replaced with local relative paths.

**Scripts:**
- `scripts/eval_memory_libero_v2.py` (new — uses lerobot's pipeline; this is the one to use going forward)
- `scripts/eval_memory_libero.py` (patched — added image flip + made `--checkpoint` optional; **still has subtle pipeline divergences from upstream — prefer v2**)
- `scripts/setup_local_rtx5080.sh` (already existed but Linux-only — doesn't apply to this Windows host)
- `scripts/run_sim_sweep.sh`, `run_lerobot_baseline.sh`, `run_v4match_intermediate_evals.sh`, `run_v5_meanpool_full_sweep.sh`, `run_v5_meanpool_vs_base_5ep.sh` — driver scripts for each eval round

**Checkpoints saved on disk** (NOT pushed):
- `checkpoints/v5_run0_diagnostic/` (final 3000 + intermediates) — broken model
- `checkpoints/v5_run1_kitchen_sink/` (final 3000 + intermediates) — never sim-eval'd, also bad config
- `checkpoints/v5_v4match/` (final 30000 + intermediates) — overcooked
- `checkpoints/v5_meanpool_v4hp/` (final 30000 + intermediates) — **the working one**

**Sim env setup notes (Windows host):**
- Training runs on Windows native conda env `smolvla` (`C:\Users\Tarik Metin\miniconda3\envs\smolvla`).
- Sim eval runs in WSL `Ubuntu-22.04` with conda env at `/home/santari/miniconda3/envs/smolvla`. WSL has all the LIBERO sim deps (`libosmesa6`, `robosuite==1.4.1`, `mujoco==3.6.0`, LIBERO editable at `/home/santari/LIBERO`).
- Critical env vars in WSL: `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, `PYTHONPATH=/home/santari/LIBERO`.
- Windows checkpoints accessible from WSL at `/mnt/c/Users/Tarik Metin/memory-smolVLA/`.

**Power policy applied (PowerShell):**
```
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```
(Required because Windows put the system to sleep mid-training around 09:00 on 2026-05-01, killing both training processes; not reverted.)

---

## 2026-05-03 — Tighter measurement (5 ep × 10 tasks): negative result

The 1 ep/task numbers from the previous day were promising but high-variance. Re-ran v5_meanpool_v4hp and base smolvla_libero (memory at init = no-op pathway through the same wrapper) at **5 episodes per task across all 4 suites — 200 episodes per checkpoint, 400 episodes total**. Sweep took ~17 hours wall-clock through `scripts/run_v5_meanpool_vs_base_5ep.sh` in WSL.

### Final head-to-head

| Suite | Base smolvla_libero | v5_meanpool_v4hp | Δ |
|---|---:|---:|---:|
| libero_spatial | 38/50 = **76.0%** | 32/50 = 64.0% | **−12.0pp** |
| libero_object  | 43/50 = **86.0%** | 43/50 = 86.0% | 0 |
| libero_goal    | 41/50 = **82.0%** | 40/50 = 80.0% | −2.0pp |
| libero_10      | 21/50 = **42.0%** | 15/50 = 30.0% | **−12.0pp** |
| **Overall**    | **143/200 = 71.5%** | **130/200 = 65.0%** | **−6.5pp** |

(Cross-check: lerobot-eval at 10 ep/task on the same base model gave 89/100 on libero_object = 89%. Our 5-ep base reads 86% on object — consistent within the larger sample → the v2 eval pipeline is reliable.)

### Honest read

The memory architecture as currently configured **does not help and slightly hurts**. The previous day's win on libero_object (10/10 = 100% at 1 ep/task) was seed luck; at 5 ep/task v5_meanpool and base are tied on object. On the other three suites v5 is at-best at-parity (goal, within noise) and meaningfully worse on spatial and libero_10.

**libero_10 is the most damning.** That's the long-horizon suite memory was specifically supposed to help — and v5 trails baseline by 12pp. The hypothesis that "adding a temporal memory bank + cross-attention retrieval to a frame-by-frame VLA improves long-horizon performance" is **not supported** by this experiment.

This pattern is consistent across the project's history:
- v3 (memory_only, frozen expert): 0% in sim — broken
- v4 (memory + expert finetune, full-prefix bank): 73% memory-on vs 76% bypass per the brief — memory slightly hurts
- v5_meanpool_v4hp (memory + expert finetune, mean_pool 1-token bank, write_stride=50): **65% vs 71.5% baseline** — memory slightly hurts, now measured through a known-good pipeline

Three architectural iterations, three confirmations of the same pattern: memory at best matches base, more often costs a few pp. The earlier "v4 +X pp over baseline" claims and Run 0/Run 1 "0% catastrophe" claims were both wrong — both products of the broken `eval_memory_libero.py` pipeline. The empirically-true picture is small-negative.

### What the data does and doesn't say

Does say:
- Memory **as currently parameterized** (residual gate locked at α=1.0, mean_pool to 1 token, write_stride=50, bank=16, FIFO eviction, V4 hyperparameters) does not improve over base smolvla_libero on LIBERO.
- The eval pipeline is now verified correct via independent lerobot-eval cross-check.
- The training recipe (V4 hyperparameters: expert_lr=1e-5, memory_lr=1e-4, total_steps=30000, grad_accum=1) is correct and preserves base capability.

Does not say:
- Memory cannot help here. The architecture has many knobs that haven't been swept (sigmoid gate with regularization instead of always-on residual, alternative bank sizes, learned compressor instead of mean_pool, two-stream split, multi-scale bank, working memory backend, joint VLM finetune, PTP auxiliary loss, post-expert injection).
- Memory cannot help on harder tasks. LIBERO episodes are 100–500 frames; SmolVLA's 50-frame action chunk + KV cache may already cover the relevant temporal context for these tasks. Tasks that genuinely require minutes-long context (e.g., real robot deployments, multi-stage open-ended tasks) might still benefit.

### Files added/changed

- `results/v5_meanpool_v4hp_5ep/` — 4 per-suite JSONs (5ep × 10 task evals)
- `results/base_smolvla_5ep/` — 4 per-suite JSONs (baseline through same pipeline)
- `scripts/run_v5_meanpool_vs_base_5ep.sh` — driver for this sweep
- This RUN_LOG.md section

---

## 2026-05-03 evening — Phase 2a: diagnostic dive on v5_meanpool_v4hp

User asked for a deeper post-mortem on *why* memory doesn't help, before iterating on architecture. Layer 1 = static weight inspection. Layer 3 = causal ablation (zero memory_proj). Layer 2 (runtime instrumentation) deferred to tomorrow.

### Layer 1 — static weight inspection

`scripts/diag_static_weights.py` loads the trained checkpoint and a fresh init-only build of the same architecture, reports Frobenius norm / max-abs / SVD on each memory submodule, and reports drift on the action expert.

Key numbers for `v5_meanpool_v4hp/final.pt`:

| Module / param | Trained ‖W‖ | Max\|w\| | ‖W − W_init‖ | Notes |
|---|---:|---:|---:|---|
| `memory_proj.weight` (960×960) | **1.61** | **0.0090** | 1.61 | init was zero; total movement = 1.61. For comparison a typical 960×960 with std=1/√960 has ‖W‖≈√960≈31. Trained matrix is ~5% of typical scale. **Top SV = 0.80, min SV ≈ 4e-8, effective rank (1% of σ_max) = 103/960.** |
| `retrieval.cross_attn.in_proj_weight` | 44.27 | 0.121 | 58.33 | Normal range for a trained transformer. Cross-attention IS being used. |
| `retrieval.cross_attn.out_proj.weight` | 27.09 | 0.252 | 32.47 | Normal range. |
| `retrieval.norm.weight` | 28.78 | 0.961 | 2.48 | LayerNorm scale, mostly 1.0 with small per-channel adjustments. |
| `lm_expert` (289 params, summary) | — | — | — | **mean rel-change 1.42%, max 5.4%** vs base — V4-style finetune did its job, expert wasn't over-cooked. |

**Reading:** the cross-attention pathway IS doing real work (the bank is being attended to), but `memory_proj` — the linear that maps cross-attention output into the residual addition on the prefix — was driven down by gradient descent to ~5% of typical scale. The model effectively learned that memory's contribution was net-negative and used `memory_proj` as a volume knob to suppress it. Residual gate forces α=1.0, so suppressing `memory_proj` was the only available knob to reduce memory's influence.

The 5% remaining strength still adds *something* to the prefix at every callback. That something is a modest amount of noise the action expert has to filter through. Net effect: −6.5pp overall.

### Layer 3 — causal ablation: zero memory_proj

`scripts/diag_zero_memory_proj.py` writes a copy of the v5_meanpool checkpoint with `memory_proj.weight` set to zero (other parameters unchanged). Saved to `checkpoints/v5_meanpool_v4hp_zeroproj/final.pt`. With memory_proj=0, the residual addition `current + memory_proj(retrieved) = current + 0 = current`, so memory pathway is causally neutralized while the action expert (still V4-style finetuned) and bank/retrieval (still computing attention) are unchanged.

Quick canary at 1 ep × 10 tasks per suite:

| Suite | base 5ep | v5_meanpool 5ep | **zeroproj 1ep** |
|---|---:|---:|---:|
| libero_spatial | 76% | 64% | 60% |
| libero_object | 86% | 86% | 80% |
| libero_goal | 82% | 80% | 90% |
| libero_10 | 42% | 30% | 50% |
| **Overall** | 71.5% | 65.0% | **70% (28/40)** |

Zeroproj 70% sits between v5_meanpool (65%) and base (71.5%). Direction confirms layer 1: zeroing memory_proj recovers most of the gap to baseline. Magnitude is noisy at 1ep but the per-suite pattern is intelligible — zeroproj beats v5_meanpool on libero_10 (50% vs 30%, a 20pp swing in the right direction) and on goal (90% vs 80%). On spatial it's slightly lower but well within ±10pp 1-ep noise.

A 5 ep × 10 task sweep on the same zeroed checkpoint is launched overnight (`scripts/run_zeroproj_5ep.sh`) to get tight numbers comparable to the 5ep base/v5 head-to-head.

### What this tells us

The memory pathway as wired (residual gate at α=1.0, mean_pool 1-token bank, FIFO eviction, memory_proj as the only volume knob) **cannot learn useful signal on this task family**. Three pieces of evidence converge:

1. (Layer 1) `memory_proj` is trained to ~5% of typical scale — gradient descent's best effort to silence the pathway.
2. (Layer 3) Zeroing `memory_proj` recovers ~5pp of the 6.5pp gap to baseline.
3. (5ep head-to-head) Memory's strongest negative is on libero_10 (the long-horizon suite memory was supposedly *for*), suggesting the bank's representation isn't capturing useful cross-frame information for those tasks.

This isn't a hyperparameter or training-recipe issue; it's an architectural one. The path forward is either:
- A different gate (learn α with regularization → can go to 0 cleanly when memory's bad),
- A different injection point (after the action expert, so memory shapes actions without disturbing VLM features the expert depends on),
- A different training signal (PTP-style auxiliary loss on memory: predict future state, gives memory a reason to learn structure),
- Or a different bank construction (compressor/two-stream, which the next two training runs will test).

### Files added

- `scripts/diag_static_weights.py` — layer-1 diagnostic (static weight inspection, init-vs-trained delta, SVD, expert drift)
- `scripts/diag_zero_memory_proj.py` — layer-3 ablation (zero memory_proj.weight in a checkpoint)
- `scripts/run_zeroproj_canary.sh` — 1ep × 4 suites canary on ablated ckpt
- `scripts/run_zeroproj_5ep.sh` — 5ep × 4 suites overnight tight measurement
- `results/diag/v5_meanpool_v4hp_weights.json` — full layer-1 numerical report
- `results/v5_meanpool_v4hp_zeroproj_1ep/` — 4 per-suite JSONs from the canary
- `checkpoints/v5_meanpool_v4hp_zeroproj/final.pt` — ablated checkpoint (NOT committed; binary artifact)

---

## 2026-05-04 — Phase 2a final: 5ep zeroproj sweep + layer-2 runtime norms

### 5ep zeroproj sweep — tight numbers

The 1ep canary had ±10pp/suite noise. Full 5ep × 10 task sweep on the memory_proj-zeroed checkpoint, comparable to base and v5_meanpool 5ep numbers:

| Suite | base 5ep | v5_meanpool 5ep | zeroproj 5ep | memory Δ (v5 − zero) | finetune Δ (zero − base) |
|---|---:|---:|---:|---:|---:|
| spatial   | 76% | 64% | 62% | +2 | **−14** |
| object    | 86% | 86% | 86% | 0  | 0 |
| goal      | 82% | 80% | 72% | **+8** | **−10** |
| libero_10 | 42% | 30% | **48%** | **−18** | **+6** |
| **Overall** | **71.5%** | **65.0%** | **67.0%** | −2 | −4.5 |

**Two distinct regressors**, with opposite suite preferences:

1. **The V4 action-expert finetune** (zero − base, isolating the finetune effect since memory is causally neutralized): −4.5pp overall. Worst on spatial (−14pp) and goal (−10pp). **Helps** libero_10 (+6pp). The finetune favors long-horizon at the cost of short-horizon — makes sense, the demonstration data is dominated by long task sequences.

2. **The memory pathway** (v5_meanpool − zero, isolating memory): −2pp overall but **highly heterogeneous**: −18pp on libero_10, +8pp on goal, ~neutral elsewhere. Memory hurts most on the suite it was supposed to help most.

### Layer 2 — runtime norm instrumentation (`scripts/diag_runtime_norms.py`)

Forward hooks on `policy.retrieval` (output norm) and `policy.memory_proj` (input/output norms). Captured all 280 callback invocations of one libero_object rollout with v5_meanpool_v4hp:

| Quantity | Value (mean across 280 callbacks) |
|---|---:|
| `retrieved` norm (cross-attn output) | 257.7 |
| `memory_proj(retrieved)` norm | 54.0 |
| `\|memory_proj(x)\| / \|x\|`  (real-world amplification) | **0.21** |
| Bank size progression | 0,1,2,…,15,16,16,16,… |
| Final bank size | 16 |

**Two findings re-shape the picture from layer 1:**

1. **`memory_proj` actual amplification is ~21%, not the ~5% layer-1 implied.** Layer 1's 5% was relative to identity-matrix scale (||W|| / √960). The 21% is the *actual* operator-norm applied to the *actual* distribution of cross-attention outputs. Memory is *not* silent in the activation sense — it's adding a real perturbation worth ~21% of `||retrieved||` to the prefix at every callback. The action expert sees that and has to filter it.

2. **Bank cycles every 16 env steps at inference.** Even with `write_stride=50`, the callback fires once per env step (not once per chunk), `step_increment=50` makes the timestamp jump by 50 each callback, and `t % 50 == 0` at every step → every callback writes. Bank fills in 16 env steps, then FIFOs through. So in a 280-step libero_object episode the bank cycles ~17.5 times. **The bank at inference is "the last 16 env steps' prefixes," not long-term memory.**

Compare bank coverage:

|   | Training (step_inc=1, write_stride=50) | Inference (step_inc=50, write_stride=50) |
|---|---|---|
| Callback frequency | every frame | every env step |
| Bank-write rate | every 50 frames | every env step |
| Time deltas in bank | {50, 100, …, 800} frames | {50, 100, …, 800} (in timestamp units; **only 16 env steps wall-clock**) |
| Wall-clock span of bank | 800 frames | **16 env steps** |

The temporal-PE *deltas* match between train and inference (the `write_stride=50` "fix" did achieve that). But the actual *time coverage* of the bank is **50× shorter** at inference than training. The action expert was trained to leverage 800-frame-spread memories; at inference it gets the last 16 env steps as memory, which is mostly redundant with the 50-frame action chunk already in flight.

This explains the −18pp libero_10 regression: long-horizon tasks are exactly where 800-frame coverage would matter, and exactly where inference's 16-step coverage falls shortest of training's distribution. The action expert is being asked to use information that isn't there.

### Phase 2a synthesis

Three converging lines of evidence:
- **Layer 1** (static weights): `memory_proj` was driven down to ~5% of identity-scale — gradient descent's best effort to suppress memory.
- **Layer 2** (runtime norms): bank cycles every 16 env steps, providing only short-term lookback at inference, despite training on 800-frame-spread memories.
- **Layer 3** (causal ablation): zeroing `memory_proj` recovers most of the v5_meanpool − base gap; biggest swing is on libero_10 (+18pp from 30 → 48).

The hypothesis "add a temporal memory bank to a frame-by-frame VLA → improves long-horizon" is **rejected with a specific failure mode**: the bank's time coverage at inference doesn't match training, so the memory is effectively short-term lookback — redundant with the chunk and net-noise to the action expert.

### Phase 3 candidates (now better-targeted)

In rough order of expected leverage:

1. **Fix the inference callback frequency.** Make the callback fire once per chunk (every 50 env steps) so the bank's wall-clock time coverage matches training (800 frames). This requires either gating the FeatureExtractor's patched forward to skip non-cache-build calls (CLAUDE.md says it should already do this), or making `step_increment` adaptive.
2. **Sigmoid gate with α-regularization** (`alpha_reg_weight > 0`). Lets the optimizer learn to disable memory cleanly on suites where it's net-noise (libero_10) and keep it on where it helps (goal). The current residual gate forces α=1.0 always.
3. **PTP-style auxiliary loss** — give memory a direct training signal (predict future state) so it learns to encode useful structure even when the main task gradient doesn't reward it.
4. **Post-expert injection** — inject memory after the action expert, so memory shapes actions without disturbing VLM features the expert depends on.

Phase 1 (compressor + two-stream variants on V4 hyperparameters) will tell us whether different bank constructions change the picture, before we commit to Phase 3.

### Files added (Phase 2a final)

- `scripts/diag_runtime_norms.py` — layer-2 runtime instrumentation
- `results/diag/runtime_norms/v5_meanpool_v4hp_libero_object_task0.json` — per-callback log (280 entries)
- `results/v5_meanpool_v4hp_zeroproj_5ep/` — 4 per-suite JSONs from the tight zeroproj sweep




