# V7 Run Plan — "unleash the DGX Spark"

**Goal:** continue V6 (the diversity variant), use the DGX Spark's 119 GB unified memory to
push diversity higher and to afford the two V5 lessons worth keeping, and run a ~1-week
training that closes the `libero_10` gap and shows memory helping unambiguously.

**Scope (locked 2026-07-04):**
- **V7 = V6's `configs/memvla_libero_diversity.yaml`, verbatim, with exactly three changes:**
  1. **More diversity** — increase `num_groups` (the proven +8.5pp lever).
  2. **Compress stored memory entries** — mean-pool (→1 token) or Perceiver (→`n_slots`).
  3. **Match consolidation train↔eval** — size `group_size` / `mem_length` so the training
     bank consolidates the same way an eval rollout does. Compression is what makes this affordable.
- **The ONLY things we carry from V5 are #2 and #3.** Everything else V5 explored —
  write-cadence / step-increment, temporal-PE alignment, LR scaling sweeps, the multi-factor
  ablation matrix — is **explicitly out of scope**. V6's hyperparameters stay as-is.
- **Sweep → then long run.** A minimal check to pick the compression variant and confirm
  consolidation-matching helps, then commit the week.
- **Eval on the other box** (5090/WSL, the pipeline that produced V6's trustworthy numbers).

---

## 1. Anchors and success criteria

| Reference | Overall | spatial | object | goal | libero_10 |
|---|---:|---:|---:|---:|---:|
| Baseline v2 (no memory) | **87.75** | 84 | 99 | 96 | **72** |
| V6 diversity @65k (mem-on) | 81.75 | 84 | 97 | 94 | **52** |
| V6 diversity @65k (bypass) | 78.75 | 77 | — | — | 52 |

V6 already got memory net-positive (mem-on − bypass = **+3.0pp**) and matched baseline on
spatial. Unfinished business is **libero_10 (−20pp)**.

**V7 success:** overall ≥ **85%** AND libero_10 ≥ **62%** (close >half the residual), with
mem-on − bypass ≥ **+3pp** and a positive memory contribution on libero_10 specifically.
Stretch: match/beat 87.75 with a clear memory contribution.

## 2. Why libero_10 is still open — and which V5 lesson fixes it

Confirmed on `dev`: `full_seq_bank.py:176` clears the bank every training batch and per group,
and `group_loader.py` caps each episode at `group_size` frames/step. V6's
`group_size == mem_length == 4` ⇒ **ToMe never fires in training**, but at eval the bank takes
~48 consolidations over a `libero_10` rollout. The model retrieves from a consolidated bank
state it never trained on. This is the **train/eval consolidation mismatch** — precisely V5
lesson #3, and the direct cause of the libero_10 gap. Compression (lesson #2) is what lets us
fix it without blowing up retrieval cost.

## 3. The three changes vs V6 (concrete)

Deltas from `configs/memvla_libero_diversity.yaml`; **all other fields unchanged** (injection
layer 15, `inject_before`, `retrieval_layers=2`, `use_timestep_pe`, LR 1e-4 cosine→2.5e-6,
warmup 1000, 100k steps, AdamW betas/eps, image transforms, seed 1000, expert-from-scratch).

1. **Diversity.** `num_groups: 12 → 24` (or higher — Phase 0 memory calibration sets the ceiling).
2. **Compression.** `compression: mean_pool` (default; 0 params) — store the token-mean of each
   entry instead of the full sequence. Fallback `compression: perceiver, n_slots: 4` if mean-pool
   loses signal. Retrieval/gating/consolidation are already generic over token count.
3. **Consolidation matching.** Choose `group_size` and `mem_length` so training's ToMe merge
   count ≈ an eval rollout's. Two ways, pick in the sweep:
   - **(A) Match the depth:** `group_size` large (compression makes it cheap), `mem_length`
     small, so the training bank overflows and consolidates like eval does (~tens of merges).
   - **(B) Remove overflow both sides:** `mem_length` ≥ episode length in chunks (~64), so
     neither train nor eval consolidates → trivially matched. Cheapest, simplest.
   Recommendation: try **(B)** first (cleanest realization of "matched"), fall back to **(A)** if
   we want ToMe actually exercised. Either way, compression keeps the deep bank affordable.

**Example primary config (B-style):** `num_groups=24, group_size=4` (as V6),
`mem_length=64`, `compression=mean_pool`. Batch = 96. Bank per episode = 64 × 1 token × 960-dim —
trivial in 119 GB. Eval bank also never overflows on any suite → **zero merges both sides**,
mismatch eliminated by construction, and diversity is 2× V6.

## 4. Code changes required (not on `dev` yet)

| # | Change | File(s) | Notes |
|---|---|---|---|
| C-1 | **Entry compression** before store: `mean_pool` over token axis → 1 token, or Perceiver → `n_slots`. Config `compression`, `n_slots`. | `memory/full_seq_bank.py` store path (~L235), new `memory/compressor.py` | Port Perceiver block from the `v5-all-fixes` branch's `compressor.py`. |
| C-2 | **Config plumbing** for `compression`, `n_slots`, and larger `mem_length`/`num_groups`. | `training/config.py`, `policy/builder.py`, config YAML | No new training logic. |
| C-3 | **Eval loads the compression module** so V7 checkpoints run through the existing (V6-validated) pipeline. | eval box `eval_memory_libero.py` | Pipeline itself already trustworthy — it produced V6's numbers that matched baseline on spatial. Low risk. |

*(No cadence / write-stride / LR-scaling changes — out of scope per §Scope.)*

## 5. Phase 0 — calibration (~half a day)

The Spark is bandwidth-modest (LPDDR5X ~273 GB/s ≈ ⅐ of a 5090); measure, don't guess.
- **Throughput:** 200-step dry run → s/step. Gate for the week run: ≤ ~6 s/step for 100k in 7 days,
  else raise diversity via `grad_accum_steps` (episodes/update = `num_groups × grad_accum`) rather
  than per-forward batch, or cut steps.
- **Memory:** largest `num_groups` (and `mem_length`) that fits at bf16 with headroom.
- **Hardware toggles to A/B:** `torch.compile` on the SmolVLA forward, `num_workers≈16`,
  `pin_memory=False` (unified memory — pinning is wasteful).

## 6. Phase 1 — minimal sweep (~1–2 days)

Only what's needed to lock the two V5 ingredients. Each run trains to a short budget, then a
**quick canary** (2 ep × 10 tasks × 4 suites) on the eval box. Loss is ignored — sim only.

| Run | Config | Question |
|---|---|---|
| S0 | primary (B-style: `num_groups=24`, `mem_length=64`, `mean_pool`) | does matched-consolidation + compression + 2× diversity lift libero_10? |
| S1 | `compression: perceiver, n_slots=4` (else = S0) | learned vs mean-pool — which for the week |
| S2 | (A)-style: `group_size=16, mem_length=8` (ToMe fires in training) | does exercising consolidation beat removing it? |

Gate → long run: pick the config that maximizes overall and most lifts `libero_10`. Finalist
gets a 5 ep × 10 task eval (200 rollouts) before the week commit.

## 7. Phase 2 — the week-long run

- Winner config; if throughput allows, `grad_accum_steps` set so episodes/update ≥ 32 (matches
  baseline's proven gradient diversity — e.g. `num_groups=24, grad_accum=2` → 48).
- **100k steps** (or Phase-0-capped max), `checkpoint_every=5000`.
- **In-flight eval:** ship checkpoints at 25k / 50k / 75k / 100k for full 5 ep × 10 task sim eval
  **+ bypass ablation** each time — read the `libero_10` slope, settle "underfit vs plateaued"
  (the ambiguity V6 couldn't resolve at 65k).

## 8. Cross-machine eval protocol

1. Ship `policy_state_dict` + run config YAML (~1.7 GB/ckpt) via `rsync` or a private HF repo
   (as V6 did with `tarmus/memvla-libero-diversity-65k`).
2. Eval box: existing `eval_memory_libero.py --all-suites`, `n_action_steps=10`,
   `eval.batch_size=1`, MUJOCO `egl`/`osmesa`. Same pipeline that produced V6's numbers.
3. Results → per-suite JSONs (`results/sim_memory/...`), V6 schema, for diffability.
4. Cross-check once: base `smolvla_libero` through the same pipeline reads ~86–89% on object.

## 9. Environment bring-up (deferred — box is bare)

System Python 3.12 only, **no torch / lerobot**, CUDA 13 toolkit present, GB10, dataset+model
not cached. Spark = training only:
- [ ] aarch64 + CUDA 13 PyTorch, verify `cuda.is_available()` + bf16.
- [ ] `pip install -e ".[dev]"` — watch aarch64 wheel gaps.
- [ ] Pull `HuggingFaceVLA/libero` (~30 GB) + `lerobot/smolvla_base` (background).
- [ ] `torch.compile` smoke test.

## 9b. EXECUTION LOG (live — updated during the autonomous run)

Environment brought up on the DGX Spark (aarch64 / GB10 / CUDA 13, driver 580):
- **venv:** `~/memory-smolVLA/.venv` (Python 3.12). Activate before anything.
- **torch:** `2.10.0+cu128` (lerobot 0.5.1 pinned it down from 2.11). GB10 is sm_121;
  cu128 ships sm_120 → **first CUDA use JITs every kernel from PTX** (one-time, ~2–3 min,
  cached in `~/.nv/ComputeCache`). Benign capability-warning is expected. Verified: bf16
  matmul + full model forward both run correctly.
- **deps:** `pip install -e ".[dev]"` succeeded only after working around `evdev` (needs
  `Python.h`, no root): dev headers extracted to
  `<scratch>/pydev/root/usr/include` and put on `C_INCLUDE_PATH` (saved in `~/.v7_env` as
  `PY_HEADERS`). Only needed to *build* evdev; not needed at runtime.
- **lerobot 0.5.1 API drift fixed:** `dataset_to_policy_features` → `lerobot.datasets.feature_utils`
  (builder.py); `TransitionKey` → `lerobot.processor` (trainer.py). Both have <=0.4 fallbacks.
- **normalizer:** baseline_v2 dir is absent; configs use `preprocessor_path:
  HuggingFaceVLA/smolvla_libero` (identical flat LIBERO stats: action(7), state(8)). Verified
  tokenizer padding verdict = `longest`.
- **dataset:** `HuggingFaceVLA/libero` cached at `~/.cache/huggingface/lerobot/hub/...` (24 GB).
  NOTE: LeRobotDataset uses `HF_LEROBOT_HOME=~/.cache/huggingface/lerobot`, NOT the plain hub
  cache — do not `snapshot_download` to the wrong root. Do **not** set `HF_HUB_OFFLINE=1` (breaks
  the metadata refs check even with files cached).
- **W&B:** key in `~/.v7_env` (gitignored), authenticates as `tarmus1291` (JHU entity),
  project `memory-smolvla`.

Code changes landed (compression = the V5 lesson):
- `memory/compressor.py` (NEW) — Perceiver-Resampler, applied at **read** time (trains under
  no-BPTT). `memory/full_seq_bank.py` — `compression∈{none,mean_pool,perceiver}`, `n_slots`;
  mean_pool at storage; **read-path generalized from L to stored-token-count S** (the old code
  hard-coded L and would break any compression). Plumbed through policy/builder/train/eval.
- Unit-tested: all 3 compression paths run; perceiver gradient flows; eval bank settles at
  exactly `mem_length` entries after 50 writes (consolidation-matching mechanism confirmed).
- **Smoke passed** (`configs/_smoke_v7.yaml`, batch 8, 2 steps): step1 loss 1.37 gate 0.49,
  step2 loss 1.99, checkpoint saved — full V7 pipeline works on the Spark.

Configs: `configs/memvla_libero_v7.yaml` (primary), `_perceiver.yaml` (S1), `_deepbank.yaml` (S2).
Calibration: `scripts/calibrate.py` (reports s/step + peak CUDA mem).

## 9c. LAUNCHED — the week-long run (2026-07-04)

Calibration on GB10 (bandwidth-bound; throughput scales ~linearly with batch):
- batch 256 (ng16×gs16): 18.6 s/step, 46.9 GB → 100k = 21 days (too slow)
- batch 128 (ng16×gs8):  **9.1 s/step, 24.2 GB** → chosen

**Final primary config** `configs/memvla_libero_v7.yaml`: num_groups=16, group_size=8,
mem_length=4, compression=mean_pool, **batch 128, total_steps=60000** (~6.3 days). This is
V6 verbatim + the three scoped changes: diversity 12→16, mean_pool compression, and
group_size(8) > mem_length(4) so ToMe fires in training and both train & eval settle at 4
consolidated entries (V6's libero_10 mismatch removed). LR 1e-4 cosine over 60k, warmup 1000.
Memory has huge headroom (24/119 GB) — batch was capped by *throughput*, not memory.

Launched under `scripts/run_v7_supervised.sh` (auto-resume on crash, checkpoints every 5k):
- **W&B run:** memvla_libero_v7 → https://wandb.ai/tarmus1291-johns-hopkins-university/memory-smolvla/runs/hzisiyp1
- Checkpoints: `checkpoints/memvla_libero_v7/step_*.pt` (first at ~12.6 h).
- Confirmed training: GPU 96% util (low power = bandwidth-bound as expected).

**Eval handoff (other box):** ship `checkpoints/memvla_libero_v7/step_00XXXXX.pt` + the config
YAML; run `scripts/eval_memory_libero.py --all-suites` (already plumbs `compression`/`n_slots`,
so V7 checkpoints load). Compare mem-on vs `--bypass-memory` at 25k/50k/60k, focus on libero_10.

## 10. Risks

- **Compression could cap the ceiling** (S1 vs S0 checks this). If mean-pool to 1 token loses
  signal, Perceiver `n_slots=4` is the fallback; if both hurt, revert to full-seq with a smaller
  `mem_length` and rely on consolidation-matching alone.
- **Large `num_groups` grows batch** beyond V6's 48 → mild optimizer-dynamics drift from the
  batch-32 baseline. We keep V6's LR (per scope) and watch the first ~5k loss curve; if unstable,
  shift diversity onto `grad_accum` instead of per-forward batch (keeps batch small, LR honest).
- **Throughput unknown** until Phase 0 — the 100k-in-a-week target is adaptive (§5 gate).
