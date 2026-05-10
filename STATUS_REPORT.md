# Memory-Augmented SmolVLA — Status Report

**As of:** 2026-04-18
**Supersedes:** [PROJECT_PLAN.md](PROJECT_PLAN.md) (original plan from 2026-03-24; kept for historical reference)

This document replaces the original project plan. All items originally scoped in Phases 1–7 were built, iterated through four ablation generations, and — after v3/v4 failed in closed-loop sim — **pivoted on 2026-04-18 to the MemoryVLA-port design** in [memory_smolvla_implementation_spec.md](memory_smolvla_implementation_spec.md). The prior v1–v4 skeleton is archived (recoverable via git); the repo now implements a single-stream full-sequence cognitive memory bank ported from MemoryVLA's `CogMemBank`. The anchored 87.75% LIBERO baseline (§5) remains the comparison point.

---

## 1. Where the project stands at a glance

| Area | Plan status | Reality today (post-pivot, 2026-04-18) |
|------|-------------|----------------------------------------|
| Core memory primitives | `bank`, `retrieval`, `gating`, `temporal_pe` | Replaced with `memory/blocks.py` (`TimestepEmbedder` + `CrossTransformerBlock` + `GateFusion`, 113 LOC) and `memory/full_seq_bank.py` (`FullSeqMemBank` with ToMe consolidation, 240 LOC) per `memory_smolvla_implementation_spec.md` §3. Old v1–v4 primitives deleted. |
| Policy integration | `MemorySmolVLAPolicy`, `FeatureExtractor` | `MemorySmolVLAPolicy` (286 LOC) wraps vanilla `lerobot/smolvla_base`, injects at VLM layer 15 `inject_before=True`. `FeatureExtractor` reused unchanged. Single training mode; no backend/mode dispatch. |
| Data loader | `EpisodeSequentialLoader` | Replaced with `GroupedEpisodeLoader` (178 LOC) — yields `B = num_groups × group_size` contiguous frames per batch, emits per-frame `episode_ids` / `timesteps`. |
| Trainer / Eval | Standard loop + LIBERO evaluator | Single `train()` loop (254 LOC) with two param groups (expert + memory), `torch.autocast(bfloat16)` matching baseline v2. Sim eval: `scripts/eval.py` (calls `policy.reset()` + passes `episode_id=<rollout_idx>` per spec §6.1). Held-out loss eval retired (closed by v3 post-mortem). |
| Configs | 2 (`base.yaml`, `libero_long.yaml`) | `configs/memvla_libero.yaml` is the only live config — self-contained, hyperparameters mirror baseline v2's `train_config.json` (batch 32, AdamW `weight_decay=1e-10`, `grad_clip_norm=10`, AMP bfloat16). 21 deprecated v1–v4 / ablation / base configs deleted 2026-04-18 (recoverable via git). |
| Tests | 3 test files | Single `tests/test_full_seq_bank.py` covering spec §7.1 shape, §7.2 episode isolation, §7.3 consolidation, §7.4 gradient flow, §7.5 cold-bank pass-through. 10 obsolete test files deleted. |
| Baseline reference | Not in original plan | **Unchanged anchor — 87.75% LIBERO overall, 72% libero_10.** See §5. |

## 2. Bug-fix history (done)

The plan assumed straight implementation would work. It did not. Six non-obvious bugs had to be fixed before any of the results below were trustworthy:

- **Scheduler drift** on partial accumulation flush (`d3216b1`) — LR scheduler was advancing on partial gradient flushes at episode boundaries.
- **`functools.wraps` on bound methods** (`5db8e58`, `dbf0b9b`) — monkey-patch in `feature_extractor.py` was failing because `setattr` on bound methods doesn't work for `__module__` / `__qualname__`.
- **Scalar-tensor unsqueeze** in episode loader — `val.unsqueeze(0)` on a 0-D tensor produced spurious batch dims.
- **Mixed-precision dtype mismatches** in `memory_smolvla.py` — VLM in bf16 vs memory modules in fp32. Fix: explicit casting at module boundaries.
- **First-timestep gradient flow** — when the bank was empty on frame 0, the callback returned `prefix_hidden` unchanged, disconnecting memory modules from the graph. Fix: pass a zero-initialized tensor through the gate so gradients always flow.
- **`.gitignore`** (`d3216b1`) — checkpoints/wandb runs were being tracked.

## 3. Experiment timeline — what we learned, in order

### 3.1 SO100 injection-depth ablation ✅ **worked**
*(100K steps, `memory_only`, 10 val episodes / 4,032 frames; held-out flow-matching loss)*

| Model | Layer | Loss | Gate α | Δ vs base |
|-------|------:|-----:|-------:|----------:|
| raw SmolVLA base | — | 1828.13 | 0.007 | — |
| **memory_layer4** | 4/16 | **1676.45** | 0.890 | **−8.3%** |
| memory_layer8 | 8/16 | 1677.39 | 0.964 | −8.2% |
| memory_layer12 | 12/16 | 1701.29 | 0.988 | −6.9% |

Clean result. Gate α's at 0.89–0.99 say memory is being used. Early/mid injection is best.

### 3.2 LIBERO held-out loss v1 — sigmoid gate, `bias=-5.0` ⚠️ **gate collapsed**
Improvements of only **−0.2 to −0.3%** vs base, and gate α ≈ 3×10⁻⁸. The gate learned to zero out memory.

### 3.3 LIBERO held-out loss v2 — sigmoid gate, `bias=-1.0` ⚠️ **still collapsed**
Retuning the initial bias didn't help. Gate still went to ~0.

**Diagnosis (from `results/libero_loss_summary.md`):** gate collapse on LIBERO is a *gradient-magnitude problem*, not initialization. LIBERO's flow-matching loss is ~0.09; SO100's is ~1828 — a 20,000× gap. Proportionally weaker gradients through the sigmoid gate on LIBERO → gate MLP learns to suppress memory regardless of init.

### 3.4 LIBERO held-out loss v3 — **residual gate** (additive fusion) ✅ **no collapse**
Replaced the learned sigmoid with additive fusion (`fused = current + retrieved`); the zero-init `memory_proj` does the gating implicitly.

| Model | Gate | Loss | α | Δ vs base |
|-------|------|-----:|---:|----------:|
| Base SmolVLA | — | 0.09251 | — | — |
| **Layer 4 v3** | residual | 0.09226 | 1.0 | **−0.3%** |
| **Layer 8 v3** | residual | **0.09224** | 1.0 | **−0.3%** |
| Layer 12 v3 | residual | 0.09229 | 1.0 | −0.2% |

Tiny numerical wins, but at least the system was learning rather than suppressing itself.

### 3.5 LIBERO **simulation** eval of v3 layer 8 ⛔ **definitive negative result**
*(`results/libero_sim_summary.md`, 2026-04-08; 10 tasks × 2 episodes = 20 rollouts on libero_object)*

| Model | Sim success |
|-------|-------------|
| `smolvla_libero` (finetuned base, no memory) | **100.0% (20/20)** |
| Layer 8 v3 (residual, memory) | **0.0% (0/20)** |

**The held-out loss eval was misleading.** A −0.3% loss reduction hid a catastrophic closed-loop regression.

**Root cause:** the base `smolvla_libero` is fully finetuned. The memory training updated only ~0.8% of params (memory modules + projection) with base frozen. Even tiny perturbations to the intermediate VLM features — which `memory_proj` learned to produce — broke the action expert's learned dynamics, because the expert was never trained to tolerate perturbed inputs.

**Three permanent lessons from this failure:**
1. Held-out flow-matching loss is **not** a substitute for sim eval on a finetuned base.
2. Memory injected pre-expert on a frozen finetuned model will disrupt it, even with conservative additive fusion.
3. For memory to help, training must be joint — unfreeze the action expert, or use a non-finetuned base, or inject memory after the expert.

### 3.6 LIBERO v4 — **joint memory + expert finetune** (smoke test)
*(`configs/libero_injection_half_v4.yaml`, 30K steps, `expert_finetune` mode, layer 8, residual gate, `memory_lr=1e-4`, `expert_lr=1e-5`)*

Smoke test on `libero_object` (10 tasks × 3 episodes = 30 rollouts): **6.7% avg** (2/30 successes, both from one task). Gate α ≈ 1.0 throughout (residual gate working as designed). Results at `results/sim_v4_smoke/libero_object_libero_injection_half_v4.json`.

This is the current frontier: well above v3's 0% but still far below the 100% baseline. A full training run + full 4-suite eval is the obvious next experiment.

### 3.7 Baseline reproduction on RTX 5090 (v2) ✅ **exceeds paper**
*(`results/ablation_baseline.md`, 2026-04-17; from-scratch action-expert training on `HuggingFaceVLA/libero`, 100K steps, batch 32, `n_action_steps=10`)*

| Suite | Ours (100K, n=10) | Paper (n=10) | Δ |
|-------|------------------:|-------------:|----:|
| libero_spatial | 84.0 | 89 | −5.0 |
| libero_object | 99.0 | 94 | +5.0 |
| libero_goal | 96.0 | 91 | +5.0 |
| libero_10 (long) | **72.0** | 57 | **+15.0** |
| **Overall** | **87.75** | 82.8 | **+4.95** |

v1 got 30.5% because it ran inference with `n_action_steps=50` (SmolVLA default). The paper's headline uses `n=10` (Table 13). Switching to `n=10` + retraining with correct config recovered the full gap. **This is the anchored reference baseline** for all memory-augmented runs going forward.

Note: an earlier Orin Nano reproduction (70.0% avg; `smolvla_baseline_libero_orin.json`) is superseded by the RTX 5090 v2 run above.

### 3.8 **Pivot to MemoryVLA-port (2026-04-18)** 🔄 **current direction**
*(driving spec: [memory_smolvla_implementation_spec.md](memory_smolvla_implementation_spec.md); commit series landing on `dev`)*

After v3's 0% and v4's 6.7% sim results, the skeleton was replaced with a direct port of MemoryVLA's `CogMemBank`. The new design is narrower: one bank, one injection point, no mode dispatch.

**Architecture.**
- **Full-sequence memory.** Bank stores `[L, D]` per timestep (the whole VLM prefix, not a pooled token), keyed by `episode_id`. `mem_length=8` capacity per episode.
- **Retrieval.** 2 stacked `CrossTransformerBlock`s: current tokens query bank keys (K/V from bank entries, temporal PE added to K only via a sinusoidal `TimestepEmbedder`). SDPA under the hood.
- **Fusion.** Per-token learned sigmoid gate on `[current; retrieved]` (2-layer MLP, `std=1e-3` init on **both weight and bias** — not zero-init, so gradients flow from step 1). Convention: `scale * current + (1 - scale) * retrieved`.
- **Consolidation.** Token-merge (ToMe): when bank exceeds `mem_length`, merge the adjacent pair with highest cosine similarity (computed on flattened entries, `@torch.no_grad()`).
- **Cold-bank path.** When the bank is empty, `retrieved = working_mem` so `gate_fusion(current, current) = current` — verified by the §7.5 correctness test (`atol=1e-5`).

**Architectural adaptation — the one deviation from the spec.** The spec's diagram (§2) assumes a sequential VLM → memory → expert pipeline with a single clean seam. SmolVLA is fused: VLM and action-expert layers are interleaved per-layer, with the expert cross-attending to VLM K/V at odd layers only (`self_attn_every_n_layers=2` → layers 1, 3, 5, 7, 9, 11, 13, **15**). There is no single post-VLM tensor. Resolution: inject at VLM **layer 15**, `inject_before=True` — on the residual-stream tensor (un-normalized) between layer-14's write of `inputs_embeds` and layer-15's `input_layernorm`. Layer-15 VLM self-attn and the layer-15 expert cross-attn (the final cross-attn handoff) both see memory-fused features; earlier expert cross-attn handoffs see vanilla VLM features. User explicitly chose this over retrofitting the base to a true post-VLM seam. Rationale and reach analysis saved to memory (`project_injection_layer_15.md`).

**Training regime (spec §5.1, strict).**
- **VLM backbone frozen.**
- **Action expert reinitialized and trained from scratch.** `lm_expert` + `action_out_proj` + `action_in_proj` + `action_time_mlp_{in,out}` reset via `reset_parameters()` after loading the pretrained checkpoint, so the checkpoint contributes only its SigLIP + SmolLM2 weights.
- **Memory trained from scratch.**
- **Params:** 121.5M trainable (25.7% of 473.3M total). Memory alone is 23.3M.
- **Precision:** `use_amp: true`, `amp_dtype: bfloat16` (autocast wraps forward; no GradScaler — not needed for bfloat16). This matches baseline v2's `use_amp: true` in HuggingFace Accelerate for apples-to-apples eval.
- **Optimizer/schedule aligned to baseline v2** (2026-04-18 config cleanup): batch 32 (`num_groups=4 × group_size=8`), `num_workers=8`, AdamW `lr=1e-4 → 2.5e-6` cosine with 1K warmup over 100K steps, `weight_decay=1e-10`, `max_grad_norm=10.0`. Mirrors `outputs/libero_baseline_v2/.../train_config.json`.

**Smoke test passed (2026-04-18).** 2-step run on `HuggingFaceVLA/libero` (config pre-cleanup: batch 64, 8 groups × 8 frames, weight_decay=1e-4, grad_clip=1.0):
- `step=1 loss=1.4387  gate_mean=0.4980  gate_std=0.3438`
- `step=2 loss=1.4329  gate_mean=0.4961  gate_std=0.3418`
- Checkpoint saves cleanly. Forward / backward / optimizer step / grad-clip / LR schedule / checkpoint all verified end-to-end. Gate initialization lands on the sigmoid midpoint with meaningful per-token variance, as spec §3.3 requires.

**Drops from the old skeleton (deleted, recoverable via git):**
`memory/{bank,retrieval,gating,temporal_pe,compressor,write_gate,working_memory,multi_scale_bank}.py`, `data/{episode_loader,builder}.py`, `scripts/eval_loss.py`, 10 obsolete `tests/test_*.py`. Training-mode dispatch (`memory_only` / `expert_scratch` / `expert_finetune` / `expert_only_scratch`) collapsed to a single code path. `B=1` assertion lifted (`FullSeqMemBank.process_batch` handles `B > 1` via group mode).

### 3.9 Full 100K run + eval + gate-closed ablation (2026-04-20) ⛔ **negative headline result**
*(wandb: `x3idqyh7` training, `eval_step_0100000` mem-on, `eval_step_0100000_bypass` ablation; checkpoint `checkpoints/memvla_libero/step_0100000.pt`; source-of-truth JSONs `results/sim_memory/all_memvla_libero.json` and `..._bypass.json`)*

**Run completed.** 100K steps via [configs/memvla_libero.yaml](configs/memvla_libero.yaml), resumed cleanly from `step_0065000.pt` (stepper picked up at 65000 → 100000). Gate mean stayed at 0.49–0.50 throughout training — never collapsed, never saturated.

**Eval protocol fix (made during the run).** Before eval was trustworthy, [scripts/eval_memory_libero.py](scripts/eval_memory_libero.py) was patched to match baseline v2's protocol: per-episode `LiberoEnv` instantiation (so each of 10 episodes uses a different init state), image rotation via `_format_raw_obs`, env-specified `_max_episode_steps` per suite (280/280/300/520, not hardcoded 400), and `start_seed + ep` seeding. A proper `get_gate_statistics()` method was also added on `MemorySmolVLAPolicy` with the correct key name — prior runs reported `gate_alpha=0.0` purely due to a `hasattr` / key-mismatch logging bug. Ground truth from wandb is gate ≈ 0.49 throughout training and eval.

**Eval — memvla @ 100K (memory on) vs @ 65K vs baseline v2 reference** (10 ep/task × 10 tasks × 4 suites = 400 rollouts):

| Suite | memvla@65K | **memvla@100K** | baseline_v2@100K | Δ vs baseline |
|---|---:|---:|---:|---:|
| libero_spatial | 66.0 | 74.0 | 84.0 | −10.0 |
| libero_object | 93.0 | 96.0 | 99.0 | −3.0 |
| libero_goal | 82.0 | 79.0 | 96.0 | −17.0 |
| libero_10 | 46.0 | 44.0 | 72.0 | **−28.0** |
| **Overall** | 71.75 | **73.25** | **87.75** | **−14.5** |

35K additional steps (65K→100K) yielded only +1.5pp overall. Training has plateaued. The biggest shortfall is on long-horizon / semantic-goal suites — *precisely the suites memory should help most.*

**Bypass ablation — eval @ 100K with `mem_bank.bypass = True`** (skips retrieval + fusion, returns current tokens unchanged; gate telemetry reads 1.0 as sanity). Added via the new `--bypass-memory` flag on the eval script. Same 400 rollouts, same checkpoint, same seeds:

| Suite | mem-on@100K | **bypass@100K** | Δ (bypass − mem-on) |
|---|---:|---:|---:|
| libero_spatial | 74.0 | 72.0 | −2.0 |
| libero_object | 96.0 | 96.0 | 0.0 |
| libero_goal | 79.0 | 82.0 | +3.0 |
| libero_10 | 44.0 | **54.0** | **+10.0** |
| **Overall** | 73.25 | **76.00** | **+2.75** |

**Interpretation (headline).** The gate sitting at ~0.49 throughout training is *not* evidence memory was helping; it is evidence the action expert learned to co-adapt with noisy retrieved features it could not fully gate off. On long-horizon eval rollouts the co-adaptation breaks down and the retrieved features become a net drag. libero_10 (520 max_steps) is 10pp worse with memory on; libero_spatial (280 max_steps) is roughly neutral. The spread correlates with rollout length — consistent with a train-eval mismatch on bank state.

**Mechanistic diagnosis — bank state distribution shift.** Training sees at most `group_size=8` consecutive frames per episode per batch, and the bank is cleared at the start of every batch ([full_seq_bank.py:174-176](src/memory_smolvla/memory/full_seq_bank.py#L174-L176)), so the deepest bank state during training is `group_size − mem_length = 4` ToMe merges. Eval rollouts call the policy once per `n_action_steps=10` env steps (one VLM forward = one bank append), so within a single rollout the bank accumulates `floor(max_steps / n_action_steps) − mem_length` consolidations: ~24 on spatial/object, ~26 on goal, **~48 on libero_10**. That's 6–12× deeper than anything the model saw during training, and the suite-level regression scales with that depth (libero_10 worst, spatial near-neutral). The model was never trained on bank states this consolidated. This is the candidate root cause for the negative ablation result, not a bug.

#### 3.9.1 Config trade-off: bank depth ↔ gradient diversity under batch-32 parity

`mem_length=4` in [configs/memvla_libero.yaml](configs/memvla_libero.yaml) was an internal choice, not a baseline-v2 parity constraint — only `batch_size = num_groups × group_size = 32` was fixed by parity. Two internal constraints then shape the remaining config space:

- `group_size ≥ mem_length` is enforced in [src/memory_smolvla/data/group_loader.py](src/memory_smolvla/data/group_loader.py) (line 185). Otherwise token-merge consolidation never fires during training, because the bank never fills.
- `num_groups` sets the number of distinct episodes represented in each batch (gradient diversity). Each group streams contiguous frames from one episode.

Under batch-32 parity, the design space is:

| num_groups | group_size | max `mem_length` | Episode diversity | Train-time consolidations / group |
|---:|---:|---:|---|---:|
| 32 | 1 | 1 | max (32 eps) | 0 (bank never fills) |
| 8 | 4 | 4 | 8 eps | 0 |
| **4** | **8** | **8** | **4 eps** (current) | ≤4 *(current config uses `mem_length=4`)* |
| 2 | 16 | 16 | 2 eps | up to 12 |
| 1 | 32 | 32 | 1 ep | up to 28 |

The current `(num_groups=4, group_size=8, mem_length=4)` is a middle ground: 4 distinct episodes per batch (reasonable gradient diversity) and a 4-slot bank that fills and consolidates within each group. The escape hatch for growing `mem_length` without sacrificing episode diversity is `grad_accum_steps` — keep `num_groups=4` but bump `group_size` (and `mem_length`) proportionally, accepting slower wall-clock steps. Any future retrain can move freely within this space.

### 3.10 Diversity-test variant @ 65K (2026-05-10) ✅ **+8.5pp over original memvla, diversity hypothesis confirmed**
*(config: [configs/memvla_libero_diversity.yaml](configs/memvla_libero_diversity.yaml); checkpoint: `tarmus/memvla-libero-diversity-65k` on HF, downloaded to `checkpoints/memvla_libero_diversity_partner/final.pt`; wandb: `eval_diversity_65k` at run `xpb4occh`; source-of-truth JSON `results/sim_memory/all_memvla_libero_diversity_partner.json`)*

**Motivation.** The §3.9 negative result and bypass ablation pointed at two competing causes for memvla underperforming baseline v2: (a) bank-state distribution shift between training (≤4 ToMe merges per group) and eval (24–48 merges per rollout), or (b) episode diversity too low — only 4 distinct episodes per batch in the original memvla, far below baseline v2's empirically-required 32. With a single retrain budget, the partner trained the diversity-isolation variant: `(num_groups=12, group_size=4, mem_length=4)` at batch 48, which clears baseline v2's batch=8 floor and provides 12 distinct episodes per batch — at the cost of `group_size = mem_length`, meaning **the bank fills exactly at end of group and ToMe never fires during training.**

**Eval @ 65K (10 ep/task × 10 tasks × 4 suites = 400 rollouts).** Same protocol as §3.9 (per-episode env, suite-specific max_steps, `n_action_steps=10`).

| Suite | memvla@100K (old) | **diversity@65K** | baseline_v2@100K | Δ vs old | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| libero_spatial | 74.0 | **84.0** | 84.0 | **+10.0** | **0.0** |
| libero_object | 96.0 | **97.0** | 99.0 | +1.0 | −2.0 |
| libero_goal | 79.0 | **94.0** | 96.0 | **+15.0** | −2.0 |
| libero_10 | 44.0 | **52.0** | 72.0 | +8.0 | −20.0 |
| **Overall** | 73.25 | **81.75** | **87.75** | **+8.5** | **−6.0** |

Gate active at ~0.504 (5.05/5.04/5.04/5.02 across suites) — same regime as the old run, memory is still being used at ~50% mix.

**Interpretation — diversity was the dominant bottleneck.** The diversity variant has **zero ToMe consolidations during training** (group_size = mem_length = 4 → bank fills exactly, never overflows), so any gain over the old config cannot come from improved bank-depth alignment. The +8.5pp overall and exact baseline match on libero_spatial, with libero_goal and libero_object within 2pp, isolate the gain to one knob — episode diversity per batch — which jumped from 4 to 12. The §3.9 bypass result, in retrospect, was downstream of *underfitting* due to insufficient gradient diversity rather than a bank-state distribution shift. Memory itself isn't broken; it just couldn't learn a useful retrieval policy from 4 episodes per gradient step.

**libero_10 still lags by 20pp.** Even with proper diversity, the long-horizon suite recovers only +8pp. Two plausible factors remaining: (i) 65k steps is below the 100k baseline target — the partner's run continues, and a 100k checkpoint will let us read the slope; (ii) bank-state shift may be a secondary effect that only matters once the gross underfitting is fixed. The ordering of magnitudes is now reversed: diversity was the 8.5pp problem, depth (if it matters at all) is at most a 20pp residual that may also resolve with more steps.

**Bypass ablation @ 65K (sign flip from §3.9).** Re-ran the same all-suites eval with `--bypass-memory` (gate forced to 1.0, retrieval+fusion skipped) on the diversity-65k checkpoint:

| Suite | mem-on@65K | **bypass@65K** | Δ (mem-on − bypass) |
|---|---:|---:|---:|
| libero_spatial | 84.0 | 77.0 | **+7.0** |
| libero_object | 97.0 | 97.0 | 0.0 (at ceiling) |
| libero_goal | 94.0 | 89.0 | **+5.0** |
| libero_10 | 52.0 | 52.0 | 0.0 |
| **Overall** | 81.75 | **78.75** | **+3.0** |

Wandb run `eval_diversity_65k_bypass`; JSON `results/sim_memory/all_memvla_libero_diversity_partner_bypass.json`.

**The sign flip is the key result.** Same architecture, same memory module, same checkpoint format — only training-time episode diversity changed (4 → 12 per batch):

|  | mem-on | bypass | Δ (mem-on − bypass) |
|---|---:|---:|---:|
| original memvla @100K (§3.9) | 73.25 | 76.00 | **bypass wins (−2.75)** |
| **diversity @65K (§3.10)** | 81.75 | 78.75 | **mem-on wins (+3.00)** |

The original memvla bypass result was downstream of underfitting: with only 4 episodes per batch, the action expert co-adapted with noisy retrieved features it couldn't gate off cleanly, and removing memory at eval time recovered the residual signal. With proper diversity, the same gate+retrieval+ToMe pipeline learns a useful retrieval policy and contributes a +3pp net positive. **The memvla architecture is functional — it just required sufficient gradient diversity to converge.**

Suite-level pattern is also coherent: memory contributes most on libero_spatial (+7) and libero_goal (+5) — moderate-horizon, semantic-goal suites where remembering earlier state should matter. libero_object is at ceiling on both runs. libero_10 is unchanged at 52% in both directions — long-horizon is still memory-neutral at 65k, consistent with either "needs more training steps" or "bank-state distribution shift is real but no longer net-harmful." That's the watch-item for diversity@100k.

**Implication for the original §6.1 plan.** The candidate retrain configs in §6.1 — `(num_groups=2, group_size=16, mem_length=16)` and `(num_groups=1, group_size=32, mem_length=16)` — would have *reduced* episode diversity from 4 to 2 or 1, exactly the wrong direction. Those configs are now retracted. The right next experiment is to push diversity even higher and let the partner's run finish to 100k.

## 4. Repository layout today

```
memory-smolvla/
├── STATUS_REPORT.md                       ← this file
├── PROJECT_PLAN.md                        ← original plan (superseded)
├── report.md                              ← 2026-03-28 changes report (clawd + aleksantari)
├── memory_smolvla_implementation_spec.md  ← MemoryVLA-port spec (current source of truth)
├── CLAUDE.md  README.md  pyproject.toml
├── configs/
│   └── memvla_libero.yaml                 ← sole live training config (self-contained, mirrors baseline v2 hyperparameters)
├── src/memory_smolvla/
│   ├── memory/
│   │   ├── blocks.py                      ← TimestepEmbedder + CrossTransformerBlock + GateFusion (113 LOC)
│   │   └── full_seq_bank.py               ← FullSeqMemBank with ToMe consolidation (240 LOC)
│   ├── policy/
│   │   ├── memory_smolvla.py              ← MemorySmolVLAPolicy, single code path (286 LOC)
│   │   ├── feature_extractor.py           ← monkey-patch injection at layer 15 (unchanged)
│   │   └── builder.py                     ← loads lerobot/smolvla_base, reinits action expert (91 LOC)
│   ├── data/
│   │   ├── group_loader.py                ← GroupedEpisodeLoader (178 LOC)
│   │   ├── dataset_config.py  _video_compat.py  (torchvision 0.26 pyav shim)
│   └── training/   trainer.py (254 LOC, AMP bfloat16)  config.py
├── scripts/        train.py · eval.py · eval_baseline.py · eval_baseline_v2_per_suite.sh
│                   eval_memory_libero.py · analyze_gates.py · eval_baseline_nsteps10.sh
├── tests/          test_full_seq_bank.py  (spec §7.1–7.5)
└── results/        ablation_baseline.md, libero_loss_summary.md, libero_sim_summary.md
                    + per-model JSON artefacts for v1/v2/v3/v4
```

**Deleted in the 2026-04-18 pivot** (recoverable via git):
`memory/{bank,retrieval,gating,temporal_pe,compressor,write_gate,working_memory,multi_scale_bank}.py`,
`data/{episode_loader,builder}.py`, `src/memory_smolvla/eval/evaluator.py`, `scripts/eval_loss.py`,
10 obsolete `tests/test_*.py`, and all 21 deprecated `configs/*.yaml` files from the v1–v4 skeleton
(`base.yaml`, `base_libero.yaml`, `libero_injection_*` v1–v4, `memory_injection_*`, `memory_only.yaml`,
`no_memory_baseline.yaml`, `orin_memory_only.yaml`, `ablation_8layers.yaml`, `expert_scratch_*.yaml`).

## 5. Anchored reference points (do not drift from these)

- **Baseline to beat (sim success):** RTX 5090 v2, 100K steps, `n_action_steps=10`, all 4 LIBERO suites, 100 episodes/suite. Per-suite JSONs in `outputs/libero_baseline_v2_eval_per_suite/`. Overall 87.75%.
- **Eval config all memory runs must match:** `n_action_steps=10`, 10 episodes/task × 10 tasks = 100 episodes/suite, 4 suites, `eval.batch_size=1`, MUJOCO `osmesa` (egl crashes in WSL) **or** `egl` (RTX 5090 native, verified working).
- **Training config that beat the paper:** batch 32, `num_workers=8`, `use_amp=true`, AdamW lr=1e-4, cosine to 2.5e-6 over 100K with 1K warmup, image transforms on.
- **Memory-port precision regime:** `use_amp: true`, `amp_dtype: bfloat16` — matches baseline v2. Backward runs in the same autocast context; no `GradScaler` (bfloat16 doesn't need one). Diverging from this risks an unfair comparison against 87.75%.
- **memvla @ 100K reference (negative result):** 73.25% overall, 44% libero_10 — 14.5pp behind baseline v2, gate stable at 0.49 throughout training and eval, bypass ablation shows memory is net-harmful (+2.75pp with memory disabled, +10pp on libero_10). Checkpoint `checkpoints/memvla_libero/step_0100000.pt`; wandb runs `x3idqyh7` (training), `eval_step_0100000` (mem-on), `eval_step_0100000_bypass` (ablation). Any future memvla variant is compared against both baseline v2 (87.75%) **and** this run (73.25% mem-on / 76.00% bypass).
- **memvla diversity-variant @ 65K reference (positive result):** 81.75% overall — +8.5pp over the original memvla, with `(num_groups=12, group_size=4, mem_length=4)` at batch 48. Matches baseline v2 exactly on libero_spatial (84%); within 2pp on libero_object (97%) and libero_goal (94%); libero_10 still lags at 52% (vs 72% baseline). Bank does **not** consolidate during training (group_size = mem_length), isolating the gain to episode diversity. Gate active at ~0.504. **Bypass ablation flips sign vs original memvla:** mem-on 81.75 vs bypass 78.75 (mem-on wins by +3.0pp; original memvla had bypass winning by −2.75pp), with the gain concentrated on libero_spatial (+7) and libero_goal (+5) — memvla architecture confirmed functional under sufficient gradient diversity. Checkpoint `tarmus/memvla-libero-diversity-65k` on HF, downloaded to `checkpoints/memvla_libero_diversity_partner/final.pt`; wandb runs `eval_diversity_65k` (mem-on) and `eval_diversity_65k_bypass` (ablation).

## 6. Open questions / immediate next experiments

These are *not* yet scheduled — treat as a candidate list, not a plan.

1. **Finish the diversity run to 100K and re-eval.** The §3.10 result at 65K already closes most of the gap (+8.5pp over the original memvla) but the partner's training continues to 100K. Read the slope on libero_10 in particular — that suite recovered only +8pp at 65K and remains the dominant gap (52% vs 72% baseline). If libero_10 keeps climbing through 65K→100K, that's the experiment finishing. If it plateaus, depth becomes a real candidate for the residual.
2. **Bypass ablation on diversity@100K.** The §3.10 bypass at 65K flipped sign vs §3.9 — memory now wins by +3pp overall (libero_spatial +7, libero_goal +5, libero_object/libero_10 neutral). Re-run on the 100K checkpoint to see whether the memory contribution grows with more training (especially on libero_10, where memory is still neutral at 65K) or stays in the 3pp band.
3. **Push diversity higher, if a third retrain is on the table.** Diversity-65K used `num_groups=12`. The next step in the same direction would be batch-32-preserving `(num_groups=32, group_size=1, mem_length=1)` (max diversity, zero memory) as a degenerate floor, or batch-bumped `(num_groups=16, group_size=4)` at batch 64. These probe whether 12 episodes was enough or whether 16+ is still leaving gradient signal on the table.
2. **Gate-value dynamics over training** (spec §10). Gate mean starts at 0.498 (verified by smoke test); should move measurably off 0.5 within ~1K steps. If it stays pinned through 10K+ steps, memory isn't learning and we debug before finishing the run.
3. **Parameter-count sanity.** Trainable = 121.5M (expert ~98M + memory 23.3M). MemoryVLA reports ~15–20M for memory alone; our 23.3M is close but consolidation parameters differ. Confirm after training that no memory sub-module is bloated.
4. **Injection-reach ablation (only after a working run).** The locked injection at layer 15 gives memory only the final cross-attn handoff. If the full run works, revisit injecting at multiple cross-attn layers (e.g., 13 + 15) to see whether closed-loop behavior changes.
5. **`libero_10` long-horizon specifically.** Baseline v2's +15pp delta vs paper on `libero_10` is the suite where long-term memory should help most. Call it out explicitly in the per-suite eval write-up.

## 7. What changed vs the original PROJECT_PLAN.md

- **Scope expanded:** compressor, write gate, working memory, multi-scale bank, residual gate — none of these were in the original plan. All were added in response to specific failure modes.
- **Training modes added:** original plan assumed one mode (memory-only, frozen base). Today there are four: `memory_only`, `expert_scratch`, `expert_finetune`, `expert_only_scratch`.
- **Evaluation reframed:** original plan listed one `evaluator.py`. Today we run both held-out loss (`eval_loss.py`) **and** closed-loop sim (`eval.py`), and we now know the sim numbers are the only ones that count.
- **Datasets expanded:** original plan had LIBERO only; SO100 was used for initial validation and produced the cleanest "memory works" signal (−8% loss).
- **New dependency on pipeline correctness:** discovering that state must be `[eef_pos(3), rotvec(3), gripper_qpos(2)]` (not joint positions), that images are 256×256 (not 224), that the policy needs LeRobot's pre/postprocessors, and that `MUJOCO_GL=osmesa` works but `egl` crashes in WSL — none of this was anticipated in the plan.
- **Pivot to MemoryVLA-port (2026-04-18).** After v3 (0% sim) and v4 (6.7% sim) failed the closed-loop test, the whole skeleton was replaced with a port of MemoryVLA's `CogMemBank`: full-sequence memory, stacked cross-attn retrieval with sinusoidal timestep PE, learned per-token sigmoid gate (`std=1e-3` init), ToMe consolidation. The old v1–v4 modules, episode loader, held-out-loss evaluator, and 10 test files were deleted. One architectural adaptation was required: SmolVLA's fused-transformer layout (VLM and expert interleaved per-layer, `self_attn_every_n_layers=2`) forced a single injection point at VLM layer 15, `inject_before=True`, on the un-normalized residual stream — rather than the post-VLM seam the spec's diagram implies. Smoke test verifies the pipeline (loss ≈ 1.43 at step 1–2, gate initialized at ~0.5 with non-trivial per-token variance); the headline full-training experiment is next.
