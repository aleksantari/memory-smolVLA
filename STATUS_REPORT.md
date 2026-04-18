# Memory-Augmented SmolVLA — Status Report

**As of:** 2026-04-17
**Supersedes:** [PROJECT_PLAN.md](PROJECT_PLAN.md) (original plan from 2026-03-24; kept for historical reference)

This document replaces the original project plan. All items originally scoped in Phases 1–7 are now built. The project has moved past implementation into experimental iteration and, critically, into a phase defined by a **definitive negative sim result** that reframed what we thought we knew from held-out loss.

---

## 1. Where the project stands at a glance

| Area | Plan status | Reality today |
|------|-------------|---------------|
| Core memory primitives | `bank`, `retrieval`, `gating`, `temporal_pe` | All present, **plus** `compressor`, `multi_scale_bank`, `working_memory`, `write_gate`, and a second gate variant `ResidualGate` |
| Policy integration | `MemorySmolVLAPolicy`, `FeatureExtractor` | Built; now 589 LOC with 4 training modes and 2 memory backends; `inject_before` flag added |
| Data loader | `EpisodeSequentialLoader` | Built; `builder.py`, `dataset_config.py`, and `_video_compat.py` (torchvision 0.26 patch) added alongside |
| Trainer / Eval | Standard loop + LIBERO evaluator | Built (349 LOC trainer); **multiple** eval scripts: `eval.py` (sim), `eval_loss.py` (held-out), `eval_baseline.py`, `eval_memory_libero.py` |
| Configs | 2 (`base.yaml`, `libero_long.yaml`) | **20+** configs covering 3 ablation generations (v1 sigmoid → v2 sigmoid tuned → v3 residual → v4 joint finetune) |
| Tests | 3 test files | **11** test files; new coverage for `compressor`, `gating`, `write_gate`, `working_memory`, `multi_scale_bank`, `episode_loader`, plus an end-to-end `test_integration.py` |
| Baseline reference | Not in original plan | **Established** — see §5 |

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

## 4. Repository layout today

```
memory-smolvla/
├── STATUS_REPORT.md                       ← this file
├── PROJECT_PLAN.md                        ← original plan (superseded)
├── report.md                              ← 2026-03-28 changes report (clawd + aleksantari)
├── CLAUDE.md  README.md  pyproject.toml
├── configs/                               (20 configs: v1/v2/v3/v4 generations, ablations, baselines)
├── src/memory_smolvla/
│   ├── memory/     bank · multi_scale_bank · retrieval · gating · temporal_pe
│   │               compressor · working_memory · write_gate
│   ├── policy/     memory_smolvla · feature_extractor · builder
│   ├── data/       episode_loader · builder · dataset_config · _video_compat
│   ├── training/   trainer · config
│   └── eval/       evaluator
├── scripts/        train.py · eval.py · eval_loss.py · eval_baseline.py
│                   eval_memory_libero.py · analyze_gates.py · eval_baseline_*.sh
├── tests/          11 test files (primitives + integration)
└── results/        ablation_baseline.md, libero_loss_summary.md, libero_sim_summary.md
                    + per-model JSON artefacts for v1/v2/v3/v4
```

## 5. Anchored reference points (do not drift from these)

- **Baseline to beat (sim success):** RTX 5090 v2, 100K steps, `n_action_steps=10`, all 4 LIBERO suites, 100 episodes/suite. Per-suite JSONs in `outputs/libero_baseline_v2_eval_per_suite/`. Overall 87.75%.
- **Eval config all memory runs must match:** `n_action_steps=10`, 10 episodes/task × 10 tasks = 100 episodes/suite, 4 suites, `eval.batch_size=1`, MUJOCO `osmesa` (egl crashes in WSL) **or** `egl` (RTX 5090 native, verified working).
- **Training config that beat the paper:** batch 32, `num_workers=8`, `use_amp=true`, AdamW lr=1e-4, cosine to 2.5e-6 over 100K with 1K warmup, image transforms on.

## 6. Open questions / immediate next experiments

These are *not* yet scheduled — treat as a candidate list, not a plan.

1. **Full v4 training run** (currently smoke at 30K / 6.7% object). Needs full 100K or more, and full 4-suite eval, to know whether joint finetuning actually recovers memory benefit.
2. **Alternative to v4**: inject memory **after** the action expert rather than before, so expert dynamics aren't perturbed. This was listed as an option in the v3 post-mortem but never attempted.
3. **Memory on the non-finetuned base** (i.e., training memory + action expert jointly from scratch on LIBERO, rather than memory-on-top-of-finetuned). The v2 baseline gives us the reference numbers to compare against.
4. **Is loss eval ever trustworthy here?** v3 taught us closed-loop sim is required. But can we find a proxy that actually correlates with sim success on a finetuned base? (e.g., loss on trajectories where memory provides disambiguation.)
5. **Does `libero_10` respond differently to memory than the other suites?** The v2 baseline's +15pp delta on libero_10 vs paper is the biggest suite-level win; long-horizon is exactly the regime memory should help most with. Worth a dedicated ablation once a non-collapsing, non-disruptive memory config exists.

## 7. What changed vs the original PROJECT_PLAN.md

- **Scope expanded:** compressor, write gate, working memory, multi-scale bank, residual gate — none of these were in the original plan. All were added in response to specific failure modes.
- **Training modes added:** original plan assumed one mode (memory-only, frozen base). Today there are four: `memory_only`, `expert_scratch`, `expert_finetune`, `expert_only_scratch`.
- **Evaluation reframed:** original plan listed one `evaluator.py`. Today we run both held-out loss (`eval_loss.py`) **and** closed-loop sim (`eval.py`), and we now know the sim numbers are the only ones that count.
- **Datasets expanded:** original plan had LIBERO only; SO100 was used for initial validation and produced the cleanest "memory works" signal (−8% loss).
- **New dependency on pipeline correctness:** discovering that state must be `[eef_pos(3), rotvec(3), gripper_qpos(2)]` (not joint positions), that images are 256×256 (not 224), that the policy needs LeRobot's pre/postprocessors, and that `MUJOCO_GL=osmesa` works but `egl` crashes in WSL — none of this was anticipated in the plan.
