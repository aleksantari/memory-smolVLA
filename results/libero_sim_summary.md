# LIBERO Simulation Eval Summary

**Date:** 2026-04-08
**Suite:** libero_object (10 tasks)
**Episodes per task:** 2
**Pipeline:** lerobot's official LiberoEnv via osmesa rendering

## Pipeline setup (after long debug)

After many failed attempts with custom env wrappers, the working
combination is:

- `robosuite==1.4.1`
- `mujoco==3.6.0`
- `libosmesa6` (apt package)
- `MUJOCO_GL=osmesa` (NOT egl — egl crashes with memory corruption in WSL)
- `lerobot.envs.libero.LiberoEnv` (the official lerobot wrapper)
- `lerobot.policies.factory.make_pre_post_processors` for normalization

Critical bugs we fixed before this worked:
1. State format must be `[eef_pos(3), rotvec(3), gripper_qpos(2)]`,
   NOT joint positions
2. Image resolution 256x256 (not 224x224)
3. Use `check_success()` not `is_success()`, returns plain bool
4. Use the lerobot pre/postprocessor pipeline (handles MEAN_STD norm)
5. Gripper action must be clipped to ±1 (binary in training data)
6. Need `policy.reset()` between episodes

## Results

| Model | Type | libero_object Success | Δ vs Base |
|-------|------|----------------------|-----------|
| **smolvla_libero (baseline)** | no memory | **100.0%** (20/20) | — |
| **Layer 8 v3 (residual gate)** | memory | **0.0%** (0/20) | **−100.0%** |

Per-task breakdown for layer 8 v3 — all 10 tasks: 0/2 success.
Gate alpha = 1.0 throughout (residual gate active as designed).

## Interpretation

**The held-out loss eval was misleading.** Layer 8 v3 had the best
held-out loss (0.09224 vs base 0.09251, −0.3%) but completely fails
in simulation (100% → 0%). The 0.3% loss reduction did not reflect
that the memory module was silently breaking the base policy.

**Why the memory module catastrophically degrades performance:**

1. The base `smolvla_libero` is a fully finetuned policy that works
   perfectly out of the box (100% on libero_object).
2. Our memory training only updated ~0.8% of parameters (memory
   modules + projection layer), with the base policy frozen.
3. The residual gate forces memory to be added at every forward
   pass: `fused = current + retrieved`. Even though the projection
   was zero-initialized and learned small values, those small
   perturbations to the frozen base's intermediate features are
   enough to break the action expert downstream.
4. The action expert was never trained to handle these perturbations,
   so any non-zero memory contribution disrupts its learned dynamics.

## Key takeaways

1. **Held-out flow-matching loss is NOT a reliable proxy for real
   task performance** when adding new components to a finetuned
   model. Sim eval is essential.

2. **Memory injection at the VLM level disrupts a frozen action
   expert**, even with conservative additive fusion. The action
   expert's learned dynamics are tightly coupled to the exact VLM
   features it was trained on.

3. **For memory-augmented VLAs to work, training must be joint** —
   either:
   - Unfreeze the action expert during memory training
   - OR train memory on data where the base isn't already optimal
   - OR inject memory AFTER the action expert (not before)

4. **The SO100 results (8% loss reduction with α=0.89) were
   misleading** — they showed memory was learning useful signals
   on a non-finetuned base. But on a finetuned base, the same
   approach fails.

## Comparison to held-out loss results

| Model | Held-out Loss | Δ vs Base | Sim Success | Δ vs Base |
|-------|--------------|-----------|-------------|-----------|
| Base smolvla_libero | 0.09251 | — | 100.0% | — |
| Layer 8 v3 (residual) | 0.09224 | −0.3% | 0.0% | −100.0% |

The loss difference is in noise range (0.3%) but the behavioral
difference is total. **The loss eval is not a substitute for sim
eval.**
