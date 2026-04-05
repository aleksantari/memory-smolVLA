# Changes Report: 3655d3f → 1d2bbe0

**Period:** 2026-03-24 to 2026-03-28  
**Commits:** 8 (5 by Tarik Metin, 3 by aleksantari)

---

## Summary

Your friend (Tarik / "clawd") did the initial round of bug fixes and infrastructure work (Mar 24–25), then aleksantari (you or another collaborator) ran the ablation experiments and LIBERO evaluations (Mar 26–28). The work breaks down into three areas: **bug fixes**, **training/eval infrastructure**, and **experiment results**.

---

## 1. Bug Fixes

### Scheduler drift on partial accumulation flush (`d3216b1`)
When an episode ended mid-gradient-accumulation and the trainer flushed partial gradients, the LR scheduler was being advanced — causing it to drift ahead of the actual training step count. Fix: only advance the scheduler on full accumulation cycles.

### `functools.wraps` on bound methods (`5db8e58`, `dbf0b9b`)
`functools.wraps()` fails on bound methods because `setattr` for `__module__`/`__qualname__` doesn't work. The monkey-patching in `feature_extractor.py` was hitting this. Fix: plain assignment instead of `@wraps`.

### Scalar tensor unsqueeze in episode loader (`5db8e58`)
`val.unsqueeze(0)` was being called on scalar tensors (e.g., `episode_index`), creating spurious batch dimensions. Fix: only unsqueeze tensors with `ndim >= 1`.

### Mixed-precision dtype mismatches (`memory_smolvla.py`)
When the VLM runs in bf16 but memory modules are fp32, dtype mismatches caused failures. Fix: explicit dtype casting at memory module boundaries (cast to compute dtype on entry, back to VLM dtype on exit).

### First-timestep gradient flow (`memory_smolvla.py`)
Previously, when the memory bank was empty (first frame), the callback returned `prefix_hidden` unchanged — meaning memory modules were disconnected from the computation graph and got no gradients. Fix: pass a zero-initialized tensor through the gate even on the first timestep so gradients always flow.

### `.gitignore` added (`d3216b1`)
The repo now properly ignores `__pycache__/`, `.pyc`, checkpoints, wandb runs, etc.

---

## 2. New Infrastructure

### SmolVLA preprocessor pipeline (`1ee7350`)
`train.py` now builds LeRobot's `make_pre_post_processors` pipeline (tokenization, normalization, device movement) and passes it through to the trainer. Also added feature-map key remapping so dataset column names can differ from what the policy expects.

### `_video_compat.py` — torchvision >= 0.26 compatibility
`torchvision.io.VideoReader` was removed in torchvision 0.26. This module monkey-patches `lerobot.datasets.video_utils.decode_video_frames_torchvision` with a pure PyAV fallback.

### `eval_loss.py` — held-out loss evaluation script
New script for comparing flow-matching loss across multiple checkpoints on a validation split. Evaluates models on the same episodes, reports per-episode losses and gate alpha values. Used for the ablation study.

### `eval.py` — major rewrite
Replaced old class-based evaluator with procedural LIBERO evaluation. Now dynamically loads LIBERO task suites, handles image format conversion, tokenization, and robot state assembly inline. Outputs per-task success rates and gate alpha statistics to JSON.

### `inject_before` mode in feature extractor
New parameter allows memory injection *before* a layer's attention (not just after its MLP). Enables injection at the final VLM layer (layer 15) so augmented features enter the KV cache visible to the action expert.

---

## 3. New Configs

| Config | Purpose |
|--------|---------|
| `base_libero.yaml` | Base LIBERO training — LIBERO-finetuned SmolVLA, injection layer 8, 100k steps |
| `libero_injection_half.yaml` | LIBERO ablation: inject at layer 8 (mid) |
| `libero_injection_quarter.yaml` | LIBERO ablation: inject at layer 4 (early) |
| `libero_injection_three_quarter.yaml` | LIBERO ablation: inject at layer 12 (late) |
| `memory_injection_full.yaml` | SVLA ablation: inject at layer 15 (final, `inject_before=true`) |
| `memory_injection_quarter.yaml` | SVLA ablation: inject at layer 4 |
| `memory_injection_three_quarter.yaml` | SVLA ablation: inject at layer 12 |
| `no_memory_baseline.yaml` | No-memory control: `expert_only_scratch` mode |
| `orin_memory_only.yaml` | Orin Nano-optimized: batch=1, 50k steps, 0 workers |

---

## 4. Experiment Results

### Injection Depth Ablation (SVLA dataset)
100k steps, memory_only mode, 10 validation episodes (4,032 frames):

| Model | Injection Layer | Avg Loss | Gate Alpha | vs Baseline |
|-------|----------------|----------|------------|-------------|
| raw_smolvla_base (untrained) | N/A | 1828.13 | 0.007 | — |
| **memory_layer4 (quarter)** | 4/16 | **1676.45** | 0.890 | **-8.3%** |
| memory_layer8 (half) | 8/16 | 1677.39 | 0.964 | -8.2% |
| memory_layer12 (three_quarter) | 12/16 | 1701.29 | 0.988 | -6.9% |

**Takeaways:** All memory models beat raw SmolVLA by 7–8%. Early/mid injection (layers 4–8) slightly outperforms late injection (layer 12). Gate alpha confirms the memory system is being actively used (0.89–0.99 vs 0.007 at init).

### LIBERO Held-Out Loss (338 val episodes, 41,394 frames)
Models trained on LIBERO with injection at different depths:

| Model | Avg Loss |
|-------|----------|
| raw_smolvla_libero (no memory) | 0.09251 |
| memory_layer4 | 0.09230 |
| memory_layer8 | 0.09227 |
| **memory_layer12** | **0.09204** |

Loss differences are much smaller on LIBERO (~0.3–0.5%) compared to SVLA (~8%). Interestingly, layer 12 performs best here — opposite of the SVLA results.

### SmolVLA LIBERO Baseline (Jetson Orin Nano Super)
46-hour eval, 10 episodes per task:

| Suite | Ours | Paper | Delta |
|-------|------|-------|-------|
| libero_spatial | 68.0% | 78.4% | -10.4 |
| libero_object | 92.0% | 92.6% | -0.6 |
| libero_goal | 84.0% | 84.0% | 0.0 |
| libero_10 | 36.0% | 50.6% | -14.6 |
| **Overall** | **70.0%** | **76.4%** | **-6.4** |

Orin reproduction is ~6% below paper numbers overall. Object and goal suites nearly match; spatial and libero_10 lag more significantly.

---

## 5. New Branches

Three new remote branches were fetched:
- `claude/evaluate-vla-memory-SePiG`
- `clawd/bug/fix-scheduler-drift-and-gitignore`
- `clawd/bug/fix-test-failures`
