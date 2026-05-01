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

## Run 0: diagnostic (TEMPLATE — fill in when starting)

**Status:** not started
**Config:** `configs/libero_v5_run0_diagnostic.yaml`
**Seed:** 1000
**Started:** —
**Hardware:** —
**Branch HEAD:** —

### Training

(fill after training completes)

### Quick eval

(fill after eval completes)

### Decision

(fill after deciding next run)

---

## Run 1: kitchen sink (TEMPLATE)

**Status:** not started
**Config:** `configs/libero_v5_run1_kitchen_sink.yaml`
**Seed:** 1000
...

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
