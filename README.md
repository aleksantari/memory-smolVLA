# Memory-Augmented SmolVLA

Adding a temporal **memory system** to [SmolVLA](https://huggingface.co/papers/2506.01844)
(a compact ~450M-parameter Vision-Language-Action model) for long-horizon robotic manipulation,
evaluated on [LIBERO](https://libero-project.github.io/). The project iterates through a series
of memory designs (V6 → V9), each isolating one architectural idea.

![Architecture: V6 → V7 → V8](docs/arch_v6_v7_v8.png)

## Results (LIBERO success rate, 10 ep × 10 tasks × 4 suites)

| Config | Overall | spatial | object | goal | libero_10 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Baseline v2 (no memory) | 87.75 | 84 | 99 | 96 | 72 | tuned SmolVLA, 100k steps |
| **V6** — diversity | 81.75 | 84 | 97 | 94 | 52 | full-prefix memory |
| **V7** — compression + consolidation-matching | **86.75** | 85 | 99 | 89 | **74** | mean-pool memory · **best trained config** |
| **V7 + action ensembling** | **88.75** | 85 | 99 | 95 | 76 | test-time only, free +2pp |
| **V8** — reasoning tokens + PTP aux | 85.0 | 87 | 99 | 90 | 64 | learned tokens; regressed long-horizon (negative result) |
| **V9** — + truncated BPTT through the bank | *training* | | | | | credit-assignment fix for V8's myopia |

## The idea

The memory module is injected at **VLM layer 15** (on the residual-stream prefix, before its
input-layernorm) and its output is fused back into the frozen VLM's stack:

```
cameras + language + robot state
  → frozen SmolVLM2 (16 layers)
      → layer 15: FullSeqMemBank
          · write current frame → per-episode bank (capacity = mem_length)
          · retrieve: prefix cross-attends bank K/V + sinusoidal timestep PE
          · fuse: learned per-token sigmoid gate  (g·current + (1−g)·retrieved)
          · consolidate: ToMe merges the most-similar adjacent pair when full
  → action expert (width-½ copy, per-layer K/V cross-attn, trained from scratch)
  → flow-matching head → action chunk (predict 50, execute 10)
```

Only the memory bank + action expert are trained; the VLM is frozen. The versions differ **only
in what each frame writes into the bank**:

- **V6** — the full VLM prefix per frame (`group_size == mem_length` ⇒ ToMe never fires in
  training but ~48 merges at eval ⇒ train/eval mismatch). Diversity: `num_groups=12`.
- **V7** — a single **mean-pooled** token per frame, and `group_size 8 > mem_length 4` so ToMe
  fires in training (**consolidation-matching** fixes the mismatch). `num_groups=16`. This is the
  best trained config: `libero_10` 52 → 74.
- **V8** — 8 **learned reasoning tokens** (a Perceiver-Resampler summary head) supervised by a
  PTP auxiliary loss (predict future robot state). Helped spatial/goal but regressed `libero_10`
  — the tokens became *myopic* because bank writes are detached, so only the near-future PTP
  signal shaped them.
- **V9** — **truncated BPTT through the bank**: stored reasoning tokens keep their graph within a
  training group, so a later frame's action loss backprops into the summary head that produced the
  memory it retrieved. The summary head now gets *long-horizon credit* on top of PTP.

Design docs: [`V7_RUN_PLAN.md`](V7_RUN_PLAN.md) · [`V8_PLAN.md`](V8_PLAN.md) ·
[`V9_PLAN.md`](V9_PLAN.md). Interactive diagram: [`docs/arch_v6_v7_v8_viewer.html`](docs/arch_v6_v7_v8_viewer.html).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,libero]"          # libero extra pulls the sim stack for eval
```
Trained on an NVIDIA DGX Spark (GB10, aarch64) with PyTorch cu128.

## Train

```bash
python scripts/train.py --config configs/memvla_libero_v7.yaml   # or _v8 / _v9
# long unattended runs use the auto-resume supervisor:
nohup bash scripts/run_v9_supervised.sh configs/memvla_libero_v9.yaml > .train_v9.log 2>&1 &
```

## Evaluate (LIBERO closed-loop sim)

```bash
MUJOCO_GL=glfw python scripts/eval_memory_libero.py \
    --checkpoint checkpoints/memvla_libero_v7/final.pt \
    --config configs/memvla_libero_v7.yaml \
    --all-suites --n-episodes 10 [--bypass-memory] [--ensemble]
```
`--bypass-memory` disables the memory pathway (ablation control); `--ensemble` enables ACT-style
temporal action ensembling (free +2pp, no retraining).

**Eval note:** lerobot 0.5.1's `LiberoEnv` returns sim frames rotated 180° vs the training data;
`_img_tensor` applies the flip (validated — the base model reads 88% object through this pipeline).

## Layout

- `src/memory_smolvla/` — policy wrapper, `FullSeqMemBank`, reasoning head, grouped loader, trainer.
- `configs/` — `memvla_libero_v{7,8,9}.yaml` (+ sweep/smoke variants).
- `scripts/` — train / eval / calibrate / supervised launchers / diagram generator.
- `results/` — per-suite eval JSONs. `docs/` — architecture diagram + viewer.
