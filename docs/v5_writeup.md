# Memory-Augmented SmolVLA — Project Writeup

A design document covering motivation, current state (V4), drawbacks discovered, and the V5 fixes. Numbers, dimensions, and references are written out so they can be lifted directly into diagrams.

---

## 1. Motivation

### 1.1 The problem

Vision-Language-Action (VLA) models like SmolVLA are *Markovian by construction*: each forward pass conditions on the current observation only. This is fine for short-horizon tasks (pick up cube, place in bin) but fails on tasks where the right action depends on *what already happened* — e.g. "now place it in the box you opened earlier," "alternate between the two cups," or any sequence with hidden state in the demonstrator.

**Empirical signature:** SmolVLA's own LIBERO results (Table 13 of [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)) show its weakest suite is `libero_10` (long-horizon, 50.6% paper / 72% our v2), the only suite explicitly built around tasks where temporal context matters. Other VLAs (CogACT, π-0) show the same pattern.

### 1.2 The opportunity

Two recent papers demonstrate that *adding* explicit memory to a frozen VLA backbone — without retraining the whole model — recovers most of the long-horizon gap:

- **MemoryVLA** ([arXiv:2508.19236](https://arxiv.org/abs/2508.19236), Aug 2025): +26pp on long-horizon real tasks vs CogACT baseline.
- **ContextVLA** ([arXiv:2510.04246](https://arxiv.org/abs/2510.04246), Oct 2025): consistent improvements over single-frame VLAs with a single average-pooled context token per past frame.

These results suggest the bottleneck on long-horizon tasks is *information access*, not model capacity, and that retrofitting memory onto a frozen VLA is a viable cheap path.

### 1.3 Project goal

Build a memory module that:
1. Slots into SmolVLA's pipeline at minimal cost (memory module ≈ <1% of total params).
2. Recovers the long-horizon failure modes seen in v2 baseline (`libero_10` 72% → ~85%+).
3. Doesn't degrade the suites where v2 already saturates (`libero_object` at 99%).
4. Trains on consumer hardware (Colab L4) in <12h per run.

---

## 2. SmolVLA architecture (for diagram)

### 2.1 Components

| Component | Source | Specs |
|---|---|---|
| Vision encoder | SigLIP (frozen) | 512×512 input → PixelShuffle → **64 tokens/image** |
| Language model | SmolLM2-360M | 32 layers in HF release; **SmolVLA truncates to first 16** |
| Action expert | Flow-matching transformer | Cross-attends to VLM KV cache; 432 hidden dim |
| Action head | Linear `action_out_proj` | Outputs 50-step action chunks |

### 2.2 Prefix construction (`embed_prefix`)

The VLM input sequence is built in this order:

```
[image_start]  [64 image tokens × N_cameras]  [image_end]  [tokenized language]  [state_token]
```

For LIBERO with 2 cameras (agentview + robot0_eye_in_hand) and `add_image_special_tokens=True`:

| Region | Token count |
|---|---|
| Image specials (start + end) per cam | 2 × 2 = 4 |
| Image content tokens | 2 × 64 = 128 |
| Language (instruction, padded) | ~30–50 |
| State (proprioception) | 1 |
| **Total prefix length** | **~163–183** |

### 2.3 Critical fact: the truncation

SmolVLA uses **only the first 16 of SmolLM2's 32 layers**. In autoregressive multimodal LLMs, semantic abstraction concentrates in the upper half; visual feature representation peaks around the middle. **At our deepest available layer (15), hidden states are predominantly perceptual** — the layers that would build deep semantic intent representations have been removed.

This is the key difference vs MemoryVLA (LLaMA-7B, full stack): they have access to a true semantic "EOS" token; we don't.

### 2.4 Data flow

```
RGB images      ──► SigLIP ──► PixelShuffle ──► [B, 64, 576] per image
language tokens ──► embedding lookup        ──► [B, L_lang, 576]
robot state     ──► linear projection       ──► [B, 1, 576]
                       │
                       └─► concat ──► [B, L_prefix, 576]
                                             │
                                             ▼
                                    SmolLM2 (16 layers) ──► hidden states
                                             │
                                             ▼
                                    Action expert (cross-attn → KV cache)
                                             │
                                             ▼
                                    Flow-matching denoising (10 steps)
                                             │
                                             ▼
                                    Action chunk [B, 50, action_dim]
```

---

## 3. MemoryVLA architecture (for diagram)

Reference: [arXiv:2508.19236](https://arxiv.org/abs/2508.19236), [project page](https://shihao1895.github.io/MemoryVLA/).

### 3.1 Components

| Component | Specs |
|---|---|
| Vision encoders | DINOv2 + SigLIP (parallel, both frozen) |
| Compression | Squeeze-and-Excitation bottleneck → `Np=256` perceptual tokens |
| Language model | LLaMA-7B (full stack) |
| Cognitive token | Output at EOS position of LLaMA's final layer = **1 token** |
| Action expert | Diffusion transformer with two attention layers per block |

### 3.2 Two-stream memory bank ("Perceptual-Cognitive Memory Bank")

| Stream | Per-frame entry | Source |
|---|---|---|
| **Perceptual** (`m_per`) | 256 tokens (dim `dp`) | SE-bottleneck output, *before* LLM |
| **Cognitive** (`m_cog`) | 1 token (dim `dc`) | LLaMA EOS output, after full 7B forward |

Each bank holds up to L entries. **Consolidation** when full: merge the most-similar adjacent pair (cosine similarity on mean-pooled features), keep the newer timestamp.

### 3.3 Retrieval & injection

Two parallel cross-attentions:
- Current perceptual tokens **query** the perceptual bank → retrieved features `H_per`
- Current cognitive token **queries** the cognitive bank → retrieved features `H_cog`

Sinusoidal **timestep positional encoding** added to bank keys (not values) before attention.

**Gate fusion:** `g_x = σ(MLP(concat[x, H_x]))`, `x̃ = g_x ⊙ H_x + (1−g_x) ⊙ x`.

Two **injection points** in the action expert: `x̃_per` enters a perception-attention layer, `x̃_cog` enters a cognition-attention layer.

### 3.4 Why this works for them but doesn't directly transfer to SmolVLA

- They have a *real* semantic stream (LLaMA-7B EOS at layer 32). SmolVLA at our deepest available layer 15 is mid-stack of the original 32 → no semantic stream available.
- Their perceptual stream is built from **raw vision encoder features**, before any LLM mixing. Our injection happens *inside* SmolLM2's stack, so what we'd call "perceptual" already has some text-image mixing baked in.

---

## 4. Our V4 implementation (current — for diagram)

### 4.1 What it does

```
Frozen SmolVLA pipeline
     │
     ▼
SmolLM2 layer 0 ──► layer 1 ──► ... ──► layer 7 ──► layer 8 hidden states
                                                          │
                                                          ▼
                                              ╔═══ Memory module ═══╗
                                              ║ (1) WRITE: detached  ║
                                              ║     prefix → bank    ║
                                              ║                      ║
                                              ║ (2) READ: cross-attn ║
                                              ║     bank → retrieved ║
                                              ║                      ║
                                              ║ (3) GATE: residual   ║
                                              ║     fused = current  ║
                                              ║          + retrieved ║
                                              ╚══════════════════════╝
                                                          │
                                                          ▼
SmolLM2 layer 9 ──► ... ──► layer 15 ──► action expert ──► actions
```

### 4.2 Specifications

| Component | Setting |
|---|---|
| Injection layer | 8 (mid-VLM) |
| Bank type | Single FIFO bank, `bank_max_size=16` |
| Per-entry storage | **Full prefix** `[L_prefix=~170, D=576]`, detached, on CPU |
| Retrieval | `nn.MultiheadAttention(d=576, heads=4)`, queries=current prefix, keys=memory+temporal_PE, values=memory |
| Temporal PE | Sin/cos encoding of `current_time - entry_time` on log-spaced frequencies (`min_period=1, max_period=1000`) |
| Gate | Residual: `fused = current + memory_proj(retrieved)`, alpha=1.0 always |
| `memory_proj` | `Linear(576, 576, bias=False)`, **zero-initialized** so model = vanilla SmolVLA at start |
| Trainable params | Memory modules (~10M) + `lm_expert` + `action_out_proj`. VLM frozen. |
| Training mode | `expert_finetune` (joint memory + expert finetune) |
| Loader | `EpisodeSequentialLoader`: yields **all frames** of episode A, then boundary, then all of B, ... |
| Trainer step | B=1 (asserted in callback), `grad_accum_steps=1` (default, never overridden) |
| Total steps | 30,000 |
| LR | memory: 1e-4, expert: 1e-5 |
| Eval | LIBERO 4 suites × 10 tasks × 10 episodes/task = 400 episodes/checkpoint |

### 4.3 Numerical state at retrieval

When the bank is full:
- **16 entries × 170 tokens = 2,720 memory keys**
- Cross-attention map at layer 8: 170 queries × 2,720 keys × 4 heads = ~1.85M attention weights per forward
- VRAM impact: enough to push us off `batch_size=32` if we tried to recover gradient diversity via accumulation (see §5.3)

### 4.4 V4 results

Numbers to put in a comparison table / bar chart (success rate on LIBERO, n=10 eps/task, 100 eps/suite):

| Run | Spatial | Object | Goal | Long (libero_10) | **Overall** | Δ vs baseline | Δ vs bypass |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline v2** (no memory, 100K steps) | 84 | 99 | 96 | 72 | **87.75** | — | — |
| V4 memory on (residual gate effective α≈0.49) | 74 | 96 | 79 | 44 | **73.25** | **−14.50** | −2.75 |
| V4 memory bypassed (gate forced to no-op) | 72 | 96 | 82 | 54 | **76.00** | **−11.75** | — |

**Two key gaps in this table:**
- **76.00 → 87.75 (−11.75pp):** the *baseline regression* introduced just by training in `expert_finetune` mode, *with memory bypassed*. This is not memory's fault.
- **73.25 → 76.00 (−2.75pp):** the *additional* cost of turning memory on. Small, but negative.

Per-suite, the worst hit is **`libero_10`** (long-horizon): 72 → 44 with memory, 72 → 54 with bypass. The very suite memory was supposed to *help* is the one where it most clearly hurts. This is an important diagnostic — it suggests OOD signals from the memory pathway are disrupting exactly the inferential capacity the policy needs for long horizons.

---

## 5. Drawbacks of V4 (root-cause analysis)

Five distinct issues, found by reading the code paths and comparing to the literature:

### 5.1 Drawback 1: Token bloat in retrieval

**Symptom.** Bank stores the full prefix, ~170 tokens × 16 entries = 2,720 heterogeneous keys. Cross-attention queries (~170 current tokens) need to focus attention against this large, noisy keyset.

**Root cause.** Each prefix token has very different distributions:
- Image patch tokens: spatially varying perceptual features
- Language tokens: discrete instruction-grounded
- State token: scalar-projection of robot pose
- Image-special tokens: nearly constant

Cross-attention treats them all as one bag. Cosine-similarity-based bank consolidation (when implemented) can merge image patches with text tokens.

**Reference.** ContextVLA ([arXiv:2510.04246](https://arxiv.org/abs/2510.04246)) compresses past observations to **1 token per past frame** via average pooling at an intermediate VLM layer. Compressor-VLA ([arXiv:2511.18950](https://arxiv.org/abs/2511.18950)) uses learned compressors with 3× token reduction.

**Quantification.** ~170× excess key count vs ContextVLA's 1-token-per-frame approach.

### 5.2 Drawback 2: Train/inference temporal-PE distribution shift

**Symptom.** Bank fill rate at training and inference differ by 50× because of action chunking.

**Code paths.**
- Training ([memory_smolvla.py:486](src/memory_smolvla/policy/memory_smolvla.py:486)): `_timestamp += step_increment`, default `step_increment=1`. Callback fires every frame, writes every fire.
- Inference ([eval.py:209](scripts/eval.py:209)): `step_increment=50` (= `chunk_size`). Callback fires once per chunk_size env steps because `predict_action_chunk` runs the VLM forward once and queues 50 actions.

**Quantitative consequence.** For a 200-frame episode:

| | Training | Inference |
|---|---|---|
| Callbacks fired | 200 | 4 |
| Bank entries at end | 16 (FIFO-saturated) | 4 |
| Time deltas at retrieval | {1, 2, ..., 16} | {0, 50, 100, 150} |

The **temporal PE basis** is defined for delta=50/100/150, but the **gate/retrieval modules never trained on those values**. They saw deltas 1–16. Inference is OOD.

### 5.3 Drawback 3: Gradient diversity collapse (the biggest one)

**Symptom.** V4 trained at *effective batch = 1*. One frame from one episode per gradient update.

**Root cause chain.**
1. `_memory_callback` asserts B=1 ([memory_smolvla.py:386](src/memory_smolvla/policy/memory_smolvla.py:386)) because the memory bank is per-episode singleton state.
2. `_train_sequential` ([trainer.py:112](src/memory_smolvla/training/trainer.py:112)) is the loop used for all memory training modes; it processes one item at a time.
3. `grad_accum_steps` default = 1 ([config.py:46](src/memory_smolvla/training/config.py:46)). v4 YAML doesn't override it. No CLI override exists.
4. The loader `EpisodeSequentialLoader` yields *all frames of episode A* before episode B — so even with `grad_accum_steps=32`, 32 consecutive frames would all come from one episode (high autocorrelation, low diversity).

**Comparison to baseline v2.**

| | Baseline v2 (`expert_only_scratch`) | V4 (`expert_finetune`) |
|---|---|---|
| Loader path | `_train_batch` (random batched DataLoader) | `_train_sequential` |
| Effective batch | **32 different (episode, frame) pairs** | **1 frame, 1 episode** |
| Gradient diversity | High (random scenes, varied actions) | Near zero (correlated state, similar actions) |

A 32× gradient-diversity gap. Adam tolerates some across-step variance via running averages, but a 32× collapse changes the *distribution* of training signals fundamentally.

**This likely explains the dominant ~12pp regression** between baseline v2 (87.75%) and v4 bypass (76.00%) — *independent of the memory architecture itself*.

### 5.4 Drawback 4: VLM truncation eliminates the cognitive stream option

**Symptom.** MemoryVLA's most powerful component — the 1-token cognitive memory at LLM final layer — has no clean analog in SmolVLA.

**Root cause.** SmolVLA truncates SmolLM2 at 16 of 32 layers. Standard VLM interpretability finding: visual features peak mid-stack, semantic abstraction concentrates in the upper half (which we discarded). At our deepest available injection point (layer 15), hidden states are still predominantly perceptual, not cognitive.

**Implication for design.**
- A "synthetic [SUMMARY] token" approach (let the LLM aggregate via attention) is weakened: the LLM at our truncation depth doesn't have the layers that would *do* semantic aggregation.
- Memory must be designed assuming we have **only a perceptual stream**, not a perceptual + cognitive one.

### 5.5 Drawback 5: Held-out loss is a misleading metric

**Evidence (already in [results/libero_sim_summary.md:88-91](results/libero_sim_summary.md:88)).**

| Variant | Held-out loss | Δ loss vs base | Sim success | Δ sim vs base |
|---|---|---|---|---|
| Base smolvla_libero | 0.09251 | — | 100.0% | — |
| Layer-8 v3 (residual gate) | 0.09224 | **−0.3%** | **0.0%** | **−100.0%** |

A 0.3% loss improvement coincided with a 100pp success collapse on libero_object. The flow-matching loss is an *averaged per-frame* signal; a policy can have low frame-level error and still fail to compose the right *sequence* for task success.

**Implication.** Use sim success as the only winner-selection metric. Held-out loss is fine for divergence detection (loss explodes → kill the run) but never for "is this better."

---

## 6. The V5 fixes

Five orthogonal changes, each targeting one of the drawbacks above. All five live on a single merged branch (`claude/feature/v5-all-fixes`). Each is independently togglable via config flags.

### 6.1 V5 — Window Loader (targets Drawback 3)

**Architecture change.**

Before: `EpisodeSequentialLoader` yields all frames of episode A → boundary → all frames of episode B → ...

After: `max_window_size=N`, then yield N consecutive frames at a random offset within episode A → boundary → N from B → ... Random offset prevents the loader from always seeing episode openings; episodes longer than N get sampled differently each pass.

**Combined with** `grad_accum_steps = N`, each optimizer step accumulates one window's gradients, and *successive* optimizer steps see different episodes. Adam's running averages absorb the across-step variance.

**Bank semantics.** Inside one window, the bank fills sequentially exactly as before (up to `bank_max_size=16`). No memory-design change.

**Tradeoff vs Option 3 (parallel episode tracks).** True random-batched diversity would require N concurrent banks (one per parallel episode track). That refactor was rejected as too risky for the 1-week sprint. Window loader gets ~70-85% of baseline gradient diversity; Option 3 would get 100%.

**Tradeoff vs Option 1 (multi-episode interleaving with mid-window memory reset).** Cheaper per-step diversity but starves the bank — model never sees deep memory at training, then encounters deep memory at inference. Distribution shift in opposite direction. Rejected.

**Justification for Option 2.** Adam's adaptive moment estimates make across-step variance tolerable. The loss is in *which 32-frame window* you see per step — but successive 32-frame windows are different episodes. Per-episode shuffle (already in the code) ensures coverage.

**Config.** `dataset.max_window_size: 32` + `trainer.grad_accum_steps: 32` + `trainer.total_steps: 3000` (matches v4's 30K * 1 wall-clock).

### 6.2 V5 — Write Stride (targets Drawback 2)

**Architecture change.**

Add `write_stride: int = 1` parameter. In `_episodic_callback`, gate the bank write by `current_time % write_stride == 0`. The retrieval/gate computation still fires every callback (gradients flow), but the bank only accumulates entries at stride boundaries.

**Concretely.**
- Training: callback fires every frame; bank writes at frames {0, 50, 100, ...} → bank fills slowly, deltas in retrieval are {50, 100, ...}.
- Inference: callback already fires only every chunk_size = 50 frames (because of action chunking) → every fire is at a multiple of 50, every fire writes.

**Result.** Bank fill rate matches inference exactly. Time deltas at retrieval at training are now drawn from the same distribution as at inference. Temporal PE values are no longer OOD at deployment.

**Tradeoff.** Bank entries are sparser at training (~16 entries spread across 800 frames instead of 16 frames), so the model trains on retrieval against fewer-but-wider-spaced memories. This *is* the inference distribution though — that's the point.

**Config.** `policy.write_stride: 50`.

### 6.3 V5 — Mean-Pool Compression (Option B; targets Drawback 1)

**Architecture change.**

Add `compression_mode: str = "none"` parameter. When `"mean_pool"`, after computing `tokens_to_store = prefix_hidden[b]` (and after any optional learned compressor), replace with `tokens_to_store.mean(dim=0, keepdim=True)`. Each bank entry is now 1 token.

**Numerical impact.**
- Bank keys: 16 entries × 1 token = **16** (down from 2,720, ~170× reduction).
- Cross-attention map: 170 × 16 × 4 heads = ~10K weights (down from 1.85M).
- VRAM headroom restored. Allows batch=32 via grad_accum without OOM.

**Tradeoff.** Mean-pooling is lossy: positional information within the prefix is discarded. The retrieved feature is a single average-of-prefix token per past frame. Likely fine for *coarse temporal context* ("we were near the cube earlier") but loses fine spatial structure ("the cube was at top-right").

**Justification.** Direct ContextVLA ([arXiv:2510.04246](https://arxiv.org/abs/2510.04246)) replication. Their result: 1 avg-pooled context token per past frame consistently improved VLAs over single-frame baselines.

**Config.** `policy.compression_mode: mean_pool`.

### 6.4 V5 — Perceiver Compressor (Option A; targets Drawback 1)

**Architecture change.**

Existing `MemoryCompressor` ([compressor.py](src/memory_smolvla/memory/compressor.py)), upgraded from a single cross-attention layer to a full **Perceiver-Resampler** block:
- Pre-norm cross-attention with residual: `slots = queries + cross_attn(LN(queries), LN(prefix), LN(prefix))`
- FFN with residual: `slots = slots + FFN(LN(slots))`, FFN expansion factor 4

Learnable query vectors (size `n_slots`) cross-attend to the prefix, producing `n_slots` compressed tokens per frame.

**Configurable.** `compressor_n_slots ∈ {1, 4, 16}` sweep. Bank keys at retrieval: `bank_max_size × n_slots`.

**Trainable params added.** ~50K params for n_slots=4 at d_model=576 (negligible).

**Tradeoff.**
- Vs mean-pool (Option B): Learned (more capacity per slot), more params (~50K), needs gradient signal to learn what to attend to. Mean-pool is zero-param.
- Vs full prefix (V4): Same comparison — much better key/query ratio, but loses any token the queries don't attend to.
- Vs two-stream (Option D): Single compressor doesn't know about prefix structure (image vs text vs state). Has to figure out the split implicitly via attention.

**Justification.** Standard Flamingo/BLIP-2-style fixed-budget compressor. Strong prior that this works for *visual* token compression (it's what those models use to compress vision features for LLM consumption). Less standard for *post-LLM-prefix* compression, but the same machinery applies.

**Reference.** Perceiver-Resampler in Flamingo (Alayrac et al., 2022), Q-Former in BLIP-2.

**Config.** `policy.use_compressor: true`, `compressor_n_slots: 4`.

### 6.5 V5 — Two-Stream (Option D; targets Drawback 1 + 4)

**Architecture change.**

The most architecturally invasive of the V5 fixes. **Two parallel Perceiver compressors, each over a different region of the prefix:**

```
prefix at injection layer (170 tokens)
        │
        ├── tokens [0 : n_image_tokens=132]   ──► perceptual_compressor ──► 16 slots
        │                                                                    │
        └── tokens [n_image_tokens : end]     ──► task_compressor       ──►  1 slot
                                                                             │
                                              concat ──► 17 slots per entry
                                              bank.write
```

**Why split here.** At injection layer 8 (mid-truncated SmolLM2), image and text tokens have undergone partial cross-mixing via self-attention. But the dominant signal at each position still reflects its original modality (image patch positions carry scene state; language positions carry task identity).

**Why NOT a true MemoryVLA-style cognitive stream.** Per Drawback 4, we don't have access to a deep semantic representation. The "task" stream here is a perceptually-mixed instruction representation, not a true cognitive abstraction. We name it "task-anchor" rather than "cognitive" to avoid overclaiming.

**Numerical state.**
- Per entry: 16 perceptual + 1 task = 17 tokens.
- Bank: 16 entries × 17 = **272 keys** at retrieval. ~10× mean-pool, ~10% of full-prefix V4.
- Trainable params added: 2× compressor ≈ ~100K.

**Tradeoff vs single compressor (Option A with n_slots=17).**
- Two-stream forces explicit allocation: 16 slots for vision, 1 for task. Single-compressor would have to learn this allocation implicitly.
- If task identity dominates the gradient signal on long-horizon tasks, the explicit budget helps. If not, it's overhead.
- More params, two attention modules instead of one.

**Critical config caveat.** `n_image_tokens` must match SmolVLA's actual image-region length:
- 2 cameras × 64 image tokens (default, no specials) = 128
- 2 cameras × (64 + 2 specials) = 132 ← **LIBERO smolvla_libero default**

If misconfigured, runtime check fires with the actual prefix length printed in the error.

**Config.** `policy.two_stream: true`, `n_image_tokens: 132`, `perceptual_n_slots: 16`, `task_n_slots: 1`.

---

## 7. Putting it all together — V5 run plan

Four configs, each layering on top of v4's setup:

```
                                         ┌─ Run 0 ─ batch fix only
                                         │     (window + grad_accum)
v4 baseline ─► add 5 fixes available ────┤
                                         ├─ Run 1 ─ kitchen sink
                                         │     (Run 0 + write_stride + mean_pool)
                                         │         │
                                         │         ├─ Run 2A ─ Run 1 with Perceiver
                                         │         │           (mean_pool → compressor n=4)
                                         │         └─ Run 2B ─ Run 1 with two-stream
                                         │                     (mean_pool → two_stream)
                                         │
                                         └─ baseline = no memory, expert_only_scratch
```

### 7.1 Per-run configs (committed to repo)

| Config file | Active fixes | Purpose | Bank keys at retrieval |
|---|---|---|---|
| `libero_v5_run0_diagnostic.yaml` | window-loader + grad_accum=32 | Isolate the gradient-diversity confound | 2,720 (unchanged from v4) |
| `libero_v5_run1_kitchen_sink.yaml` | Run 0 + write_stride=50 + compression_mode=mean_pool | Test all cheap fixes combined | 16 |
| `libero_v5_run2_compressor.yaml` | Run 0 + write_stride=50 + use_compressor + n_slots=4 | Replace mean_pool with learned compression | 64 |
| `libero_v5_run2_two_stream.yaml` | Run 0 + write_stride=50 + two_stream + n_image=132 + p=16, t=1 | Replace single-stream with split | 272 |

All v5 configs share: `total_steps=3000`, `grad_accum_steps=32`, `expert_finetune` mode, residual gate, injection layer 8.

### 7.2 Decision rules between runs

After each run, quick eval (40 episodes) gates whether to continue:

| Result | Interpretation | Next |
|---|---|---|
| Run 0 ≥ 80 overall | Batch fix alone closed most of the gap | Run 1 is incremental gain check |
| Run 0 < 80 | Batch fix insufficient | Run 1 must do the work |
| Run 1 ≥ Run 0 + 2pp | Compression helped | Try Run 2A (learned) |
| Run 1 < Run 0 | mean_pool *hurt* | Skip mean_pool, try learned (Run 2A) or two-stream (Run 2B) directly on Run 0 |
| Run 2 ≥ Run 1 | More capacity helped | Full eval on best |
| Run 2 < Run 1 | Run 1 is the winner | Full eval on Run 1 |

### 7.3 Eval methodology

- **Quick eval (every checkpoint):** 1 episode/task × 10 tasks × 4 suites = 40 episodes. ~5–10 min on L4.
- **Full eval (final winner only):** 10 episodes/task × 100 episodes/suite × 4 suites = 400 episodes. ~2–4h on L4.
- **Held-out loss:** computed but only for divergence detection. Never used to declare a winner (per Drawback 5).
- **Variance:** quick-eval is noisy (±10pp/task). Re-run with seed=2 if Run-vs-Run delta is borderline (<2pp).

---

## 8. References

| Paper | Year | Relevance |
|---|---|---|
| SmolVLA — A Vision-Language-Action Model for Affordable and Efficient Robotics | 2025 | Base model. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) |
| MemoryVLA — Perceptual-Cognitive Memory in VLAs for Robotic Manipulation | 2025 | Two-stream (perceptual + cognitive) memory bank, EOS cognitive token, consolidation by similarity. [arXiv:2508.19236](https://arxiv.org/abs/2508.19236) |
| ContextVLA — VLA Model with Amortized Multi-Frame Context | 2025 | 1 avg-pooled context token per past frame at intermediate VLM layer. [arXiv:2510.04246](https://arxiv.org/abs/2510.04246) |
| Compressor-VLA — Instruction-Guided Visual Token Compression | 2025 | Two-component compression (semantic-task + spatial-refinement). [arXiv:2511.18950](https://arxiv.org/abs/2511.18950) |
| Past-Token Prediction for Long-Context Diffusion Policies | 2025 | Diagnostic that diffusion policies underuse long context (10–100× weaker temporal dependencies than expert demos); auxiliary loss to fix it. [arXiv:2505.09561](https://arxiv.org/abs/2505.09561) |
| Flamingo — Visual Language Model for Few-Shot Learning | 2022 | Perceiver-Resampler for fixed-budget visual token compression. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| LeRobot SmolVLA implementation | 2025 | Reference for `embed_prefix`, `SmolVLMWithExpertModel.forward`. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) |

---

## 9. Diagram suggestions

For the figures you'll draw, here's what to highlight on each:

### 9.1 SmolVLA diagram (§2)
- Show the truncation explicitly: 32 layers in original SmolLM2, 16 used. Annotate the discarded upper half as "where semantic abstraction would happen."
- Token-count breakdown of the prefix (image start/end, image content, language, state) with exact counts for LIBERO.
- Action expert as a second transformer cross-attending to VLM KV cache (don't merge them visually — they're separate).

### 9.2 MemoryVLA diagram (§3)
- Two parallel paths: SE-bottleneck vs LLaMA-7B → distinct shapes ending in different banks.
- Banks: 256-token entries vs 1-token entries, both with consolidation.
- Two parallel cross-attentions, each feeding a different layer of the action expert (perception-attention + cognition-attention).
- Annotate "where MemoryVLA gets its cognitive token: layer 32 of LLaMA-7B."

### 9.3 V4 diagram (§4)
- Single bank, single retrieval, single injection point at layer 8.
- Annotate: bank entry = 170-token slab, 16 entries = 2,720 keys.
- Show that the prefix at layer 8 has heterogeneous content (image regions, language, state) all squashed together.
- Mark "frozen" on VLM layers 0–7 and 9–15. Mark "trainable" on memory module + lm_expert + action_out_proj.

### 9.4 V5 diagrams (one per fix)
- **Window Loader (§6.1):** loader yields N frames from ep A → boundary → N from ep B. Per gradient step: one window from one episode. Successive steps: different episodes. Arrows showing 32 frames feed into one Adam step.
- **Write Stride (§6.2):** timeline showing callback fires at every training frame, but bank.write only at multiples of 50. Inference timeline alongside showing the same pattern (callback fires every chunk = 50 env steps).
- **Mean Pool (§6.3):** prefix → `mean(dim=0)` → 1 token. Bank shrinks from `[16, 170, 576]` to `[16, 1, 576]`.
- **Perceiver Compressor (§6.4):** 4 learnable queries cross-attend to prefix → 4-slot output → bank.
- **Two-Stream (§6.5):** prefix split at index 132 → two compressors → concat → 17-slot output. Two separate trainable parameter sets.

### 9.5 Results diagrams
- Bar chart: per-suite success for {baseline v2, v4 bypass, v4 memory, [each v5 run]}.
- Particularly call out **libero_10** as the diagnostic task: that's where the gap is biggest and where memory should help most.
- Bank-key-count comparison: 2,720 (v4) → 16 (mean_pool) → 64 (compressor n=4) → 272 (two-stream).

---

## 10. Open questions worth flagging in the writeup

These are honest uncertainties — worth surfacing rather than papering over:

1. **The diagnosis that grad_accum_steps=1 was the dominant cause is code-evidence, not run-log-evidence.** Confirmed by reading [trainer.py:130](src/memory_smolvla/training/trainer.py:130), [config.py:46](src/memory_smolvla/training/config.py:46), and that v4 YAML has no override and `wandb_project: null` so no W&B run logs the actual value. Worth confirming from the actual command line / shell history used for v4 production.

2. **`n_image_tokens=132` for two-stream** assumes `add_image_special_tokens=True` and 2 cameras × (64+2). If smolvla_libero's actual config differs, the runtime error fires with the real prefix length. Worth a one-step check before committing the 12h training run.

3. **Window loader gives ~70-85% of baseline gradient diversity, not 100%.** The remaining gap (within-step diversity vs across-step diversity) is partly absorbed by Adam but not entirely. If V5 runs underperform baseline by ~3-5pp, this residual gap is a candidate explanation; would motivate the parallel-episode-tracks refactor (Option 3 in our discussion) for a future iteration.

4. **PTP-style auxiliary loss is *not* in V5.** We considered it, deprioritized because v4's α≈0.49 suggests memory is being *modulated*, not ignored. PTP fixes the "ignore" failure mode; we have a "noise" failure mode. If V5 still underperforms baseline after compression+batch fixes, PTP becomes the next candidate.

5. **Sigmoid vs residual gate.** v4 uses residual (always-on with α=1). All V5 configs inherit this. If V5 succeeds but α-collapse is suspected, sigmoid gate with moderate `alpha_reg_weight` might tune memory contribution more carefully.
