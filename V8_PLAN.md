# V8 — porting MolmoAct2 ideas into memory-SmolVLA: evaluation + plan

**Question (user):** port some MolmoAct2 innovations into memory-SmolVLA for V8.
**Short answer:** Yes — but only *one* of MolmoAct2's ideas is both novel-to-us and
thesis-advancing. Most of its headline architecture is either **already in SmolVLA**
or **irrelevant to the memory story**. The one worth porting directly attacks a *known*
weakness of this project. This doc separates the wheat from the chaff, then plans the port.

Grounded in the actual MolmoAct2 code (`~/isaac-so101/models/MolmoAct2-SO100_101/
modeling_molmoact2.py`, `config.json`) and our own `smolvlm_with_expert.py` /
`full_seq_bank.py`.

---

## 1. What MolmoAct2 actually is (from its code)
- VLM backbone Molmo2-ER: Qwen3-4B-ish, **36 layers, hidden 2560**, SigLIP2 ViT (27 layers).
- **Action expert**: dedicated DiT, `hidden 768, 36 layers`, adaLN-style timestep modulation
  (`ActionExpertModulation`, 9-way chunk = adaLN-zero), flow-matching (10 steps, Beta(1,1.5)
  time sampling). Each `ActionExpertBlock` = self-attn + **cross-attn to the same-depth VLM
  K/V** (`ActionExpertCrossAttention` reuses VLM K/V directly; only q/out are learned).
- **Discrete state**: proprio quantized into a **256-token state vocab**, fed as tokens in the
  VLM context (`state_format="discrete"`, `num_state_tokens=256`).
- **Two action heads** (`action_mode="both"`): continuous flow-matching **and** OpenFAST
  discrete tokens (`num_action_tokens=2048`, BPE over frequency-domain actions).
- **Depth "Think" reasoning**: emits a **10×10 depth-code grid** (`num_depth_codes=100`) as an
  inspectable spatial-reasoning trace *before* acting (`enable_depth_reasoning`, off in the
  SO-101 checkpoint). This is the MolmoAct signature (v1's "perception tokens"/waypoints).

## 2. The honest filter — what's novel to US and what isn't

| MolmoAct2 idea | Already in SmolVLA? | Helps the *memory* thesis? | Portable to 0.5B? | Verdict |
|---|---|---|---|---|
| Per-layer expert↔VLM **KV bridge** | **YES** — `lm_expert` is a depth-matched copy; every layer cross-attends to same-depth VLM K/V (`smolvlm_with_expert.py:110-121`) | n/a | n/a | **Skip — we already have it** |
| Dedicated DiT action expert (adaLN, deeper) | Partial (ours is a width-½ LLM copy) | No (action-head quality, not memory) | Costly; off-thesis | Skip for V8 |
| OpenFAST discrete action tokens | No | No (action representation, not memory) | Medium | Skip (dilutes thesis) |
| Beta(1,1.5) flow-matching time schedule | Different schedule | Marginal | Trivial | **Free side-experiment** |
| **Discrete state tokenization** (256-vocab, state-as-tokens) | No (we use 1 continuous state token) | **Yes** — proprio *history* is what memory should track (gripper cycles, joint state over time) | Cheap | **Port (secondary)** |
| **Intermediate spatial-reasoning "Think" tokens** (depth grid / perception tokens) | **No** | **YES — directly** | Medium (lighter variant) | **PORT (primary)** |

## 3. The one that matters — and *why* it's the right port

Our own STATUS_REPORT names this project's deepest limitation (§ "Drawback 4 / VLM
truncation"): **SmolVLA truncates SmolLM2 to 16 layers, so there is no true "cognitive
token."** MemoryVLA's whole design stored a *cognitive* token (EOS of LLaMA's final layer)
alongside perceptual tokens; we couldn't, so our bank stores raw/mean-pooled perceptual
prefix tokens — pixel-ish, not semantic. That is very likely *why* memory helps only
weakly on long-horizon: **the memory holds perception, not a distilled scene/task state.**

MolmoAct2's intermediate reasoning tokens are exactly a mechanism to **manufacture a
compact spatial/semantic summary** that a truncated VLM cannot otherwise produce. So the
port isn't cosmetic — it supplies the missing cognitive-memory stream:

> **V8 thesis:** replace/augment the memory bank's *content* — from raw perceptual prefix
> tokens to a small set of MolmoAct2-style **learned spatial-reasoning tokens** generated
> per frame. Memory then stores *what/where the scene is about*, not pixels → the
> cognitive-memory stream MemoryVLA had and we lacked. This is the natural successor to
> V7's compression work (V7 compressed the raw prefix; V8 replaces it with a better token).

This composes with everything V6/V7 established (episode diversity, consolidation matching,
compression) — it changes only *what tokens enter the bank*, not the loader/diversity/gate.

## 4. V8 plan (phased, reuses the V7 harness)

Anchor: V7 result (in progress) + baseline v2 (87.75%). Eval = LIBERO sim on the other box,
mem-on vs `--bypass-memory`, focus libero_10. Same GroupedEpisodeLoader, gate, ToMe, W&B.

**Phase A — cheap wins / de-risk (days, no new data):**
- A1. **Discrete state tokenization**: quantize proprio into K bins → state tokens appended to
  the VLM prefix (mirror MolmoAct2's `num_state_tokens`). Store those in memory too. Isolctes
  "does richer proprio-in-memory help long-horizon?" Cheap, self-contained.
- A2. **Beta(1,1.5) flow time schedule** ablation — one-line change, free signal on action-head.

**Phase B — the primary port: cognitive/spatial reasoning tokens as memory content:**
- B1. Add a small **"summary head"** at the injection layer: N learned latent queries
  cross-attend the VLM prefix → N *reasoning tokens* per frame (reuse V7's `MemoryCompressor`
  Perceiver as the mechanism — we already built and tested it). This is the *light* version of
  MolmoAct's Think tokens: no depth codebook, no external supervision yet.
- B2. **Store the reasoning tokens (not the raw prefix) in `FullSeqMemBank`.** One-line swap of
  `to_store`; retrieval/gate/ToMe unchanged. Now memory = distilled scene state over time.
- B3. **Optional supervision** to make the tokens *mean* something (the real MolmoAct trick):
  auxiliary loss predicting a cheap spatial target — object/gripper 2D position or a
  monocular-depth grid (from an off-the-shelf depth model run offline on LIBERO frames). This
  is what turns "learned latents" into genuine spatial-reasoning tokens. Gated behind B1/B2
  showing life, since it needs an offline data-prep pass.

**Phase C — evaluate as a memory upgrade, not a new model:**
- Head-to-head at matched compute vs V7: overall + per-suite + bypass ablation; the KPI is
  **libero_10** (does distilled-token memory finally move long-horizon?) and whether the gate
  weights memory *more* (mem contribution ↑).
- Ablation isolating content: same architecture, bank stores {raw prefix (V7) | reasoning
  tokens (B) | reasoning+state (A1+B)} — attributes any gain to *content*, cleanly.

## 5. What I would NOT do (and why)
- **Don't** swap in MolmoAct2's DiT expert / OpenFAST head / Molmo2-ER backbone — that's
  becoming MolmoAct, abandoning the compact-SmolVLA + memory thesis, and needs their data.
- **Don't** re-implement the KV bridge — SmolVLA already has per-layer expert↔VLM cross-attn.
- **Don't** start with full depth-reasoning — port the *idea* (distilled reasoning tokens in
  memory) in the light form first; add depth/position supervision only if B shows promise.

## 6. Risks / kill-criteria
- **Reasoning tokens underperform raw prefix** (B1/B2 ≤ V7 on libero_10): then the bottleneck
  wasn't content — revert; the honest finding is "memory content isn't the limiter."
- **Latent tokens don't learn structure without supervision**: expected; that's why B3 exists.
  If even supervised tokens don't help, spatial memory isn't the lever for LIBERO (episodes
  may be too short — same caveat V5 raised).
- **Compute/timing**: V8 training reuses V7's ~6-day budget on this box; only start after V7
  finishes + is evaluated (don't contend for GPU). Prototype B1/B2 forward on CPU/small first.

## 6b. IMPLEMENTED + LAUNCHED (2026-07-11)

Built the primary port (Phase B + B3, the reasoning-tokens + PTP-supervision core):
- `memory/reasoning.py` (NEW, Fable-5 agent): `ReasoningSummaryHead` (Perceiver latents →
  `n_slots` reasoning tokens per frame; wraps the V7 `MemoryCompressor`) + `FutureStatePredictor`
  (mean-pool → MLP → future state). 7 unit tests pass.
- `data/group_loader.py` (Fable-5 agent): emits per-frame `future_states` + `future_valid`
  (state `future_horizon` frames ahead within the group). 11 unit tests pass.
- `memory/full_seq_bank.py`: `compression="reasoning"` stores the learned reasoning tokens as
  bank content; keeps a grad-enabled copy (`_last_reasoning`); `aux_future_loss()` = masked MSE
  from those tokens to the future state.
- `policy/memory_smolvla.py`: adds `aux_loss_weight`, computes the PTP aux loss in `forward`
  (`loss + λ·aux`); `builder.py`/`train.py`/`trainer.py` plumb `state_dim`, `aux_loss_weight`,
  `future_horizon` and route the future targets around the preprocessor.
- Bank integration test: reasoning tokens stored (n_slots×D), aux loss computes, grads flow to
  BOTH the summary head and predictor. Full 2-step smoke: `aux=1.33` fires on real data. 27 tests pass.

**Config** `configs/memvla_libero_v8.yaml` = V7 verbatim + `compression: reasoning`, `n_slots: 8`,
`future_horizon: 5`, `aux_loss_weight: 0.1`. Everything else (diversity 16, consolidation-matching
group_size 8 > mem_length 4, LR, 60k steps) identical to V7 so the comparison is clean.

**Running:** `scripts/run_v8_supervised.sh`, batch 128, 60k steps (~6 days), auto-resume,
W&B run `memvla_libero_v8` (fkcrvaex). Eval when done: same pipeline, content-ablation vs V7
(mem-on/bypass), KPI = whether reasoning-token memory lifts goal/libero_10 above V7's 89/74.

## 7. Recommendation
Do **Phase A + B** as V8: it's a focused, thesis-true port of MolmoAct2's *one* genuinely
transferable innovation — structured intermediate reasoning tokens — repurposed to give our
memory bank the cognitive/spatial content SmolVLA's truncated VLM can't produce. Treat B3
(supervision) as the swing-for-the-fences follow-on. Skip the rest of MolmoAct2; it's either
already ours or off-thesis.
