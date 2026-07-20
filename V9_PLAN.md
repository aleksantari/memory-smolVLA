# V9 — Track A: finish LIBERO

**Goal:** push LIBERO from V7's **86.75%** toward the realistic ceiling **~93–95%**, using
levers that don't require a new architecture. (No published VLA hits 100% on all four suites;
LIBERO-Long is everyone's wall. 100% is beyond current SOTA — we target the frontier.)
MoE and multi-benchmark generalization are **Track B (V10+)**, not here.

## 1. Where the points are (V7 mem-on, corrected eval)

| Suite | V7 | gap to 100 | bottleneck |
|---|---:|---:|---|
| object | 99 | ~0 | solved — leave alone |
| spatial | 85 | 15 | spatial precision / grounding |
| goal | 89 | 11 | semantic goal + precision |
| libero_10 | 74 | 26 | long-horizon: compounding error + memory |
| **overall** | **86.75** | — | — |

The remaining points are **long-horizon + precision**, not capacity. So the levers are
test-time action quality, more training, and spatial supervision — in that ROI order.

## 2. The three levers

### A1 — Test-time action ensembling (ACT-style)  ✅ DONE: +2pp, FREE
**RESULT (2026-07-11, V7 final.pt, query_every=10 decay=0.1):** overall 86.75 -> **88.75**
(spatial 85->85, object 99->99, **goal 89->95 (+6)**, **libero_10 74->76 (+2)**). Zero extra
compute, no retraining, bank cadence pinned to match V7 -> isolates pure action smoothing.
**88.75 now beats baseline v2 (87.75).** Spatial unchanged confirms it's a grounding problem
(-> A3), not smoothing. Applies to EVERY future checkpoint (V8, V9-100k). Results in
`results/v9_ensemble/`. Follow-up: sweep query_every=5 (denser overlap) + decay for more.


**Hypothesis.** SmolVLA predicts a 50-step action chunk but we execute `n_action_steps=10` then
re-query — so actions are stale mid-chunk and jerky at chunk seams, which compounds over
long-horizon rollouts. Overlapping-chunk temporal ensembling (Zhao et al., ACT 2023) smooths
this and is a known large win on precision + long-horizon, with **zero training**.

**Implementation** (eval-only, `scripts/eval_memory_libero.py::run_rollout`):
- Replace the streaming `select_action` (internal queue) with `predict_action_chunk` (returns
  the full chunk) + our own ensemble buffer.
- Keep `buf[t] = list of (action, chunk_issue_step)` for upcoming timesteps.
- Every `query_every` env steps (start with 1–2), run the policy, write its chunk into `buf`.
- Executed action at step `t` = exp-weighted average over `buf[t]`: `w_i ∝ exp(-m·age_i)`
  (start `m≈0.1`). Gripper dim: take the most-recent (or majority-sign), don't average.
- Memory bank: keep one write per *executed* step (advance `_infer_timestep` on executed
  steps only, not per query) so bank cadence is unchanged.

**Params to sweep:** `query_every ∈ {1,2,3}`, decay `m ∈ {0.01,0.1,0.25}`, chunk horizon used.
**Cost:** eval-only; ~3–5× more forward passes per rollout (slower eval, no training).
**Test now on** `checkpoints/memvla_libero_v7/final.pt` (needs the GPU — see §5 scheduling).
**Expected:** biggest gain on libero_10 + spatial; object already at ceiling.

### A2 — Train to 100k  [cheap-confidence, but ~10.5 days]
**Hypothesis.** V7/V8 stop at **60k** steps (capped for the ~week budget); the 87.75% baseline
used **100k**. We are likely undertrained — free points on the table.
**Plan.** Take the winning config (V7 mean_pool vs V8 reasoning — decided by V8's eval),
set `total_steps=100000`, cosine LR over 100k, warmup 1000 (baseline parity). Checkpoint every
5k; **eval at 60k / 75k / 100k to read the slope** — if it's still climbing at 60k, 100k pays off.
**Cost:** batch 128 ≈ 9.1 s/step → 100k ≈ **10.5 days** on the Spark. Commit only after the
V8 60k eval + intermediate-checkpoint slope justify it (don't burn 10 days on a plateau).
**De-risk first:** eval V8's 25k/50k checkpoints to see the 25→50→60k trajectory before committing.

### A3 — Spatial-reasoning tokens  [research, GATED on V8 result]
**Hypothesis.** The spatial suite (85) needs the memory/reasoning tokens to encode *where*
objects are. V8 already supervises reasoning tokens to predict future proprioceptive state
(PTP). V9 adds/swaps a **spatial** aux target so the tokens encode geometry.
**Only pursue if V8's reasoning tokens beat V7 on spatial/goal.** Target options, cheapest first:
1. **Future eef waypoints** — predict the eef trajectory over the next K steps (richer than V8's
   single future state; uses `observation.state`, no new data). Encodes the spatial path.
2. **Object positions** — predict object 2D/3D pos. Requires object poses; check if extractable
   from LIBERO bddl/sim (dataset may not ship them). If yes, strongest spatial signal.
3. **Depth/occupancy grid** (MolmoAct-style) — offline monocular depth on LIBERO frames →
   coarse grid target. Most work; do last.
Reuse V8's `ReasoningSummaryHead`; swap `FutureStatePredictor` for a `SpatialPredictor`.

## 3. Cheap extras (quick ablations, eval-only)
- **More flow-matching denoise steps at eval** (`num_steps` up from 10) — often improves action
  fidelity for free; quick per-suite ablation.
- **Seed/checkpoint ensembling** — average action chunks across 2–3 seeds/checkpoints for the
  last ~1pp (slow eval, no train).
- **Re-plan frequency** — subsumed by A1 but worth a standalone `n_action_steps` sweep as a floor.

## 4. Sequencing
1. **A1 first** (free, highest ROI): implement temporal ensembling, eval on V7 final.pt, sweep
   `query_every`/`m`. If it lifts libero_10/spatial, it applies to *every* future checkpoint.
2. **Read the V8 slope** (eval V8 25k/50k/60k): pick the base config (mean_pool vs reasoning) and
   decide if A2 (100k) is worth the 10.5 days.
3. **A2** committed long run at 100k with the winner.
4. **A3** only if V8 reasoning tokens show spatial promise.
5. **Final V9** = winning config @100k + A1 ensembling (+ A3 if it landed) + best cheap extras.

## 5. Scheduling (GPU is busy with V8)
A1's eval and A2's training both need the GPU that V8 is training on. Options:
- **A1 now:** briefly pause V8 (`kill -STOP`) to run the A1 eval on V7 (~1.5 h/condition),
  since it's the highest-ROI signal and needs no training — then resume V8. **Recommended.**
- **A2 later:** start the 100k run after V8 finishes (or if V8's early eval says its config wins,
  kill V8 and start V9-100k from that config).

## 6. Eval protocol + success criteria
Same corrected pipeline (180° flip, glfw, `n_action_steps` base=10), 10 ep × 10 tasks × 4 suites,
mem-on + bypass. **Success:** overall ≥ **92%**, with libero_10 ≥ **82%** and spatial ≥ **90%**.
Report each lever's per-suite delta so we know what bought what.

## 7. Risks / kill-criteria
- **A1 no help / hurts:** if ensembling doesn't move libero_10, the bottleneck is upstream
  (memory content or grounding), not action smoothness — drop it, lean on A2/A3.
- **A2 plateaus:** if V8 25→60k is flat, 100k won't help — skip the 10-day run, save compute.
- **A3 needs data we don't have:** if object poses aren't extractable and depth-gen is too heavy,
  fall back to the waypoint target (option 1), which needs no new data.
- **Diminishing returns near the ceiling:** past ~94%, remaining LIBERO failures may be
  irreducible (physics/grasp noise) — call it and pivot to Track B rather than chase noise.
