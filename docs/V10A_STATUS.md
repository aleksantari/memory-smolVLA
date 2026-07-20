# V10A implementation status (Coconut latent reasoning)

Spec: [`V10A_PLAN.md`](../V10A_PLAN.md) (RevB). This tracks what is built + validated
vs. what remains GPU-gated behind the running V9 control.

## Done + validated (CPU, V9 untouched)

| Step | What | Gate / evidence |
|---|---|---|
| 0 | lerobot compat audit | `scripts/compat_check.py` — ALL PASS |
| 0.5 | cached ≡ monolithic V9 | `scripts/test_path_equivalence.py` — abs_diff 0.0, grads match |
| 1 | `ForwardMode` enum; bank exposes retrieval/memory for the seed | in suite |
| 2 | Coconut modules (seed, feedback, MatchInputRMS, latent adapter) | `tests/test_coconut_modules.py` (12) |
| 3 | Thought orchestration | `tests/test_v10a.py` (10) + `scripts/test_v10a_smoke.py` (5) |
| 5 | Test suite | **46 tests pass** |

**Core invariants proven** (smoke, real LIBERO batch):
- **K=0 bit-exact vs V9** — `abs_diff 0.0` (control is uncompromised).
- **Cache grows by exactly one token/layer** per thought pass.
- **Latent adapters identity at init** (zero-init up-proj).
- **Per-example RoPE offset decoupled from visible length** (Invariant 5): under
  `final_only` K=2, `action_position_offset = valid_prefix_len + 2` while the
  visible cache holds 1 thought key — verified on a batch with *mixed* prefix
  lengths (144 vs 141).
- **Gradient routing** into seed + adapters; thought→memory blocked (detached
  cache + stop-grad seed inputs); V9 BPTT through the live prefix preserved.

**Bug fixed:** the trainer optimizer collected only `mem_bank.parameters()`; the
Coconut modules live on the policy, so they'd have silently never trained. Now
grouped at `memory_lr`.

## Remaining — GPU-gated (waits for V9 to finish, ~day 6)

- **Step 6 arms** via `scripts/run_v10a_arms.sh` (waits for the GPU, then trains
  K1 → K2 with auto-resume supervision):
  - Arm 1 (control): the running V9.
  - Arm 2: `configs/memvla_libero_v10a_k1.yaml` (K=1).
  - Arm 3: `configs/memvla_libero_v10a_k2.yaml` (K=2, headline).
  - Arms share init by construction (seed 1000 + identical module creation order).
- A brief real-GPU training smoke on the K1 arm (a few hundred steps, no NaN,
  loss tracks V9) should run first, once the GPU is free.

## Gates (pre-registered)

- **Gate 1** K2 − K1 > +5pp on `libero_10` ⇒ latent reasoning depth helps.
- **Gate 2** K≥1 noninferior to V9 on `libero_10` (`--bypass-thoughts` gives the
  in-checkpoint K=0 control).
- **Gate 3** latency Pareto (thought passes are cheap: self-attn only, cached).

## Eval

```bash
MUJOCO_GL=glfw python scripts/eval_memory_libero.py \
  --checkpoint checkpoints/memvla_libero_v10a_k2/final.pt \
  --config configs/memvla_libero_v10a_k2.yaml --all-suites --n-episodes 10 --ensemble
# add --bypass-thoughts to force K=0 (the in-checkpoint V9 control)
```
Coconut inference runs through `predict_action_chunk` (chunk / `--ensemble`);
`select_action`'s base queue path is not thought-spliced and raises if K>0.
