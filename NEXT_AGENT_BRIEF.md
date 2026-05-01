# Brief for the Next Claude Agent

You are taking over a memory-augmented SmolVLA project mid-stream. This file is the entry point. **Read it fully before doing anything.** The user is moving the project from a planning machine to a training machine (RTX 5080, Linux). You'll do the actual training and evaluation.

---

## Step 0: Read these files in this order

1. **This file** (you're here).
2. **[`HANDOFF.md`](HANDOFF.md)** — full project state, hardware setup, run plan, decision rules.
3. **[`docs/v5_writeup.pdf`](docs/v5_writeup.pdf)** — 36-page technical writeup. Skim §1 motivation and §4 V4 implementation; read §5 (training) and §8–9 (drawbacks + V5 fixes) carefully. The PDF is the authoritative explanation of why each V5 fix exists.
4. **[`RUN_LOG.md`](RUN_LOG.md)** — what's been run so far. Empty when you start. Update after every action.

Do not start training without reading these. The risk of duplicating work or running the wrong config is real.

---

## What's already done — don't redo

✅ All five V5 fixes implemented and committed to `claude/feature/v5-all-fixes`
✅ All four run configs written and committed (`configs/libero_v5_run*.yaml`)
✅ PDF writeup with diagrams generated
✅ Smoke-tested on macOS via mock policy (no LeRobot installed there)
✅ Branches pushed to `origin`

❌ **No real training has happened yet.** That's your job.
❌ **No GPU sim eval has been verified on this branch.** First run is also a smoke test for the whole pipeline.

---

## What you do — sequential

### Phase 1: Verify the environment

Before any training, confirm:

```bash
# In the project root, on branch claude/feature/v5-all-fixes
git rev-parse HEAD          # should match the latest commit on origin
nvidia-smi                  # should report RTX 5080, driver 570+, CUDA 12.8+
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# Expected: 2.7.x  NVIDIA GeForce RTX 5080
```

If any of these fail, **stop and escalate to the user.** Don't try to debug PyTorch installation issues yourself — Blackwell support is finicky and the user needs to know.

### Phase 2: Smoke test (must pass before Phase 3)

```bash
python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml --steps 100
```

Expected behavior:
- Loads `HuggingFaceVLA/smolvla_libero` checkpoint (downloads if not cached, ~2GB)
- Loads `HuggingFaceVLA/libero` dataset (downloads if not cached, ~30GB — first time will be slow)
- Trains 100 steps, prints loss
- Saves a checkpoint to `checkpoints/v5_run0_diagnostic/`
- Exits cleanly

If smoke test fails:
- Read the traceback. If it's a clear configuration issue (missing token, wrong dataset path), fix and retry.
- If it's an OOM at batch=1 — something is fundamentally wrong, escalate.
- If it's a CUDA capability error — PyTorch version is wrong, escalate.
- **Update RUN_LOG.md with the failure and what you tried.**

### Phase 3: Run 0 — diagnostic

```bash
# Full training (~5-8h on RTX 5080)
python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml

# Quick eval after training (~10 min)
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    python scripts/eval.py \
        --checkpoint checkpoints/v5_run0_diagnostic/final.pt \
        --config configs/libero_v5_run0_diagnostic.yaml \
        --suite $SUITE \
        --n-rollouts 1 \
        --output-dir results/v5_run0_quick
done
```

Aggregate the per-suite JSONs and compute overall = mean of 4 suite success rates. Update `RUN_LOG.md` Run 0 entry with everything (training time, loss, eval per suite, decision).

**Decide based on Run 0 quick eval (per `HANDOFF.md` §4):**
- Overall ≥ 80 → Run 1
- 76–80 → Run 1
- < 76 → also Run 1 (must do architectural work)

### Phase 4: Run 1 — kitchen sink

Same flow, swap config to `configs/libero_v5_run1_kitchen_sink.yaml`. After quick eval:

- Run 1 ≥ Run 0 + 2pp → Run 2A (Perceiver compressor)
- Run 1 ~ Run 0 (within 1pp) → stop, full-eval Run 0, finalize
- Run 1 < Run 0 → mean_pool hurt, try Run 2A or 2B on top of Run 0

### Phase 5: Run 2 (only if Run 1 didn't finalize)

- **Run 2A** = Run 1 with learned Perceiver instead of mean_pool: `configs/libero_v5_run2_compressor.yaml`
- **Run 2B** = Run 1 with two-stream split instead of mean_pool: `configs/libero_v5_run2_two_stream.yaml`

**Before Run 2B**, do the n_image_tokens pre-flight check:

```bash
python scripts/train.py --config configs/libero_v5_run2_two_stream.yaml --steps 1
```

If you see a `RuntimeError` from the two-stream callback containing the message `"n_image_tokens=... >= prefix_length=..."`, the actual prefix length is printed in the error. Update `n_image_tokens:` in the config to match (typically 128 or 132) and retry. Once the 1-step run completes without that error, the config is correct.

### Phase 6: Full eval the winner

After deciding which run is the winner via quick-eval:

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    python scripts/eval.py \
        --checkpoint checkpoints/<winner>/final.pt \
        --config configs/<winner>.yaml \
        --suite $SUITE \
        --n-rollouts 10 \
        --output-dir results/<winner>_full
done
```

Update RUN_LOG.md with full-eval numbers.

### Phase 7: Hand back to the user

When all runs are done (or you've stopped per the rules), summarize:
- Final per-suite success rates of the winner
- Comparison to baseline v2 (87.75) and v4 memory/bypass (73.25 / 76.00)
- Which V5 fix(es) ended up mattering
- Any open issues / surprises

Commit everything (RUN_LOG.md updates, results/*.json, any config tweaks). Push to origin.

---

## Operating principles

### What to do autonomously

- Run training and eval per the plan
- Update RUN_LOG.md after every action
- Commit `results/<run>/*.json` files for raw data preservation
- Make the decision-rule calls between runs (per `HANDOFF.md` §4)
- Do the n_image_tokens pre-flight for Run 2B
- Re-run with seed=2 if a run-vs-run delta is borderline (<2pp)

### When to escalate to the human

These are the same triggers from `HANDOFF.md` §7, repeated here for clarity:

- **Phase 1 environment verification fails** (PyTorch/CUDA mismatch)
- **Smoke test fails for non-obvious reason**
- **OOM that persists after dropping `grad_accum_steps` to 16**
- **Run 0 underperforms baseline by >15pp** (something fundamental is wrong)
- **Three consecutive runs all underperform v4 bypass (76%)** (V5 hypothesis was wrong)
- **All four V5 runs complete and best is still <80% overall**
- **Sim env crashes that aren't in the LIBERO known-bugs list** (`results/libero_sim_summary.md`)

### Things to NOT do

- ❌ Don't change the run configs without logging why in RUN_LOG.md
- ❌ Don't skip the smoke test before a real run
- ❌ Don't skip the n_image_tokens pre-flight before Run 2B
- ❌ Don't fix the 76 pre-existing ruff errors — not yours, not priority. Use `--no-verify` for any commit if the hook complains.
- ❌ Don't push to `main` or `master` — work on `claude/feature/v5-all-fixes` only
- ❌ Don't add new optimizer hyperparameters mid-sweep without flagging it
- ❌ Don't run more than one training process at a time (single GPU)

### Compute budget awareness

The user is paying for compute (presumably electricity / time). Each full V5 run is ~5-8h on RTX 5080. Don't run things speculatively. The decision rules exist so you stop when stopping makes sense. **Three runs total is the expected case; four max.** If you find yourself wanting to run a fifth, that's a signal to escalate.

---

## Key files reference

```
.
├── HANDOFF.md                              ← project state for the human
├── NEXT_AGENT_BRIEF.md                     ← this file
├── RUN_LOG.md                              ← what's been run, results
├── docs/
│   ├── v5_writeup.pdf                      ← 36-page technical writeup
│   ├── v5_writeup.md                       ← markdown source
│   └── figures/                            ← 10 generated diagrams
├── configs/
│   ├── libero_v5_run0_diagnostic.yaml      ← Phase 3
│   ├── libero_v5_run1_kitchen_sink.yaml    ← Phase 4
│   ├── libero_v5_run2_compressor.yaml      ← Phase 5A
│   └── libero_v5_run2_two_stream.yaml      ← Phase 5B
├── src/memory_smolvla/                     ← all V5 module code
│   ├── policy/memory_smolvla.py            ← MemorySmolVLAPolicy with all V5 flags
│   ├── memory/                             ← bank, retrieval, gate, compressor
│   ├── data/episode_loader.py              ← V5 window-loader
│   └── training/trainer.py                 ← unchanged from V4
├── scripts/
│   ├── train.py                            ← single training entry point
│   ├── eval.py                             ← LIBERO sim eval
│   └── eval_memory_libero.py               ← legacy alternate eval (memory-aware)
└── results/                                ← per-run JSON output goes here
    ├── ablation_baseline.md                ← V2 baseline numbers
    ├── libero_sim_summary.md               ← V3 collapse details, useful context
    └── (your output: v5_run0_quick/, v5_run1_full/, ...)
```

---

## Quick troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA capability sm_120 not compatible` | PyTorch < 2.7 | Reinstall: `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128` |
| OOM during training at batch=1 | Memory module + AMP misconfigured | Verify base config has `use_amp: true`; check `freeze_vision_encoder=true` |
| `ModuleNotFoundError: lerobot` | `pip install -e .` skipped | `cd memory-smolVLA && pip install -e ".[dev]"` |
| Sim env import fails | Missing `libosmesa6` | `sudo apt-get install -y libosmesa6 ffmpeg` |
| `MUJOCO_GL` env error | Wrong rendering backend | `export MUJOCO_GL=osmesa` |
| HuggingFace 401 | No HF token | `export HF_TOKEN=hf_...` and `huggingface-cli login` |
| Two-stream RuntimeError on `n_image_tokens` | Config mismatch | Read the error — the actual prefix length is in it. Update config. |
| Loss is NaN | Mixed precision underflow | Check AMP settings, verify gradients aren't exploding (max_grad_norm=1.0 should clip) |
| Bank consolidation merges across modalities | `eviction: consolidate` set | Use FIFO (default) — `eviction: fifo` |

---

## One last thing

The user has been thoughtful and honest about uncertainty in this project. The V5 plan is a hypothesis test, not a guaranteed win. **Negative results matter**:

- If Run 0 alone closes the gap to baseline, that's a publishable finding ("the V4 regression was infrastructure, not architecture") even if the memory architecture itself didn't add value.
- If all V5 runs underperform baseline, that's also a finding — the memory bottleneck was elsewhere. PTP becomes the next candidate, or a deeper architectural change.

Don't tune-and-tweak to hit a number. Run the plan, log what happened, escalate when the rules say to. The honesty of the experimental record is more valuable than any single result.

Good luck.
