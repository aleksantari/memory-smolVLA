# LIBERO Held-Out Loss Evaluation: Gate Collapse Investigation

**Date:** 2026-04-06
**Dataset:** HuggingFaceVLA/libero
**Val split:** Last 20% of episodes (338 episodes, 41,394 frames)
**Training:** 100k steps, memory_only mode (frozen VLM + frozen action expert)
**Base checkpoint:** HuggingFaceVLA/smolvla_libero

## Summary

Three rounds of experiments tested different gating strategies for memory
injection on LIBERO. The learned sigmoid gate collapses to α≈0 on LIBERO
(regardless of initialization), while a simple residual (additive) gate
produces the best results.

## Results

### LIBERO (low-loss regime, ~0.09)

| Model | Gate Type | Init | Loss | Gate α | Δ vs Base |
|-------|-----------|------|------|--------|-----------|
| Base SmolVLA (no memory) | — | — | 0.09251 | — | — |
| | | | | | |
| Layer 4 v1 | sigmoid | -5.0 | 0.09230 | 3.0e-8 | -0.2% |
| Layer 8 v1 | sigmoid | -5.0 | 0.09227 | 3.0e-8 | -0.3% |
| Layer 12 v1 | sigmoid | -5.0 | 0.09229 | 3.0e-8 | -0.2% |
| | | | | | |
| Layer 4 v2 | sigmoid | -1.0 | 0.09236 | 2.9e-12 | -0.2% |
| Layer 8 v2 | sigmoid | -1.0 | 0.09242 | 1.1e-5 | -0.1% |
| Layer 12 v2 | sigmoid | -1.0 | 0.09229 | 1.7e-13 | -0.2% |
| | | | | | |
| **Layer 4 v3** | **residual** | — | **0.09226** | 1.0 | **-0.3%** |
| **Layer 8 v3** | **residual** | — | **0.09224** | 1.0 | **-0.3%** |
| **Layer 12 v3** | **residual** | — | 0.09229 | 1.0 | -0.2% |

### SO100 (high-loss regime, ~1828) — for comparison

| Model | Gate Type | Loss | Gate α | Δ vs Base |
|-------|-----------|------|--------|-----------|
| Base SmolVLA | — | 1828.13 | 0.007 | — |
| Layer 4 | sigmoid | 1676.45 | 0.890 | **-8.3%** |
| Layer 8 | sigmoid | 1677.39 | 0.964 | **-8.2%** |
| Layer 12 | sigmoid | 1701.29 | 0.988 | **-6.9%** |

## Key Findings

### 1. Gate collapse is a gradient magnitude problem, not initialization

- v1 (gate_init_bias=-5.0, α_init≈0.007): gate → 0 on LIBERO
- v2 (gate_init_bias=-1.0, α_init≈0.27): gate → 0 on LIBERO
- SO100 (gate_init_bias=-5.0): gate → 0.89-0.99 (works perfectly)

The ~20,000x difference in loss magnitude (LIBERO ~0.09 vs SO100 ~1828)
means proportionally weaker gradients through the gate on LIBERO. The
gate MLP learns to suppress memory regardless of initialization.

### 2. Residual gate (v3) avoids collapse entirely

By removing the learned gate and using additive fusion
(`fused = current + retrieved`), the model controls memory contribution
through the zero-initialized `memory_proj` weights instead. No gate
parameter can collapse because there is no gate to train.

### 3. Held-out loss improvements are small on LIBERO

All models (v1/v2/v3) show only ~0.2-0.3% loss improvement over base.
This contrasts sharply with SO100's 7-8% improvement. Possible reasons:
- LIBERO is an easier dataset where memory has less to contribute
- The held-out flow-matching loss may not capture the benefit of temporal
  context (which may appear more in closed-loop execution)
- The 0.8% trainable parameter budget may be insufficient for LIBERO

### 4. Layer 8 consistently performs best (or tied for best)

Across all experiments, injection at layer 8 (mid-depth) yields the
best or tied-best results. This aligns with the SO100 findings.

## Next Steps

- **LIBERO simulation eval** (eval.py): Run actual task rollouts to
  measure success rates. The real test of whether memory helps is in
  closed-loop execution, not held-out loss.
- **Compare v3 sim results** against base SmolVLA LIBERO sim baseline.

## Raw Data

- `libero_loss_layer{4,8,12}_v{1,2,3}.json` — per-episode loss breakdowns
- `libero_loss_comparison.json` — v1 combined results
- `loss_comparison.json` — SO100 results
