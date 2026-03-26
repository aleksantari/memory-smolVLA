# Held-Out Loss Evaluation: Memory Injection Depth Ablation

**Date:** 2026-03-26
**Dataset:** danaaubakirova/svla_so100_task1_v3
**Val split:** Last 20% of episodes (10 episodes, 4032 frames)
**Training:** 100k steps, memory_only mode (frozen VLM + frozen action expert)

## Results

| Model | Injection Layer | Avg Loss | Gate Alpha | vs Base |
|-------|----------------|----------|------------|---------|
| raw_smolvla_base (no training) | N/A | 1828.13 | 0.007 | -- |
| memory_layer4 (quarter) | 4/16 | **1676.45** | 0.890 | **-8.3%** |
| memory_layer8 (half) | 8/16 | 1677.39 | 0.964 | -8.2% |
| memory_layer12 (three_quarter) | 12/16 | 1701.29 | 0.988 | -6.9% |

## Key Findings

1. **All memory models significantly outperform raw SmolVLA** -- 7-8% lower flow-matching loss. The temporal memory system is learning useful episodic context.

2. **Early injection (layer 4) performs best** by a small margin, suggesting low-level visual features benefit most from temporal memory augmentation. Upper VLM layers have more room to integrate the memory signal.

3. **Late injection (layer 12) is worst of the three** -- still much better than base, but with only 4 layers above the injection point, there is less capacity to integrate memory-augmented features before the action expert.

4. **Layer 4 vs layer 8 difference is negligible** (~0.06%), while layer 12 shows a meaningful gap (~1.5%). The sweet spot is mid-to-early injection.

5. **Gate alpha confirms learned memory usage**: trained models open the gate to 0.89-0.99, while the untrained base stays at 0.007 (identity initialization working as designed).

## Per-Episode Losses

All models show consistent relative ordering across episodes, with no episodes where memory hurts performance.

## Raw Data

See `loss_comparison.json` for full per-episode breakdowns.
