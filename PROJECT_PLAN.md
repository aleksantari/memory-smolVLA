# Memory-Augmented Compact VLAs — Project Plan

## Context

This project adds a temporal memory system to SmolVLA (a 450M-parameter Vision-Language-Action model for robotics). The memory module sits between SmolVLA's frozen VLM backbone (SigLIP + SmolLM2) and its flow-matching action expert, augmenting the VLM's intermediate representations with cross-attended episodic memory. This enables multi-step task reasoning — the action expert receives richer context about what happened earlier in the episode.

**Core architecture:**
```
SmolVLA's frozen VLM (SigLIP + SmolLM2) encodes images
  → layer N/2 output
  → memory module (write to bank → cross-attention retrieval → sigmoid-gated fusion)
  → action expert
  → action chunk
```

**Key constraint:** Training requires contiguous episode subsequences (not shuffled frames), because the memory bank must be populated sequentially during each training episode.

## Critical Architecture Insight

**SmolVLMWithExpertModel.forward() does NOT call decoder layer.forward().** It manually decomposes each layer into `layer.input_layernorm()`, `layer.self_attn.q_proj()`, `layer.mlp()`, etc. Standard PyTorch `register_forward_hook` on decoder layers won't fire. The feature extractor must work around this by replicating the forward loop with capture/injection points.

**Key attribute paths:**
- VLM layers: `policy.model.vlm_with_expert.vlm.model.text_model.layers[i]`
- Expert layers: `policy.model.vlm_with_expert.lm_expert.layers[i]`
- VLM hidden_size: 576, Expert hidden_size: 432, Default 16 VLM layers

**LeRobot source files:**
- SmolVLA policy: `lerobot/policies/smolvla/modeling_smolvla.py`
- VLM+Expert model: `lerobot/policies/smolvla/smolvlm_with_expert.py`
- Configuration: `lerobot/policies/smolvla/configuration_smolvla.py`
- Dataset: `lerobot/datasets/lerobot_dataset.py`
- LIBERO env: `lerobot/envs/libero.py`

## Project Structure

```
memory-smolvla/
├── PROJECT_PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   └── libero_long.yaml
├── src/
│   └── memory_smolvla/
│       ├── __init__.py
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── bank.py
│       │   ├── retrieval.py
│       │   ├── gating.py
│       │   └── temporal_pe.py
│       ├── policy/
│       │   ├── __init__.py
│       │   ├── memory_smolvla.py
│       │   └── feature_extractor.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── episode_loader.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       └── eval/
│           ├── __init__.py
│           └── evaluator.py
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── analyze_gates.py
└── tests/
    ├── test_memory_bank.py
    ├── test_retrieval.py
    └── test_policy_wrapper.py
```

## Implementation Phases

### Phase 1: Foundation files

Create all `__init__.py` files, `pyproject.toml`, and `README.md`.

- **pyproject.toml** — deps: `lerobot[smolvla]`, `torch>=2.2.1`, `wandb`, `pyyaml`, `einops`, `tqdm`. `src` layout with setuptools. Dev extras: `pytest`, `pytest-cov`.

### Phase 2: Memory primitives

All four modules are independent PyTorch modules with no cross-dependencies.

1. **`memory/temporal_pe.py`** — `TemporalPositionalEncoding(nn.Module)`
   - Standard sin/cos encoding for time deltas
   - `forward(time_deltas: [K]) -> [K, d_model]`
   - Log-spaced frequencies from `min_period` to `max_period`

2. **`memory/bank.py`** — `ConsolidatingMemoryBank`
   - Stores `list[(timestamp, Tensor[N_tokens, d_model])]`
   - `write(tokens, timestamp)`: appends (always `.detach()`), consolidates if `len > max_size`
   - `consolidate()`: merges two most similar entries by cosine sim of mean-pooled representations (average tensors, keep newer timestamp)
   - `read_all(device)`: returns `(Tensor[K, N_tokens, D], Tensor[K])`
   - `reset()`: clears between episodes

3. **`memory/retrieval.py`** — `CrossAttentionRetrieval(nn.Module)`
   - `nn.MultiheadAttention(batch_first=True)`
   - Adds temporal PE to memory keys only (not values)
   - `forward(current [B,L,D], memory [B,K*N,D], time_deltas [B,K*N]) -> [B,L,D]`
   - LayerNorm on output

4. **`memory/gating.py`** — `SigmoidGate(nn.Module)`
   - MLP: `concat [current, retrieved] (2*D) -> D (SiLU) -> 1 (Sigmoid)`
   - `forward(current, retrieved) -> (fused, alpha)`
   - `fused = alpha * retrieved + (1 - alpha) * current`

### Phase 3: Policy integration

5. **`policy/feature_extractor.py`** — `FeatureExtractor`
   - Monkey-patches `vlm_with_expert.forward()` to replicate the layer loop with capture/injection points
   - Supports a callback function invoked at target layer(s)
   - Must support N/4, N/2, 3N/4, final for layer-depth ablation
   - Key source to replicate: `smolvlm_with_expert.py:404-499`

6. **`policy/memory_smolvla.py`** — `MemorySmolVLAPolicy(nn.Module)`
   - Wraps `SmolVLAPolicy` (frozen)
   - Owns memory bank, retrieval, gate, feature_extractor
   - Zero-initialized `memory_proj` so model starts as vanilla SmolVLA
   - Training forward: base VLM with callback at injection layer → write to bank → cross-attention retrieval → gated fusion → replace prefix hidden states → continue upper layers → action expert uses augmented prefix
   - `reset_memory()` for episode boundaries
   - `trainable_parameters()` returns only memory module params

### Phase 4: Data loader

7. **`data/episode_loader.py`** — `EpisodeSequentialLoader`
   - Wraps `LeRobotDataset`, iterates episodes (shuffled order)
   - Yields contiguous frame windows within each episode
   - Uses `dataset.meta.episodes` with `dataset_from_index`/`dataset_to_index`
   - Yields `EpisodeBoundaryMarker` sentinels between episodes

### Phase 5: Training and eval

8. **`training/trainer.py`** — `MemorySmolVLATrainer`
   - Episode-sequential training loop
   - Per episode: reset memory → step through frames → accumulate bank → flow-matching loss → backprop through memory module only
   - wandb logging, checkpointing, cosine LR with warmup, gradient clipping

9. **`eval/evaluator.py`** — `MemorySmolVLAEvaluator`
   - Loads checkpoint, runs rollouts on LIBERO via LeRobot env wrappers
   - Records per-task success rate, gate activation timelines, memory overhead

### Phase 6: Configs and scripts

10. **`configs/base.yaml`** — injection_layer=8, bank_size=16, n_heads=4, steps=100K, lr=1e-4, cosine schedule
11. **`configs/libero_long.yaml`** — inherits base, bank_size=32, suite="libero_10"
12. **`scripts/train.py`** — CLI entry with argparse + yaml config
13. **`scripts/eval.py`** — CLI entry for LIBERO evaluation
14. **`scripts/analyze_gates.py`** — matplotlib visualization of gate activation patterns

### Phase 7: Tests

15. **`tests/test_memory_bank.py`** — write/read, consolidation, max_size, reset, detachment
16. **`tests/test_retrieval.py`** — output shapes, temporal PE, gradient flow
17. **`tests/test_policy_wrapper.py`** — forward pass shapes, memory accumulation, reset, trainable param isolation

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Split-and-resume injection** | Memory augments prefix hidden states at layer N/2 before upper layers process them. More powerful than post-hoc residual addition. |
| **Zero-init memory projection** | Model starts identical to vanilla SmolVLA; memory influence grows during training. |
| **Detached memory writes** | Features stored in bank are detached from computation graph. Gradients flow through current-step retrieval/gating only (no BPTT through memory bank). |
| **Episode-sequential training** | One episode at a time, frames in temporal order. Required for sequential memory bank buildup. |
| **Forward loop replication** | Feature extractor replicates `SmolVLMWithExpertModel.forward()` layer loop since hooks on decoder layers don't fire. |
| **Temporal PE on keys only** | Attention weights are time-aware; retrieved values remain clean. |

## Verification

1. **Unit tests**: `pytest tests/` — memory primitives produce correct shapes, bank consolidation works, gradients flow through retrieval+gate
2. **Smoke test**: `python scripts/train.py --config configs/base.yaml --steps 10` — training loop runs without errors
3. **Identity test**: With zero-initialized gate, `MemorySmolVLAPolicy` produces identical outputs to base `SmolVLAPolicy`
4. **Shape test**: All tensors through memory module maintain SmolVLA-compatible dimensions (prefix: `[B, L, 576]`, expert input: `[B, chunk, 432]`)
