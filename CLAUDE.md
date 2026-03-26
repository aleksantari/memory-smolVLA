# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project uses a Miniconda conda environment named `smolvla`.

- **Conda path**: `C:\Users\Tarik Metin\miniconda3`
- **Environment**: `smolvla` (`C:\Users\Tarik Metin\miniconda3\envs\smolvla`)
- **Python executable**: `C:\Users\Tarik Metin\miniconda3\envs\smolvla\python.exe`
- **HuggingFace**: Authenticated as `tarmus`

In Git Bash, conda is not on PATH. Use the full python path to run scripts:
```bash
/c/Users/"Tarik Metin"/miniconda3/envs/smolvla/python.exe scripts/train.py --config configs/memory_only.yaml
```

## Commands

```bash
# Install in editable mode (inside conda env)
pip install -e ".[dev]"

# Lint
ruff check src/ tests/ scripts/

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_memory_bank.py -v

# Run a single test by name
pytest tests/test_memory_bank.py::test_consolidation -v
```

A pre-commit hook runs `ruff check` automatically on staged changes to `src/`, `tests/`, and `scripts/`. Fix lint errors before committing.

## Architecture

The project wraps a frozen `SmolVLAPolicy` (from LeRobot) with a trainable temporal memory system. Only the memory modules have learnable parameters.

### The injection problem

`SmolVLMWithExpertModel.forward()` manually decomposes each transformer layer (calls `q_proj`, `mlp`, etc. individually) rather than calling `layer.forward()`. Standard PyTorch `register_forward_hook` on decoder layers **does not fire**. The solution is `FeatureExtractor`, which monkey-patches `vlm_with_expert.forward()` with an exact replication of the original layer loop plus a single injection point.

### Data flow

```
base_policy.forward(batch)
  └─► SmolVLMWithExpertModel._patched_forward()  [installed by FeatureExtractor]
        └─► layer loop 0..N
              └─► after layer `injection_layer`: _memory_callback() fires
                    ├─ write prefix hidden states to ConsolidatingMemoryBank (detached)
                    ├─ read_all() → [K, N_tok, D] memories + [K] timestamps
                    ├─ CrossAttentionRetrieval: current queries attend to memory keys+PE
                    ├─ memory_proj (zero-init Linear) projects retrieved features
                    └─ SigmoidGate fuses current + retrieved → augmented prefix
              └─► layers injection_layer+1..N-1 process the augmented prefix
        └─► text_model.norm
  └─► action expert cross-attends to augmented VLM KV cache
  └─► flow-matching loss
```

### Key design invariants

- **Identity start**: `memory_proj` weights are zero-initialized; gate final layer has `bias=-5.0` (α ≈ 0.007). The model is functionally identical to vanilla SmolVLA at init.
- **No BPTT**: Bank entries are always `.detach()`ed before storage. Gradients flow through current-step retrieval/gating only.
- **B=1 only**: `_memory_callback` asserts batch size 1. Episode-sequential training requires frames in temporal order within a single episode.
- **Injection layer**: `injection_layer` is a constructor int (0-based). Valid range for the default 16-layer SmolVLA: 0–15. Changing it requires matching the pretrained action expert's expected layer count.
- **KV cache**: During inference, the callback fires during the cache-build call (`fill_kv_cache=True`, prefix present) and is a no-op during denoising steps (`inputs_embeds[0] is None`).

### Trainable parameters

Only three modules train — everything else is frozen:
- `CrossAttentionRetrieval` (cross-attn + layer norm)
- `SigmoidGate` (2-layer MLP)
- `memory_proj` (Linear, zero-init)

Use `policy.trainable_parameters()` to pass only these to the optimizer.

### Source dependency

The patched forward in `feature_extractor.py` replicates `SmolVLMWithExpertModel.forward()` from:
`/home/santari/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py` lines 404–499

If that upstream file changes, `_patched_forward` must be updated to match.
