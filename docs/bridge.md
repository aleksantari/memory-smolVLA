# How Phase 3 Bridges Memory into SmolVLA

This document explains how the two Phase 3 files (`feature_extractor.py` and `memory_smolvla.py`) create the bridge between our custom memory system and the existing SmolVLA architecture.

## The Problem

SmolVLA processes everything in a single forward pass through `SmolVLMWithExpertModel.forward()` — a layer-by-layer loop that jointly processes the VLM prefix (images + language + state, hidden_size=576) and the action expert suffix (noisy actions, hidden_size=432). We need to intercept at layer 8 (of 16), augment the prefix with episodic memory, and let the augmented version flow through the remaining layers.

Two complications make this non-trivial:

1. **No standard hooks available.** SmolVLA's forward loop manually calls `layer.input_layernorm()`, `layer.self_attn.q_proj()`, `layer.mlp()`, etc. instead of calling `layer.forward()`. PyTorch's `register_forward_hook` on decoder layer modules will never fire.

2. **We can't just post-process.** We need to modify the prefix *mid-forward-pass* so upper VLM layers and the action expert both see enriched context. A post-hoc residual addition would be weaker because the upper layers wouldn't process the augmented information.

## The Solution: Two Files Working Together

### File 1: `feature_extractor.py` — The Surgical Instrument

This file does one thing: **replicate SmolVLA's forward loop verbatim, with a single injection point added.**

#### How it works

1. **At construction** (`__init__`, line 58), it saves the original `vlm_with_expert.forward` method, then replaces it with `_patched_forward`:

   ```python
   self._original_forward = vlm_with_expert.forward
   vlm_with_expert.forward = self._patched_forward
   ```

2. **`_patched_forward`** (lines 95-217) is a faithful copy of `SmolVLMWithExpertModel.forward()` (lines 404-499 of the LeRobot source). Every line of the original layer loop is preserved — the attention dispatch, post-attention residuals, final norm. The **only addition** is 5 lines at the injection point (lines 199-207):

   ```python
   # === INJECTION POINT ===
   if (layer_idx == self._injection_layer
       and self._callback is not None
       and inputs_embeds[0] is not None):
       inputs_embeds[0] = self._callback(inputs_embeds[0], layer_idx)
   ```

   This fires after layer 8's residual+MLP processing is complete. It replaces `inputs_embeds[0]` (the VLM prefix stream) with whatever the callback returns. Layers 9-15 then process the modified prefix.

3. **Three guards** ensure it only fires when appropriate:
   - `layer_idx == 8` — only at the injection layer
   - `self._callback is not None` — only when actively enabled
   - `inputs_embeds[0] is not None` — prefix must exist (blocks during denoise steps where only suffix is processed with cached KV)

#### Behavior across SmolVLA's three forward modes

| Mode | `inputs_embeds` | Callback fires? | Why |
|------|-----------------|-----------------|-----|
| **Training** | `[prefix, suffix]` | Yes | Both streams present, prefix exists |
| **Inference: cache build** | `[prefix, None]` | Yes | Prefix exists; augmented KV cache is stored for upper layers |
| **Inference: denoise step** | `[None, suffix]` | No | `inputs_embeds[0] is None`; but the cached KV already contains augmented info |

### File 2: `memory_smolvla.py` — The Orchestrator

This file wraps the frozen SmolVLA and provides the callback that does the actual memory work.

#### Construction (lines 54-120)

```
MemorySmolVLAPolicy.__init__(base_policy):
    1. Freeze all base policy parameters
    2. Detect d_model=576 from base_policy.model.vlm_with_expert.config
    3. Create trainable memory modules:
       - ConsolidatingMemoryBank(max_size=16)
       - CrossAttentionRetrieval(d_model=576, n_heads=4)
       - SigmoidGate(d_model=576, hidden_dim=256)
       - memory_proj = Linear(576, 576, bias=False)  ← ZERO-INIT
    4. Initialize gate bias to -5.0 → alpha ≈ sigmoid(-5) ≈ 0.007
    5. Install FeatureExtractor on vlm_with_expert at injection_layer=8
```

The zero-init `memory_proj` + negative gate bias means the model starts as **vanilla SmolVLA** — memory influence is near-zero initially and grows during training.

#### Training forward (lines 126-160)

```
MemorySmolVLAPolicy.forward(batch):
    1. Install callback → feature_extractor.set_callback(self._memory_callback)
    2. Delegate       → base_policy.forward(batch)
                         └→ VLAFlowMatching.forward()
                            └→ vlm_with_expert.forward()  ← NOW PATCHED
                               ├→ Layers 0-7: run normally
                               ├→ Layer 8 completes → CALLBACK FIRES
                               └→ Layers 9-15: run with augmented prefix
                         └→ suffix_out → action_out_proj → flow-matching loss
    3. Remove callback → feature_extractor.set_callback(None)
    4. Return (loss, loss_dict) with gate statistics
```

#### The callback — the heart of the bridge (lines 223-298)

This is what actually happens when the injection point fires:

```
Input: prefix_hidden [1, ~170, 576]  (VLM prefix after layer 8)

Step 1: WRITE to bank
        bank.write(prefix_hidden[0], timestamp=t)
        → detaches tensor, stores [170, 576] on CPU
        → if bank exceeds max_size, consolidates (merges two most similar entries)

Step 2: READ from bank
        memories, timestamps = bank.read_all(device)
        → memories: [K, 170, 576]   (K entries, each 170 tokens of 576-dim features)
        → timestamps: [K]            (when each entry was stored)

Step 3: COMPUTE TIME DELTAS
        time_deltas = current_time - timestamps → [K]
        e.g., [4, 3, 2, 1, 0] for 5 entries

Step 4: RESHAPE for cross-attention
        memory_batch:      [K, 170, 576] → [1, K*170, 576]
        time_deltas_batch: [K] → repeat each 170× → [1, K*170]

Step 5: CROSS-ATTENTION RETRIEVAL
        retrieved = retrieval(
            query=prefix_hidden,      # [1, 170, 576] — current features
            key=memory + temporal_PE,  # [1, K*170, 576] — time-aware memory
            value=memory,              # [1, K*170, 576] — raw memory content
        ) → [1, 170, 576]

        Temporal PE is added to keys only, so attention weights are
        time-aware but retrieved content stays clean.

Step 6: ZERO-INIT PROJECTION
        retrieved = memory_proj(retrieved) → [1, 170, 576]
        At initialization: all zeros (model starts as vanilla SmolVLA)
        After training: learned transformation of retrieved features

Step 7: GATED FUSION
        fused, alpha = gate(current=prefix_hidden, retrieved=retrieved)
        alpha = sigmoid(MLP(concat(current, retrieved))) → [1, 170, 1]
        fused = alpha * retrieved + (1 - alpha) * current → [1, 170, 576]

        At initialization: alpha ≈ 0.007, so fused ≈ 0.993 * current

Return: fused [1, 170, 576] → replaces prefix for layers 9-15
```

#### Inference (lines 166-199)

Inference works the same way, with one key insight about SmolVLA's two-phase inference:

1. **Cache build phase**: SmolVLA processes the prefix with `fill_kv_cache=True`, building key/value cache for each layer. Our callback fires during this phase, augmenting the prefix at layer 8. Layers 9-15 build their KV cache from the **augmented** prefix.

2. **Denoise loop** (10 iterations): SmolVLA processes only the suffix using the cached KV. Our callback does NOT fire (`inputs_embeds[0] is None`). But the cached KV from layers 9-15 already contains the augmented information — so the expert effectively cross-attends to memory-enriched context.

## Gradient Flow

During training, gradients flow back to our memory modules through the frozen VLM:

```
loss (MSE between predicted and target velocity)
  ↓
suffix_out (action expert output)
  ↓
expert upper layers cross-attend to VLM K/V states
  ↓
VLM layers 9-15 (frozen params, but computation graph builds
                  because the INPUT requires grad)
  ↓
augmented prefix = output of our gate
  ↓  ↓  ↓
  ↓  ↓  └→ memory_proj.parameters()  ← GETS GRADIENTS
  ↓  └→ retrieval.parameters()       ← GETS GRADIENTS
  └→ gate.parameters()               ← GETS GRADIENTS
```

The frozen VLM parameters don't accumulate `.grad`, but the computation graph still flows through them because their *input* (our gate output) requires grad. The memory bank entries are `.detach()`ed on write, so there is no backpropagation through time — gradients only flow through the current timestep's retrieval and gating.

## Episode Lifecycle

```
Episode start:
    policy.reset_memory()     # clears bank, resets timestamp to 0

Timestep 0:
    callback fires → writes frame 0 to bank (bank has 1 entry)
                   → retrieves from self (zero-init proj → zeros)
                   → gate output ≈ current (no augmentation yet)

Timestep 1:
    callback fires → writes frame 1 (bank has 2 entries)
                   → retrieves from both frames, time_deltas = [1, 0]
                   → starts producing meaningful augmentation

...

Timestep 16+:
    callback fires → writes frame N (bank at max_size)
                   → consolidation merges two most similar entries
                   → retrieves from 16 entries spanning the episode
                   → rich temporal context available to action expert

Episode end:
    policy.reset_memory()     # clears everything for next episode
```

## Why This Design?

| Alternative | Why we didn't use it |
|-------------|---------------------|
| PyTorch `register_forward_hook` on decoder layers | SmolVLA never calls `layer.forward()` — hooks don't fire |
| Post-hoc residual addition to expert input | Upper VLM layers wouldn't process the augmented information |
| Modify LeRobot source code directly | Fragile, hard to maintain across versions, not composable |
| Two-pass approach (full VLM forward, then re-run with augmented features) | Doubles compute cost; KV cache would be inconsistent |
| Hook on `layer.mlp` submodule | Captures MLP output but not the full layer output (misses residual connection) |

The monkey-patching approach gives us surgical precision — exact replication of the original forward with a single, clean injection point — while keeping LeRobot as an unmodified pip dependency.
