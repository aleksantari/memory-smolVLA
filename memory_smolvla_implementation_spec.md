# Memory-SmolVLA Implementation Spec

**Target executor:** Claude Code
**Goal:** Add a single-stream, full-sequence memory module to SmolVLA. The memory module stores the full VLM output sequence per timestep, retrieves historical context via cross-attention, fuses it via a learned sigmoid gate, and consolidates the bank via token-merge when capacity is reached. The module is adapted from MemoryVLA's `CogMemBank`.

---

## 1. Scientific context (for your understanding, do not re-derive)

- Baseline: vanilla SmolVLA (450M), action expert trained from scratch, LIBERO fine-tuning. Already reproduced.
- This run: SmolVLA + full-sequence cognitive memory. Single additional training run.
- Non-goals for this implementation:
  - Dual-stream (perceptual + cognitive) — single-stream only
  - Changes to SmolVLA's action expert — it must see the same shape it sees in vanilla SmolVLA
  - Changes to the SigLIP → SmolLM2 forward pass — the memory module sits entirely between VLM output and action expert input

---

## 2. Where the memory module sits

```
image ──┐
         ├─► SigLIP ─► project ─► SmolLM2 (full depth, all layers) ─► H_t [B, L, D]
text ───┘                                                                   │
                                                                            ▼
                                                          ┌───────────────────────────────┐
                                                          │  MEMORY MODULE (new)           │
                                                          │                                │
                                                          │  1. Store H_t in bank          │
                                                          │  2. Retrieve from bank via     │
                                                          │     cross-attention (query=H_t)│
                                                          │  3. Gated fusion               │
                                                          │  4. Consolidate bank           │
                                                          │                                │
                                                          │  Output: H_fused [B, L, D]     │
                                                          └───────────────────────────────┘
                                                                            │
                                                                            ▼
                                                            Action Expert (unchanged)
                                                            cross-attends to H_fused
                                                            10 flow-matching steps
                                                            → action chunk
```

**The action expert sees `H_fused` where it used to see `H_t`. Shape identical. Nothing else changes.**

---

## 3. What to implement — component by component

All components are adapted from `vla/memory_vla.py` in the MemoryVLA repo. I'll flag where we deviate and why.

### 3.1 TimestepEmbedder

Sinusoidal encoding of timestep (for temporal positional encoding on bank keys). Standard DiT-style timestep embedding.

```python
class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of integer timesteps, projected to hidden_size."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            next(self.mlp.parameters()).dtype
        )
        return self.mlp(t_freq)
```

**Usage note:** `frequency_embedding_size=hidden_size // 4` in MemoryVLA's config. Keep that.

### 3.2 CrossTransformerBlock

One transformer block with cross-attention (query = current, K/V = bank) followed by an FFN. Post-norm with residuals. Two of these are stacked in MemoryVLA's retrieval.

```python
class CrossTransformerBlock(nn.Module):
    """Cross-attention + FFN block, post-norm style.

    Adapted from MemoryVLA's vla/memory_vla.py:71. Uses scaled_dot_product_attention
    rather than nn.MultiheadAttention for Flash-Attention compatibility.
    """
    def __init__(self, feature_dim: int, ffn_mult: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(feature_dim * ffn_mult, feature_dim),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query: torch.Tensor,  # (B, N, D)   current VLM output sequence
        k: torch.Tensor,      # (B, M, D)   flattened bank keys (with temporal PE added)
        v: torch.Tensor,      # (B, M, D)   flattened bank values (no temporal PE)
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = self.attn_norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + ffn_out)
```

**Deviations from MemoryVLA:**
- No multi-head attention in the naive sense — `scaled_dot_product_attention` will handle heads if you reshape Q/K/V. For a first pass, *use single-head* (don't reshape) to keep it simple; multi-head is an easy upgrade later if you want to match MemoryVLA more faithfully.

### 3.3 GateFusion

Learned per-token sigmoid gate over concatenated `[current, retrieved]`. Matches MemoryVLA's `GateFusion` exactly.

```python
class GateFusion(nn.Module):
    """Learned per-token sigmoid gate.

    Adapted from MemoryVLA's vla/memory_vla.py:139. The small-std init produces
    gate values near sigmoid(0) = 0.5 at initialization, giving roughly equal
    weighting of current and retrieved memory at step 0. We deliberately do NOT
    zero-init here (see comment in spec section 5.2).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)

    def forward(self, x_current: torch.Tensor, x_retrieved: torch.Tensor) -> torch.Tensor:
        # x_current, x_retrieved: (B, N, D)
        scale = torch.sigmoid(self.proj(torch.cat([x_current, x_retrieved], dim=-1)))
        # NOTE: MemoryVLA's convention is scale*x1 + (1-scale)*x2 where x1=current, x2=retrieved.
        # We follow the same convention: gate near 1 → stay with current; gate near 0 → use retrieved.
        return scale * x_current + (1 - scale) * x_retrieved
```

**Do NOT change the init.** MemoryVLA uses `std=1e-3`, not zero-init. This means at step 0 the model is not identical to vanilla SmolVLA (the gate lets retrieved memory in at ~50%). This is intentional on MemoryVLA's part and we match it. If we later want a cleaner "identical at step 0" ablation we can add a flag, but match the paper first.

### 3.4 FullSeqMemBank — the main component

This is our adaptation of `CogMemBank`. Same structure, but it stores and retrieves over the **full VLM output sequence** (length L, not length 1).

```python
class FullSeqMemBank(nn.Module):
    """Full-sequence cognitive memory bank for SmolVLA.

    Stores the full VLM output sequence per timestep (not just a pooled token).
    Retrieves via cross-attention with temporal positional encoding. Fuses via
    learned sigmoid gate. Consolidates via token-merge (most-similar adjacent
    entries merged) when capacity is reached.

    Adapted from MemoryVLA's CogMemBank (vla/memory_vla.py:158). The structural
    differences from CogMemBank are:
      - CogMemBank stores N=1 token per timestep (EOS-pooled cognitive token)
      - FullSeqMemBank stores N=L tokens per timestep (full VLM output sequence)

    The inner machinery (ring buffer, retrieval, gating, consolidation) is
    unchanged. The `process_batch` logic is *already* generic over N; we just
    pass a larger N.
    """
    def __init__(
        self,
        token_size: int,                 # VLM hidden size (SmolLM2 d_model)
        mem_length: int = 16,            # max entries in bank (per episode)
        retrieval_layers: int = 2,       # stacked cross-transformer blocks
        use_timestep_pe: bool = True,
        consolidate_type: str = "tome",  # "tome" or "fifo"
        update_fused: bool = False,      # if False, store raw (pre-fusion) tokens
        dataloader_type: str = "group",  # "group" or "stream" — see §6
        group_size: int = 1,             # only used if dataloader_type="group"
    ):
        super().__init__()
        assert consolidate_type in ("tome", "fifo")
        assert dataloader_type in ("group", "stream")

        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        self.dataloader_type = dataloader_type
        self.group_size = group_size

        self.retrieval_blocks = nn.ModuleList(
            [CrossTransformerBlock(token_size) for _ in range(retrieval_layers)]
        )
        self.gate_fusion = GateFusion(token_size)

        if use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(
                token_size, frequency_embedding_size=token_size // 4
            )
        else:
            self.timestep_encoder = None

        self.reset()

    # -- bank lifecycle ---------------------------------------------------

    def reset(self):
        """Clear all episode banks. Call at start of eval or between training
        iterations if using group-mode dataloader."""
        self.bank: Dict[Any, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.eid_stream = None  # only used in stream mode

    def clear_episode(self, episode_id):
        self.bank.pop(episode_id, None)

    # -- consolidation ----------------------------------------------------

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id):
        """Merge the adjacent-in-time pair of bank entries with highest mean
        cosine similarity. Adapted from MemoryVLA's _consolidate_with_token_merge.

        NOTE: With full-sequence storage, each bank entry is a [L, D] tensor.
        We flatten to [L*D] for cosine similarity — same as MemoryVLA's code
        (which uses .flatten(1)). This gives a "sequence-average" similarity
        rather than per-token similarity.
        """
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]  # each: (L, D)

        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(0)  # (L*D,)  — NOTE: MemoryVLA uses flatten(1)
            f2 = feats[i + 1].flatten(0)
            # cosine similarity between two flat vectors
            sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=1).item()
            sims.append(sim)

        idx_max = int(torch.tensor(sims).argmax().item())
        t_i, feat_i = bank[idx_max]
        t_j, feat_j = bank[idx_max + 1]
        fused_feat = 0.5 * (feat_i + feat_j)

        bank[idx_max] = (t_i, fused_feat.detach().clone())
        bank.pop(idx_max + 1)

    @torch.no_grad()
    def _memory_consolidate(
        self,
        episode_id,
        feat: torch.Tensor,       # (L, D)
        timestep: Optional[int],
    ):
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length :]
            elif self.consolidate_type == "tome":
                self._consolidate_with_token_merge(episode_id)

    # -- main call --------------------------------------------------------

    def process_batch(
        self,
        tokens: torch.Tensor,          # (B, L, D)   VLM output sequences
        episode_ids,                   # length-B list/array of hashable episode ids
        timesteps,                     # length-B list/array of int timesteps
    ) -> torch.Tensor:
        """For each item in the batch: retrieve from its episode's bank, fuse
        with current tokens, write back to bank, return fused tokens.

        This follows MemoryVLA's process_batch (vla/memory_vla.py:253) exactly
        except that tokens has N=L (full sequence) instead of N=1.
        """
        B, L, D = tokens.shape
        outputs = []

        # --- batch-boundary episode management (training only) ---
        if self.training:
            if self.dataloader_type == "group":
                # Group mode: each batch is a self-contained set of episodes.
                # Clear the whole bank at the start of each batch.
                self.bank.clear()
                self.eid_stream = None
            elif self.dataloader_type == "stream":
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            eid = episode_ids[i]

            # per-item episode boundary handling
            if self.training:
                if self.dataloader_type == "group":
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                elif self.dataloader_type == "stream":
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            # --- 1. retrieval ---
            working_mem = tokens[i].unsqueeze(0)  # (1, L, D)

            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]                # list of (L, D)
                episode_mem = torch.stack(hist_feats, dim=0)           # (T, L, D)
                T = episode_mem.shape[0]
                episode_mem_flat = episode_mem.reshape(T * L, D).unsqueeze(0)  # (1, T*L, D)

                if self.use_timestep_pe:
                    hist_ts = torch.tensor(
                        [t for t, _ in hist], device=working_mem.device, dtype=torch.long
                    )
                    pe = self.timestep_encoder(hist_ts).unsqueeze(0)   # (1, T, D)
                    pe = pe.repeat_interleave(L, dim=1)                # (1, T*L, D)
                else:
                    pe = torch.zeros_like(episode_mem_flat)

                query = working_mem
                for block in self.retrieval_blocks:
                    # keys = memory + PE, values = memory (no PE on values)
                    query = block(query, episode_mem_flat + pe, episode_mem_flat)
                retrieved = query                                       # (1, L, D)
            else:
                # empty bank → retrieved = working_mem (so gate fusion is a no-op
                # blend of working_mem with itself, and we rely on the gate MLP
                # having been trained to either way handle this; in practice at
                # t=0 the gate outputs something but the output equals working_mem.)
                retrieved = working_mem

            # --- 2. fusion ---
            fused = self.gate_fusion(working_mem, retrieved)            # (1, L, D)
            outputs.append(fused)

            # --- 3. write to bank + consolidate ---
            ts_i = timesteps[i] if self.use_timestep_pe else None
            to_store = fused.squeeze(0) if self.update_fused else tokens[i]
            self._memory_consolidate(eid, to_store, ts_i)

        return torch.cat(outputs, dim=0)  # (B, L, D)
```

**Deviations from `CogMemBank`, annotated:**

1. **`tokens` shape is `(B, L, D)` not `(B, 1, D)`.** The only structural difference. The code is already generic over token count; we just pass the full sequence.
2. **`flatten(0)` in consolidation instead of `flatten(1)`.** MemoryVLA's consolidation had a batch dim; ours is per-item so we flatten the whole `(L, D)` to `(L*D,)` for cosine similarity. Semantically equivalent.
3. **`update_fused=False` default.** Matches MemoryVLA's default. We store the raw VLM output in the bank, not the memory-fused version. This prevents error compounding.
4. **No `BottleneckSE`.** That module was MemoryVLA's perceptual-side SE-bottleneck for compressing the 256-token visual grid. We're single-stream cognitive, so there's no 2D spatial structure to compress. It would be inappropriate here.

### 3.5 Integration — the only change to SmolVLA's model

Find where SmolVLA's VLM output (hidden states of SmolLM2) is passed to the action expert. In the SmolVLA codebase this should be one call site.

Before:
```python
vlm_output = self.vlm(images, instruction, state)          # (B, L, D)
action = self.action_expert(vlm_output, noisy_actions, tau)
```

After:
```python
vlm_output = self.vlm(images, instruction, state)          # (B, L, D)
if self.use_memory:
    vlm_output = self.mem_bank.process_batch(
        tokens=vlm_output,
        episode_ids=batch["episode_ids"],
        timesteps=batch["timesteps"],
    )
action = self.action_expert(vlm_output, noisy_actions, tau)
```

That's the entire architectural integration. One call site.

---

## 4. Dataloader requirements — THIS IS CRITICAL

The memory module only works if training batches contain **contiguous episode subsequences**. If you shuffle frames randomly, the bank will see `t=5` from episode A followed by `t=17` from episode B followed by `t=3` from episode A, and the temporal structure is destroyed.

Two viable dataloading modes (matches MemoryVLA):

### Mode "group" (simpler, recommended to start)

Each batch of size `B` is organized as `G` episodes × `group_size` contiguous frames per episode, with `B = G * group_size`. All frames from the same episode appear consecutively in the batch. The bank is cleared at the start of each batch.

Example with `B=8, group_size=4`:
```
batch positions:  [0, 1, 2, 3, 4, 5, 6, 7]
episode ids:      [A, A, A, A, B, B, B, B]
timesteps:        [0, 1, 2, 3, 0, 1, 2, 3]   (or whatever contiguous window)
```

**Dataloader must emit `episode_ids` and `timesteps` alongside the batch.** These are numpy arrays of length B.

### Mode "stream" (harder, more efficient)

Batches are contiguous chunks of a single streamed trajectory. Bank is cleared when the episode id changes. More memory-efficient but requires streaming dataloader support.

**Start with `group` mode. It's correct and simple.**

### Action items for dataloader

1. Identify SmolVLA's LIBERO dataloader path. Likely in `lerobot/` or wherever the RLDS/LIBERO conversion lives.
2. Modify it to:
   - Return episode-grouped batches
   - Emit `episode_ids` (int or string per frame)
   - Emit `timesteps` (int per frame, monotonically increasing within an episode)
3. Reasonable starting config: `B=64, group_size=8, G=8`. Gives 8 episodes per batch × 8 frames each.
4. The `group_size` should be ≥ `mem_length` (otherwise the bank never fills during training, so the model never learns to use "fuller" historical context). With `mem_length=16` and `group_size=8` this is violated; either increase `group_size` to 16+ or accept that the model trains with banks of up to 8 entries. For a first pass, set `mem_length=8` and `group_size=8` to match.

**Important trap:** if you accidentally train with shuffled frames, training will likely still produce a model, but it will have learned to ignore the memory signal (because during training the memory signal was noise). Then at eval time, memory is useless or actively harmful. You will not notice this from loss curves. Verify the dataloader is grouping correctly before kicking off training.

---

## 5. Training

### 5.1 What's trainable

- **VLM (SigLIP + SmolLM2): frozen.** Same as vanilla SmolVLA ablation protocol. No changes.
- **Action expert: trained from scratch.** Same as vanilla ablation.
- **Memory bank (retrieval blocks + gate + timestep embedder): trained from scratch.**

Total new trainable params are small: ~2 retrieval layers × (4 × D^2 for Q/K/V/FFN projections) + gate MLP + timestep embedder. For SmolLM2-360M (`D=960`): roughly 15–20M new params. That's ~4% of SmolVLA's 450M. Record this exactly when you implement — we'll want it for the paper.

### 5.2 On zero-init (deferred)

The MemoryVLA GateFusion uses `std=1e-3` not true zero-init. This means at step 0, `scale ≈ sigmoid(~0) = 0.5`, so `fused ≈ 0.5 * current + 0.5 * retrieved`. When the bank is empty (episode start), `retrieved = current`, so `fused = current` regardless. When the bank has content, the memory signal enters at ~50% weight from the start.

**We match this.** A true zero-init (so fused = current at step 0, model starts identical to vanilla SmolVLA) would be a clean ablation but is a deviation from the published method. Park for later if we want it.

### 5.3 Hyperparameters

Match the existing SmolVLA fine-tuning recipe. Memory-specific hyperparams:

| Param | Default | Rationale |
|---|---|---|
| `mem_length` | 8 | ≤ `group_size`, so bank fills during training. |
| `retrieval_layers` | 2 | Matches MemoryVLA. |
| `use_timestep_pe` | True | Matches MemoryVLA. |
| `consolidate_type` | "tome" | Matches MemoryVLA's best ablation result. |
| `update_fused` | False | Matches MemoryVLA default. |
| `dataloader_type` | "group" | Simpler than stream. |
| `group_size` | 8 | Must be ≥ mem_length. |

### 5.4 LR and optimizer

Memory module uses the same optimizer as the action expert (AdamW, same betas). If SmolVLA uses a param-group split (different LR for different submodules), add the memory module as a new param group with the same LR as the action expert. Do NOT give it the (lower) LR used for the VLM — it's being trained from scratch, not fine-tuned.

---

## 6. Inference

### 6.1 Episode boundary handling

At the start of each LIBERO evaluation rollout, **you must call `mem_bank.reset()` or equivalently clear the episode from the bank**. Otherwise you'll retrieve stale memory from the previous rollout.

The cleanest way:
```python
def eval_rollout(env, model, task):
    model.mem_bank.reset()   # <-- REQUIRED
    obs = env.reset()
    for t in range(max_steps):
        action = model.predict_action(
            obs, task,
            episode_id=0,     # arbitrary, same for all frames in this rollout
            timestep=t,
        )
        obs, _, done, _ = env.step(action)
        if done:
            break
```

### 6.2 Action expert call

Unchanged. The action expert receives `H_fused` in place of `H_t`. Same shape, same interface, same flow-matching loop.

---

## 7. Testing / verification steps

Please run these in order and only proceed to training once they all pass.

### 7.1 Shape test

```python
# Synthetic input
B, L, D = 2, 128, 960
tokens = torch.randn(B, L, D)
bank = FullSeqMemBank(token_size=D, mem_length=4, dataloader_type="group", group_size=B)

# Empty bank
out1 = bank.process_batch(tokens, episode_ids=[0, 0], timesteps=[0, 1])
assert out1.shape == (B, L, D), f"Got {out1.shape}"

# Bank should now have 2 entries
assert len(bank.bank[0]) == 2
```

### 7.2 Episode isolation test

```python
bank.reset()
# Episode 0 frames
bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[0])
bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[1])
assert len(bank.bank[0]) == 2

# Episode 1 should have its own bank
bank.process_batch(torch.randn(1, L, D), episode_ids=[1], timesteps=[0])
assert len(bank.bank[1]) == 1
assert len(bank.bank[0]) == 2  # episode 0 unchanged
```

### 7.3 Consolidation test

```python
bank = FullSeqMemBank(
    token_size=D, mem_length=3, consolidate_type="tome",
    dataloader_type="stream",
)
bank.reset()
# Write 5 entries, bank should cap at 3 via token-merge
for t in range(5):
    bank.process_batch(torch.randn(1, L, D), episode_ids=[0], timesteps=[t])
assert len(bank.bank[0]) == 3, f"Got {len(bank.bank[0])}"
```

### 7.4 Gradient test

```python
bank = FullSeqMemBank(token_size=D, mem_length=4, dataloader_type="group", group_size=2)
bank.train()
tokens = torch.randn(2, L, D, requires_grad=True)
out = bank.process_batch(tokens, episode_ids=[0, 0], timesteps=[0, 1])
loss = out.sum()
loss.backward()

# Gradient should flow back to tokens
assert tokens.grad is not None
assert tokens.grad.abs().sum() > 0

# Gradient should flow to trainable params
for name, p in bank.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"{name} has no grad"
```

### 7.5 Bank-bypass sanity test

With a cold bank (first frame of episode), `out` should equal `tokens` up to the gate fusion (which at init is ~0.5 blend of current with itself).

```python
bank.reset()
tokens = torch.randn(1, L, D)
out = bank.process_batch(tokens, episode_ids=[0], timesteps=[0])
# On empty bank: retrieved = working_mem, so gate_fusion(working_mem, working_mem)
# should equal working_mem (for any gate value, since both inputs are identical).
assert torch.allclose(out, tokens, atol=1e-5), "Cold-bank pass-through failed"
```

This is a critical test. If it fails, the gate fusion is buggy or the bank isn't handling the empty case correctly.

### 7.6 End-to-end integration test

After integrating with SmolVLA:

1. Set `mem_length=1` (bank only holds current frame).
2. Run 1 training step with a batch of all-same-episode consecutive frames.
3. Compare loss to vanilla SmolVLA baseline with the same batch. Should be in a similar ballpark (within 2×), not wildly different.
4. If loss is orders of magnitude off, something is wrong with integration (most likely: shape mismatch, wrong dtype, or memory module is bypassing/duplicating a residual).

---

## 8. Known risks / gotchas

1. **Batch size-compute scaling.** Full-sequence bank retrieval has attention of shape `[L, T·L]` per batch item. For `L=128, T=8`, that's `[128, 1024]` — fine. If `L=1000` and `T=16`, it's `[1000, 16000]` — 16M attention entries per layer per item. Profile this. If the cross-attention is the bottleneck, that's a signal we should reduce `mem_length` or consider pooling the bank keys.

2. **`episode_ids` must be hashable and stable.** Python ints work. Don't use torch tensors or floats as keys.

3. **`timesteps` must be integers.** The `TimestepEmbedder` expects integer timesteps for the sinusoidal encoding. If your dataloader emits floats, cast to int.

4. **First-episode-in-batch cold start.** In group mode, the first frame of each episode has an empty bank. Test 7.5 verifies this case works.

5. **Memory module trains from scratch — expect slow early convergence.** Retrieval blocks have ~10M params and see gradient only once per training step per batch item. Compared to the action expert which sees 10 gradient steps per batch item (10 denoising steps in flow matching, each contributing to the loss), the memory module may train more slowly. Watch the gate values during training — if they're stuck at 0.5 after 10k steps, the gate hasn't learned to discriminate.

6. **Evaluation determinism.** The bank uses a Python dict keyed by `episode_id`. Dict iteration order shouldn't matter here (we index by key, not iterate) but be aware.

---

## 9. File layout suggestion

```
memory_smolvla/
├── memory/
│   ├── __init__.py
│   ├── blocks.py         # TimestepEmbedder, CrossTransformerBlock, GateFusion
│   └── bank.py           # FullSeqMemBank
├── model/
│   └── memory_smolvla.py # Wraps SmolVLA and injects the memory module
├── tests/
│   └── test_memory.py    # Tests from section 7
└── configs/
    └── memvla_libero.yaml
```

Keep the memory code in its own package so we can swap in alternative memory designs (single-token, dual-stream) later without restructuring.

---

## 10. What to produce

When you're done, I want:

1. All the tests in section 7 passing.
2. One training run on LIBERO-Long (to match the baseline), same recipe as the vanilla ablation but with the memory module wired in.
3. A short markdown report with:
   - Exact trainable parameter count (memory module alone and total)
   - Wall-clock training time vs. vanilla baseline
   - Peak VRAM during training
   - Final success rate on LIBERO-Long
4. The gate value statistics logged throughout training (mean gate value per batch, histogram). We'll want this for the paper — it tells us whether the model is actually learning to use memory or ignoring it.

---

## 11. Questions to flag before starting

Before you start implementing, verify:

1. Where does SmolVLA's VLM output meet the action expert? Expected: one call site in the main model forward. Flag if it's more complicated.
2. What is SmolLM2's hidden size in the SmolVLA variant we're using? (Expected: 960 for SmolLM2-360M, but verify.)
3. What's the actual VLM output sequence length `L` in SmolVLA at 512×512? This determines memory footprint. If it's >1000, we may want to revisit the decision to store the full sequence.
4. Does SmolVLA's existing dataloader already support returning `episode_ids` and `timesteps`? If so, we just need to expose them. If not, that's the first implementation task.

Answer these before writing any memory code.
