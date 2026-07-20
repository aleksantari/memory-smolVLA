# V10A Engineering Implementation Plan — Rev B
## Transient Coconut-Style Continuous Latent Reasoning for Memory-Augmented SmolVLA

**Revision note.** Rev B supersedes Rev A after external technical review. Nine review corrections are adopted as blocking, two adopted with modification, one rejected with rationale. Every change is marked **[RevB]** with the reason inline. The scientific frame (strictly additive, detach-first, falsifiable-by-gate) is unchanged.

---

## TL;DR

- Implement V10A as a strictly additive, detach-first Coconut cycle inserted **between the (unchanged) V9 prefix forward and flow denoising**, pinned against lerobot's `modeling_smolvla.py` / `smolvlm_with_expert.py`.
- **[RevB] The thought pass gets its own forced dispatch path** (VLM self-attention only; never touches the expert branch), **explicitly captures per-pass delta-KV** (the pinned `fill_kv_cache=False` path does NOT write back to the cache), and **appends the visible thought's KV at all 16 layers** (visibility is a mask policy, not a per-layer-cache policy).
- **[RevB] Arm consolidation:** a "query token" and a K=1 Coconut thought are the same computation (the feedback output is dead at K=1). The headline contrast is **K2 − K1**, both trained at **fixed K** with matched saved initialization, to avoid conflating recurrence with a depth-sampling curriculum.
- **[RevB] Identity guarantee is scoped correctly:** K=0 is bit-exact V9 (unit-tested); adapter branches are locally identity at init; but **K>0 is NOT init-identity with V9** (the appended thought key takes softmax mass regardless of zero-init values). The old test #11 is replaced.
- **[RevB] Cached ≡ monolithic training equivalence is a new blocking gate** (Step 0.5) with a decision rule that determines whether the existing V9 run is a valid control or whether all arms — including the V9 control — must train on the cached path.
- Gate the headline claim on **K2 − K1 > +5 pp overall across ≥3 eval seed blocks** with a paired hierarchical bootstrap; Gate 2 is now a **noninferiority** test on libero_10; **Gate 3 (latency Pareto)** is new.

---

## Key Findings (verified implementation facts, corrected)

### lerobot source (pin commit `f7283193`, ~lerobot 0.5.x, `src/lerobot/policies/smolvla/`)

- **`make_att_2d_masks(pad_masks, att_masks)`** computes `cumsum = torch.cumsum(att_masks, dim=1)`, then `att_2d = cumsum[:,None,:] <= cumsum[:,:,None]`, ANDed with the outer product of pad masks. Prefix tokens (`att_mask`=0) form one mutually-visible block; suffix tokens are causal after them. An appended thought receives a strictly higher cumsum, so **no prefix position can ever attend a thought** — the prefix computation is numerically unchanged by adding thoughts (regression-tested).
- **KV cache is a plain `dict[layer_idx] -> {"key_states","value_states"}`**, not a HuggingFace `Cache`. `fill_kv_cache=True` writes the dict; **[RevB — corrected reading]** `fill_kv_cache=False` concatenates cached + current K/V into *local* variables that feed attention, **but never assigns the concatenation back into `past_key_values`**. The cache does not grow implicitly. Any incremental decoding loop must capture and append new K/V itself.
- **[RevB — corrected reading] Layer dispatch:** with `fill_kv_cache=False` and `attention_mode="cross_attn"`, layers not matching `self_attn_every_n_layers` route to `forward_cross_attn_layer`, which fetches the expert layer and **dereferences `inputs_embeds[1]`**. Passing `[z, None]` through the stock dispatch therefore fails (or misroutes z through expert projections) at the first cross-attn layer. The thought pass cannot reuse the stock cached path unmodified.
- **`forward_cross_attn_layer`** re-projects cached VLM K/V into expert space via `expert_layer.self_attn.k_proj/v_proj`; **expert position ids are re-zeroed internally** (shifted to start from 0); `expert_attention_mask` is sliced to the cached key length. This internal re-zeroing is left untouched by V10A; thought visibility at these layers is handled purely by widening the mask and lengthening the cache it slices against.
- **[RevB — made explicit] Training vs inference paths differ in the baseline.** Training (`forward`) concatenates prefix+suffix and runs **one monolithic joint forward** (`fill_kv_cache=False, use_cache=False`). Inference (`sample_actions`) builds a prefix KV cache, then runs cached `denoise_step`s. V10A necessarily trains through the cached decomposition (prefix forward → thought passes → cached expert forward), which the baseline never does during training. This is a potential confound handled by Step 0.5.
- **`denoise_step`** computes `prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]` and `position_ids = prefix_offsets + cumsum(suffix_pad_masks) − 1`. **[RevB]** Positions are therefore per-example *valid-token counts*, not tensor lengths — thought positions must follow the same convention.
- Config for `smolvla_libero`: `num_vlm_layers=16`, `self_attn_every_n_layers=2`, `expert_width_multiplier=0.75`, `attention_mode="cross_attn"`, hidden D=960. VLM frozen; V10A trains expert + memory + new adapters from scratch.
- **[RevB]** SmolVLA scales image/language embeddings before the VLM; new thought inputs must be RMS-matched to the actual residual-stream scale (see `MatchInputRMS`).

### Zero-init precedent — scope corrected

- LLaMA-Adapter (arXiv:2303.16199), ControlNet (ICCV 2023), and ControlVLA (arXiv:2506.16211, App. C) jointly establish that zero-initialized output projections preserve the base computation at init while gradients still flow (∂L/∂W_z depends on the nonzero input activation).
- **[RevB — scope]** This guarantees only **local identity of the adapter branches**: `LatentModeAdapter` contributes exactly 0 at init, and the feedback adapter's residual update is exactly 0 at init. It does **not** make the full K>0 policy identical to V9: the appended thought key/value pair enters the expert's softmax denominator and receives probability mass even if its value contribution were zero. Full K>0 init-identity would require a separate zero-initialized residual attention branch — deliberately out of scope. Since the expert trains from scratch in every arm, **K=0 identity is the control that matters**.

### Feedback adapter — identity claim corrected

- **[RevB]** With `fc2 = 0`, the previously proposed adapter returned `RMSNorm(h)`, not `h` — RMSNorm changes any vector whose RMS isn't already 1. Rev B uses the **residual form** `h + σ(gate)·update` (exactly `h` at init), with scale-matching handled by a separate explicit `MatchInputRMS` step before the next VLM pass. No trailing norm inside the adapter.

### bf16 — correction

- **[RevB]** The Rev A "precision trap" mitigation was overstated for the actual training setup. Under standard PyTorch autocast, parameters and optimizer state live in fp32 and only eligible ops run in bf16; zero-init parameters update normally. Resolution is empirical, not speculative: **assert at optimizer construction that every new trainable parameter has `dtype == torch.float32`** (blocking test `test_thought_params_are_fp32`). Micro-noise is dropped unless a targeted unit test demonstrates a stuck parameter. The gate scalar (init −1.0) receives zero gradient while `fc2 = 0` — this is acceptable and expected: `fc2` learns first, then the gate's gradient turns on.

### Coconut mechanic

- Coconut (Hao et al., arXiv:2412.06769): the last hidden state of the LLM is fed back as the next input embedding directly in continuous space. We adopt the feedback but keep thoughts **transient** (never written to the bank) and train at **fixed depth per arm** rather than Coconut's staged curriculum. **[RevB]** Depth-sampling (`{0:.2, 1:.3, 2:.5}`) is demoted to a later robustness variant because mixing it into the K2 arm while K1 trains fixed would conflate recurrence with a curriculum effect — the review's arm consolidation is adopted, and this confound (which the review did not flag) is closed by matching schedules.

### Statistics

- Binomial SEs at p≈0.5: n=100/suite → ±9.8 pp margin; n=400 overall → ±4.9 pp; paired-difference SE at n=400 with ρ≈0.5 → ~2.5–3.5 pp. Rollouts are clustered in 40 tasks, so i.i.d. formulas are lower bounds; the **paired hierarchical bootstrap** (tasks → episodes, B=10,000, common resample index across arms/conditions) is authoritative. Wilson intervals for base rates (Brown, Cai & DasGupta 2001); IQM reported **supplementary** to plain mean success (rliable, NeurIPS 2021) — **[RevB]** mean success is primary for interpretability.
- **[RevB — scope of claim]** Three eval seed blocks from one checkpoint measure environment-initialization variance only, not training-seed variance. All results are reported as: *"under one matched training initialization, K2 improved paired closed-loop evaluation across ≥3 environment seed blocks."* No general model-family claim without multiple training seeds.
- **[RevB — sign-flip retained, unit clarified]** The sign-flip permutation test runs over **paired task-level success differences** (40 tasks × seed blocks), where it is a valid exact test — not over 3 training runs. Review objection on this point rejected as a unit conflation; the underlying scoping point is absorbed above.

---

## Architecture invariants (new in Rev B — every test and every step must preserve these)

1. **K=0 ≡ V9, bit-exact** (fp32 atol=0), in both the monolithic and cached execution paths.
2. **The bank advances exactly once per query**, for all K. Thought generation never writes, never consolidates.
3. **Thought passes run the VLM self-attention path only.** They never instantiate, execute, or route through the action expert or the cross-attn expert branch.
4. **Thought KV is appended at all 16 layers.** Cache lengths are identical across layers at all times. `final_only` vs `all_visible` is implemented as a *mask/append policy over which thought deltas enter the expert cache*, never as per-layer divergence.
5. **Two position concepts, never derived from each other:**
   - `expert_visible_length = valid_prefix_len + (#visible thoughts)` — governs mask width and cache length seen by the expert;
   - `action_position_offset = valid_prefix_len + K` — governs RoPE positions of the action suffix, because the final thought's KV was encoded at position `valid_prefix_len + K − 1` regardless of how many thoughts are visible.
6. **Detach-first gradient routing:** thought passes read a per-tensor-detached copy of the prefix KV; the expert reads the **live** prefix KV plus the final thought delta. Seed inputs are stop-grad. Thought→memory gradients are structurally impossible; V9 BPTT is untouched.
7. **All positions are per-example valid-token counts** (`prefix_pad_masks.sum(dim=1)`), never tensor lengths.

---

## PLAN

### Step 0 — Environment & compatibility audit (BLOCKING; do first)

1. **Pin lerobot exactly** (version + upstream git SHA). lerobot 0.5.0 moved to Python 3.12 + Transformers v5 with import-path changes; the monkey-patch is version-fragile. Torch/torchvision pinned to the cu128 index (aarch64/GB10).
2. **`scripts/compat_check.py`** asserts: (a) version == pin; (b) patched forward replica matches upstream on random inputs (max-abs diff < 1e-4 bf16 / 1e-6 fp32 on outputs and every `past_key_values[layer]` tensor); (c) the `make_att_2d_masks` cumsum property on a hand-built example; (d) cache is a dict of `[B, L, H, Dh]` tensors; **[RevB]** (e) `fill_kv_cache=False` does **not** mutate the passed `past_key_values` (assert object/content identity before vs after); **[RevB]** (f) calling the stock cached path with `inputs_embeds=[z, None]` raises/misbehaves at the first cross-attn layer — documenting why the forced dispatch exists.
3. CI test `tests/test_compat_upstream.py` so future bumps fail loudly.

### Step 0.5 — Cached ≡ monolithic training equivalence (NEW, BLOCKING) **[RevB]**

V10A trains through the cached decomposition; baseline V9 trains monolithically. Before any Coconut code lands, prove (or disprove) that the two execution paths are equivalent for V9 itself:

```python
def test_cached_v9_matches_monolithic_v9():
    # Same params, batch, actions, noise sample, flow time t.
    loss_m = model.forward_monolithic_v9(batch)     # stock lerobot training forward
    loss_c = model.forward_cached_v9(batch)          # prefix KV build -> cached expert forward
    assert_close(loss_m, loss_c, rtol=TOL_R, atol=TOL_A)
    for name in [expert QKV projections, action_out_proj,
                 memory retrieval layers, fusion gate,
                 reasoning summary head,
                 one earlier-timestep memory write reached via V9 BPTT]:
        assert_gradients_close(grad_m[name], grad_c[name])
```

**Decision rule (pre-registered):**
- **Pass** at tolerance (fp32: rtol 1e-5; bf16 compute: empirically calibrated, documented) → the existing V9 run is a valid Arm 1 control (subject to the recipe/seed match check).
- **Fail** → the existing V9 run is **disqualified**; all arms **including the V9 control** train on the cached path, so the execution path is held constant and any V10 effect is attributable to reasoning, not to the decomposition.

### Step 1 — Refactor for modes (no behavior change)

`policy/feature_extractor.py`:

```python
class ForwardMode(enum.Enum):
    PREFIX = "prefix"          # memory callback ON, write ON, timestep advance ON, adapters OFF
    LATENT_THOUGHT = "thought" # memory callback OFF, write OFF, timestep OFF, latent adapters ON
    FLOW_ACTION = "flow"       # everything OFF, adapters OFF
```

**[RevB] Forced dispatch rule** inside the patched per-layer loop:

```python
if mode is ForwardMode.LATENT_THOUGHT:
    # ALWAYS the VLM self-attention path for the thought token:
    # reads merged (detached prefix + thought) cache, returns this token's delta K/V.
    # Never enters forward_cross_attn_layer; never touches expert weights.
    att_out, delta_kv_l = forward_vlm_token_layer(...)
elif <existing dispatch conditions>:
    ...
```

The memory-callback hook is a no-op unless `mode is PREFIX` (assert-enforced).

`memory/full_seq_bank.py` — `MemoryStepOutput` dataclass (`fused_prefix`, `retrieved_prefix`, `memory_slots`, `gate_values`) as in Rev A; renames only, **no numeric change**. Bit-exactness test `tests/test_v9_bitexact_refactor.py` unchanged (atol=0 on loss, `memory_slots`, `fused_prefix`, `gate_values`).

### Step 2 — Coconut modules in `memory/reasoning.py`

```python
class CoconutThoughtSeed(nn.Module):
    # z0 = e_think + W_seed[ LN(sg(mean(memory_slots))) ; LN(sg(maskmean(retrieved_prefix))) ]
    def __init__(self, d=960):
        self.e_think = nn.Parameter(torch.randn(d) * 0.02)
        self.ln_m = nn.LayerNorm(d); self.ln_r = nn.LayerNorm(d)
        self.W_seed = nn.Linear(2 * d, d)
    def forward(self, memory_slots, retrieved_prefix, r_mask):
        m = self.ln_m(memory_slots.mean(1).detach())               # stop-grad
        r = self.ln_r(masked_mean(retrieved_prefix, r_mask).detach())
        return self.e_think + self.W_seed(torch.cat([m, r], -1))

class CoconutFeedbackAdapter(nn.Module):
    # [RevB] residual form: exactly h at init (fc2 zero => update == 0).
    # No trailing norm; scale handled by MatchInputRMS outside.
    def __init__(self, d=960, hid=1920):
        self.n1 = RMSNorm(d)
        self.fc1 = nn.Linear(d, hid); self.fc2 = nn.Linear(hid, d)
        self.gate = nn.Parameter(torch.tensor(-1.0))               # fp32; sigma(-1)=0.269
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
    def forward(self, h):
        update = self.fc2(F.gelu(self.fc1(self.n1(h))))
        return h + torch.sigmoid(self.gate).to(h.dtype) * update

class MatchInputRMS(nn.Module):
    # [RevB] match thought-token scale to the actual VLM input residual stream.
    def __init__(self):
        self.log_scale_adjustment = nn.Parameter(torch.zeros(()))  # fp32
    def forward(self, thought, reference_tokens, reference_mask):
        ref_rms = masked_rms(reference_tokens.detach(), reference_mask)   # [B]
        normed = thought * torch.rsqrt(thought.square().mean(-1, keepdim=True) + 1e-6)
        return normed * (ref_rms * self.log_scale_adjustment.exp())[:, None, None]

class LatentModeAdapter(nn.Module):
    # rank-32 zero-init bottleneck; [RevB] applied on the THOUGHT ROW,
    # immediately BEFORE the layer's input_layernorm/QKV, layers 12-15 only.
    def __init__(self, d=960, r=32):
        self.down = nn.Linear(d, r, bias=False); self.up = nn.Linear(r, d, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)                             # local identity at init
    def forward(self, h):                                          # thought row only
        return h + self.up(self.down(h))

@dataclass
class ThoughtPassOutput:                                           # [RevB]
    hidden_state: Tensor                       # last-layer hidden of this thought
    delta_kv: dict[int, dict[str, Tensor]]     # exactly one token per layer

@dataclass
class CoconutState:
    thought_prefix_cache: dict     # per-tensor-detached prefix KV
    thought_deltas: list           # per-pass delta_kv dicts
    processed_hidden_states: list  # [RevB] hidden states of processed thoughts
    feedback_embeddings: list      # [RevB] adapter outputs (inputs to NEXT pass)
```

**[RevB] Adapter placement rationale:** post-residual placement made the layer-15 adapter dead for the exposed KV at K=1 (it would only shape the final hidden state, which the expert never sees under KV exposure). Pre-QKV placement on the thought row means every adapter shapes that layer's exposed K/V directly, while zero-init still preserves the frozen computation at init.

**[RevB] Naming discipline:** the action-visible representation is the final pass's **delta_kv + processed hidden state**. The last feedback-adapter output is an *unprocessed hypothetical next thought* — logged, never exposed, never called `z_final`.

Unit tests (`tests/test_coconut_modules.py`): shapes; `LatentModeAdapter(h) == h` exactly at init; `CoconutFeedbackAdapter(h) == h` exactly at init; gradient isolation (detached cache → `memory_slots.grad` empty, adapter/seed grads present); **[RevB]** gate-behavior test sets `fc2` to a known nonzero matrix first (with `fc2=0` the gate test is vacuous); `MatchInputRMS` maps arbitrary-scale inputs to reference RMS at init.

### Step 3 — Orchestration in `policy/memory_smolvla.py`

Flow: `build_memory_augmented_prefix()` (V9, unchanged, `mode=PREFIX`; **the bank advances here, exactly once per query**) → `run_coconut_reasoning()` → cached expert forward (training) / `sample_actions` denoise loop (inference).

**[RevB] Revised core algorithm** (explicit delta-KV, per-example positions, decoupled offsets):

```python
prefix_out = build_memory_augmented_prefix(...)   # V9, non-detached prefix_kv, MemoryStepOutput
if K == 0 or not cfg.coconut.enabled:
    return run_v9_action_path(...)                # bit-exact V9 (path per Step 0.5 decision)

# --- thought phase (detached world) ---
running_cache = {l: {k: v.detach() for k, v in d.items()}      # detach EACH TENSOR
                 for l, d in prefix_out.prefix_kv.items()}
valid_prefix_len = prefix_out.prefix_pad_mask.long().sum(dim=1)          # [B]
z = thought_seed(prefix_out.memory_slots, prefix_out.retrieved_prefix, r_mask)
z = match_input_rms(z, prefix_out.prefix_embs, prefix_out.prefix_pad_mask)

thought_deltas, hiddens = [], []
for k in range(K):
    out = vlm_thought_pass(                       # ThoughtPassOutput
        token=z,
        past_key_values=running_cache,
        position_ids=valid_prefix_len[:, None] + k,               # per-example
        key_padding_mask=cat([prefix_out.prefix_pad_mask,
                              ones(B, k + 1, bool)], dim=1)[:, None, :],
        mode=ForwardMode.LATENT_THOUGHT,          # forced VLM self-attn dispatch
        adapter_layers=cfg.coconut.adapter_layers,
    )
    thought_deltas.append(out.delta_kv)
    hiddens.append(out.hidden_state)
    running_cache = append_delta_kv(running_cache, out.delta_kv)  # explicit growth
    if k < K - 1:                                 # feedback only BETWEEN passes
        z = feedback_adapter(out.hidden_state)
        z = match_input_rms(z, prefix_out.prefix_embs, prefix_out.prefix_pad_mask)

# --- expert phase (live world) ---
visible = thought_deltas[-1] if cfg.coconut.expert_visibility == "final_only" \
          else merge_deltas(thought_deltas)
expert_cache = append_delta_kv(prefix_out.prefix_kv, visible)     # ALL 16 layers; LIVE prefix
expert_prefix_mask = cat([prefix_out.prefix_pad_mask,
                          ones(B, n_visible, bool)], dim=1)
action_position_offset = valid_prefix_len + K                     # NOT visible length
actions = run_cached_action_expert(expert_cache, expert_prefix_mask,
                                   action_position_offset, ...)
```

with

```python
def append_delta_kv(base, delta):
    return {l: {"key_states":   torch.cat([base[l]["key_states"],   delta[l]["key_states"]],   dim=1),
                "value_states": torch.cat([base[l]["value_states"], delta[l]["value_states"]], dim=1)}
            for l in base}
```

**[RevB] Position/offset details:**
- `denoise_step` (and the cached training forward) gains an explicit `action_position_offset` argument: `position_ids = action_position_offset[:, None] + cumsum(suffix_pad_masks, dim=1) − 1`. Under `final_only` with K=2, the visible cache holds `valid_prefix_len + 1` keys, but actions start at `valid_prefix_len + 2` — the two quantities are never derived from one another (Invariant 5).
- Cross-attn expert layers keep their internal expert-position re-zeroing untouched; the thought's visibility there comes from the lengthened cache the layer slices and the mask widened by exactly `n_visible` columns.
- Thought positions and masks are per-example; unit tests cover mixed language lengths, explicit prefix padding, missing-camera masks, and one heavily padded example in the same batch (a fixed-length-only test could pass while real LIBERO batches are positionally wrong).

**Gradient routing:** expert uses the **live** `prefix_kv` for the prefix portion (V9 BPTT preserved) and the final thought delta for the thought column; seed inputs stop-grad + detached thought cache block thought→memory gradients structurally.

**Config (`training/config.py`) [RevB — fixed depth per arm]:**

```yaml
coconut:
  enabled: true
  num_thoughts_train: 2          # FIXED per arm (K1 arm: 1; K2 arm: 2). Depth-sampling
                                 # is a later robustness variant, never mixed into the
                                 # headline arms (curriculum confound).
  num_thoughts_eval: 2           # must equal num_thoughts_train for headline arms; assert
  seed: {use_memory_slots: true, use_retrieved_prefix: true, stop_grad_inputs: true}
  detach_prefix_cache: true
  adapter_layers: [12, 13, 14, 15]
  adapter_placement: pre_qkv     # [RevB]
  adapter_rank: 32
  feedback_hidden: 1920
  feedback_gate_init: -1.0
  match_input_rms: true          # [RevB]
  expert_visibility: final_only  # {final_only, all_visible}
  write_thoughts_to_memory: false
  verification_weight: 0.0       # EMA self-verification armed but OFF
```

`--bypass-thoughts` on `scripts/eval_memory_libero.py` forces K=0 at eval regardless of config.

### Step 4 — Probes, monitoring & profiling (read-only, out of the policy gradient)

- Label pipeline in `data/group_loader.py` as Rev A: multi-horizon future proprio (h∈{5,20,50}); heuristic task-phase FSM; target-object id; offline safety predicates.
- Linear probes (`memory/probes.py`) on the **final processed hidden state** and on `memory_slots`; probe inputs detached; probe params in a separate optimizer group; probe loss logged, never added to the policy loss.
- Collapse metrics: thought RMS; `cos(z_k, z_{k+1})`; effective rank of stacked thought states; distance to the vocab-embedding manifold; per-layer expert attention mass on the thought column. EMA self-verification wired but off.
- **[RevB] RMS dashboard (every `log_every_n` steps):** RMS of thought seed, feedback token, language tokens, state token, fused layer-15 representation, plus the seed/language and feedback/language ratios. Alert outside [0.5, 2.0]. Do not discover a 30× scale mismatch six days into a run.
- **[RevB] Latency profiling hooks** (feeds Gate 3): V9 prefix time, per-thought-token time, flow-expert time, end-to-end policy query time, peak memory, async queue underruns.

### Step 5 — Full pytest checklist (`tests/test_v10a.py`; all V9 tests must still pass)

Carried over from Rev A:
1. `test_bank_advances_once_per_query[K=0,1,2]`
2. `test_thought_gen_never_writes` (spy on `bank.write`, `ToMe.consolidate`)
3. `test_k0_is_v9_exact` (atol=0 fp32; both execution paths per Step 0.5)
4. `test_final_thought_changes_with_bank`
5. `test_final_thought_changes_with_instruction`
6. `test_final_only_hides_intermediates` (expert cache/mask contain exactly the final delta)
7. `test_detach_blocks_thought_to_memory_grads`
8. `test_reset_clears_thoughts`
9. `test_adapters_inactive_in_prefix_and_flow`
10. `test_eval_K_matches_config` (+ `--bypass-thoughts` forces K=0)

**[RevB] Replacing old test #11 (`test_zero_init_identity`, which asserted a false property):**
11. `test_latent_adapters_are_local_identity_at_init`
12. `test_feedback_residual_is_zero_at_init` (adapter output == input exactly)

**[RevB] New blocking tests:**
13. `test_latent_mode_never_enters_expert_cross_branch` (spy/assert on dispatch)
14. `test_fill_false_does_not_implicitly_update_cache`
15. `test_delta_kv_is_exactly_one_token_per_layer`
16. `test_running_cache_grows_by_one_per_thought`
17. `test_cached_v9_forward_matches_monolithic_v9` (Step 0.5)
18. `test_cached_v9_gradients_match_monolithic_v9` (Step 0.5 param list)
19. `test_variable_length_position_ids` (mixed language lengths, heavy padding, missing cameras)
20. `test_final_only_uses_sparse_action_position_offset` (offset = valid_len + K, not visible length)
21. `test_cache_lengths_match_masks_at_every_layer` (Invariant 4)
22. `test_shared_parameter_initialization_matches_across_arms`
23. `test_optimizer_contains_all_coconut_params_exactly_once`
24. `test_thought_params_are_fp32` (bf16 resolution)
25. `test_thought_dtype_matches_vlm_residual_dtype`
26. `test_gate_modulates_with_nonzero_fc2` (fc2 set nonzero before testing gate)
27. `test_position_ids_and_masks` (cumsum: prefix cannot attend thoughts; mask widened by exactly `n_visible`)

Lint: `ruff` clean; all under `pytest`.

### Step 6 — Smoke protocol, run matrix, evaluation (DGX Spark GB10)

Hardware: GB10, 119 GB unified LPDDR5X, aarch64, cu128; ~9.1 s/step at batch 128 → 60k steps ≈ 6.3–9 days/run. Eval on RTX 5090.

**[RevB] Matched initialization across arms.** New modules change RNG consumption, so a shared global seed no longer guarantees identical shared-module inits. Save one canonical initialization and load it into every arm; use per-component generators for (expert, memory, Coconut, flow-noise, time-sampling, data-loader) streams. Pre-flight: `assert_shared_parameters_identical(v9, k1, k2)` (test #22).

**Smoke test first** (each arm; NOT for ranking): 2k steps + 20-rollout sanity eval. Pass: no NaNs; K=0 bit-exact green; bank once/query; `cos(z1, z2) < 0.99`; probes above chance; **[RevB]** cache grows by exactly one per thought; RMS ratios in [0.5, 2.0]; per-thought latency recorded.

**[RevB] Run matrix — consolidated** (query token ≡ K=1: the feedback output is dead computation at K=1, so the old Arm 2/Arm 3 distinction was vacuous):

| Arm | Name | Meaning | K (train, FIXED) | K (eval) | Notes |
|---|---|---|---|---|---|
| 1 | V9 control | no thought | — | — | existing run **iff** Step 0.5 passes AND recipe/seed protocol matches; else cached-V9 retrain |
| 2 | V10A-K1 | single nonrecurrent latent planning token | 1 | 1 | the nonrecurrent baseline |
| 3 | V10A-K2 | one recurrent refinement (two processed thoughts) | 2 | 2 | headline arm |
| 4 (opt) | V10A-K2-allvisible / width-4 | ablations | 2 | 2 | phase 2, budget permitting |

All arms: identical from-scratch recipe from `lerobot/smolvla_base`, 60k steps, batch 128 (16 groups × 8 contiguous frames), AdamW lr 1e-4 cosine, bf16 autocast over fp32 params, same GroupedEpisodeLoader, same image-flip correction, `bptt_memory: true`, same saved initialization. Sequence: smoke all → Arm 1 → Arm 2 → Arm 3 → Arm 4 if budget. Supervisor: per-arm config, checkpoint every 5k, auto-resume, NaN watchdog.

**Evaluation:** 2×2 grid {mem on/off} × {thoughts on/off} × ≥3 eval seed blocks, ± ensemble; LIBERO 4 suites × 10 tasks × 10 episodes = 400 rollouts/condition:

```
python scripts/eval_memory_libero.py --ckpt <arm> --seed {s1,s2,s3} \
   [--bypass-memory] [--bypass-thoughts] [--ensemble]
```

Intervention suite (exploratory, not gating): zero final thought; replace final with thought 1; shuffle final thoughts across episodes; Gaussian-noise thought; K=1 stop-early from the K2 arm; untrained K=3; memory×reasoning cross-interventions (shuffle bank order, empty bank mid-episode).

**Stats (`scripts/analyze_bootstrap.py`):** paired hierarchical bootstrap, B=10,000, tasks→episodes with a common (task, seed) resample index read across all arms/conditions; per-suite + overall success with 95% percentile CIs; paired deltas; **task-level** sign-flip permutation p; interaction term I = (SR[m1,t1]−SR[m1,t0]) − (SR[m0,t1]−SR[m0,t0]) with bootstrap CI (assessed, not gating); Wilson intervals for base rates; plain mean primary, IQM supplementary.

### Step 7 — Decision checkpoint (read against gates)

Headline: **K2 − K1** overall (paired bootstrap). Secondary: K1 − V9.

- **K2 ≈ K1** (within noise): recurrence not demonstrated → keep the single latent token only if K1 > V9 with CI excluding 0; else stop.
- **All ≈ V9**: try the injected-fusion coupling variant (couple thoughts to the memory-fusion seam) before abandoning.
- **K2 > K1 > V9** (monotone in depth): recurrence real → proceed to width/depth/interaction phase (Arm 4, width-4, higher K) — subject to Gate 3.

---

## Acceptance-gate table **[RevB]**

| Gate | Trigger | Pass condition | Consequence if fail |
|---|---|---|---|
| **Gate 0 (smoke)** | after 2k-step smoke | no NaNs; K=0 bit-exact; bank once/query for all K; cache grows by one per thought; `cos(z1,z2)<0.99`; probes above chance; RMS ratios in [0.5, 2.0] | fix mode/graph/scale bug before any full run |
| **Gate 0.5 (path equivalence)** | before Coconut code lands | cached-V9 == monolithic-V9 in loss AND listed gradients at tolerance | existing V9 run disqualified; ALL arms (incl. control) train cached |
| **Gate 1 (headline)** | after full runs + eval | **K2 − K1 > +5 pp overall** across ≥3 eval seed blocks AND paired hierarchical bootstrap CI excludes 0 AND task-level sign-flip p<0.05. Claim scoped to one matched training initialization. Interaction assessed, not gating | decision-tree branch |
| **Gate 2 (noninferiority)** | after full runs + eval | **LowerCI(SR_K2 − SR_V9) > −5 pp on libero_10** (predeclared margin; a −7 pp point estimate cannot pass via a wide interval) | reject V10A even if Gate 1 passes |
| **Gate 3 (latency Pareto)** | after full runs + profiling | K2 remains on the success–latency Pareto frontier vs V9/K1 (e.g., reject if +30% end-to-end latency buys an uncertain ~1 pp) | reject or demote to offline-only variant |

---

## Risk / pitfall list (for the coding agent)

- **[RevB] Implicit cache growth does not exist.** `fill_kv_cache=False` never writes back. If the loop "works" without `append_delta_kv`, every pass is silently attending only the prefix — the single most likely silent failure. Test #16 is the tripwire.
- **[RevB] Dispatch leak.** Any code path where a `LATENT_THOUGHT` token reaches `forward_cross_attn_layer` (or the expert branch dereferences `inputs_embeds[1] = None`) is a hard bug. Test #13.
- **[RevB] Position semantics.** All offsets from `pad_mask.sum(dim=1)`, per example. `action_position_offset = valid_prefix_len + K` even under `final_only`. Tests #19, #20.
- **[RevB] Per-layer cache divergence.** Never append the thought at a subset of layers; `final_only` is a mask/append policy over deltas, applied uniformly at all 16 layers. Test #21.
- **[RevB] Adapter placement.** Pre-`input_layernorm` on the thought row, layers 12–15, `LATENT_THOUGHT` only; never touch prefix/action rows. Post-residual placement leaves layer-15's adapter without action gradient at K=1.
- **[RevB] Scale mismatch.** `MatchInputRMS` on seed and feedback outputs; RMS dashboard from step 0 of training.
- **[RevB] dtype.** Assert fp32 on all new trainable params at optimizer construction (test #24). No micro-noise. Gate's initial zero-grad while fc2=0 is expected, not a bug.
- **Cache detachment:** detach each tensor value (`v["key_states"].detach()`), never the dict; a live dict retains the full prefix graph through every thought and re-opens the forbidden thought→memory gradient route.
- **Eval-time K must equal the arm's fixed training K** (headline arms); assert. `--bypass-thoughts` is the only sanctioned K override.
- **ToMe `no_grad` boundary untouched:** thoughts never enter consolidation; the refactor must not move the boundary.
- **Prefix numerically unchanged:** cumsum masking guarantees prefix positions never attend thoughts; regression-tested (test #27).

## What NOT to do

- Do NOT write thoughts to the memory bank (transient only).
- Do NOT add any new policy loss (flow matching + 0.1·PTP only; probes/verification detached/off).
- Do NOT warm-start canonical runs; do NOT unfreeze the VLM (adapters are the only new VLM-path params).
- Do NOT change any V9 config/hparam (seam layer-15, `mem_length=4`, 8 reasoning tokens, PTP weight 0.1, `bptt_memory`, loader, lr/schedule).
- **[RevB]** Do NOT rely on `fill_kv_cache=False` to grow the cache.
- **[RevB]** Do NOT claim or test K>0 init-identity with V9; K=0 identity is the control.
- **[RevB]** Do NOT mix depth-sampling into the headline arms; matched fixed K only.
- **[RevB]** Do NOT reuse the existing V9 run if Step 0.5 fails or its seed protocol differs.
- **[RevB]** Do NOT present results as a model-family claim from one training initialization.

## Recommendations (staged)

1. **Step 0 today**; block everything on the compat test, including the two new assertions (non-writeback; cross-attn dereference).
2. **Step 0.5 immediately after** — its outcome decides the Arm 1 strategy and possibly adds a cached-V9 retrain to the queue; know this before scheduling GPU-days.
3. Land Steps 1–2 behind K=0-bit-exact + local-identity tests; do not proceed until V9 is provably unchanged.
4. Land Step 3 + Step 5 tests; run the 2k smoke on **Arm 2 (K1) first** (cheapest real thought path) to shake out dispatch/cache/position bugs; watch the RMS dashboard from step 0.
5. Save the canonical initialization; run `assert_shared_parameters_identical`; queue Arm 1 (or cached retrain) → Arm 2 → Arm 3.
6. Evaluate on the 2×2 × ≥3-seed grid; run the bootstrap; read against Gates 1–3.
7. **Thresholds that change the plan:** K2−K1 < 5 pp → no recurrence claim; libero_10 noninferiority fail → reject; latency off-Pareto → reject/demote; collapse metrics (`cos→1`, effective rank→1, vocab-manifold collapse) → arm EMA self-verification (`verification_weight>0`) and/or reduce K.

## Caveats

- Step 0 must record the exact lerobot minor version + SHA; the monkey-patch is version-fragile (0.5.0: Python 3.12, Transformers v5, import-path changes).
- If Step 0.5 fails, the effective run budget grows by one full training run (cached-V9 control); plan GPU scheduling for both branches now.
- LIBERO suites are often near-saturated; 10 episodes/task (400 rollouts) is below the common 50/task protocol and widens CIs — if a headline delta lands near ±5 pp, raise episodes/task before concluding.
- All conclusions are scoped to one matched training initialization; generalizing to the model family requires ≥3 independent training seeds per arm (out of current budget).
- Whether transient recurrence helps at all on LIBERO remains an open empirical question; the design stays falsifiable by Gate 1 and the intervention suite.
