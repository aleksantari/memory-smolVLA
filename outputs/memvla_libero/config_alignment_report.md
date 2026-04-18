# memvla_libero ↔ baseline_v2 config alignment report

Produced 2026-04-18 from a 2-step smoke run of
`configs/memvla_libero.yaml` after applying the drift fixes in
`claude_code_config_audit_prompt.md` (Tasks 1–10).

Artifacts:
- Resolved memvla config: [checkpoints/memvla_libero/resolved_policy_config.json](../../checkpoints/memvla_libero/resolved_policy_config.json) (regenerated at every trainer init)
- Baseline ref: [outputs/libero_baseline_v2/checkpoints/last/pretrained_model/train_config.json](../libero_baseline_v2/checkpoints/last/pretrained_model/train_config.json)

---

## 1. SmolVLA policy-field parity table

Non-memory `SmolVLAConfig` fields compared head-to-head. Every field is
resolved at load time (hub config → yaml `policy.overrides:` → saved).

| Field | Baseline | memvla resolved | Match |
|---|---|---|---|
| `chunk_size` | 50 | 50 | ✓ |
| `n_action_steps` | 10 | 10 (override) | ✓ |
| `n_obs_steps` | 1 | 1 | ✓ |
| `resize_imgs_with_padding` | [512, 512] | [512, 512] | ✓ |
| `tokenizer_max_length` | 48 | 48 | ✓ |
| `max_state_dim` | 32 | 32 | ✓ |
| `max_action_dim` | 32 | 32 | ✓ |
| `min_period` | 0.004 | 0.004 | ✓ |
| `max_period` | 4.0 | 4.0 | ✓ |
| `attention_mode` | `cross_attn` | `cross_attn` | ✓ |
| `num_vlm_layers` | 16 | 16 | ✓ |
| `num_expert_layers` | -1 | 0 | cosmetic only — both ≤0 route to default branch in `smolvlm_with_expert.py:100` (identical architecture) |
| `self_attn_every_n_layers` | 2 | 2 | ✓ |
| `expert_width_multiplier` | 0.75 | 0.75 | ✓ |
| `freeze_vision_encoder` | true | true | ✓ |
| `train_expert_only` | true | true | ✓ |
| `train_state_proj` | true | true | ✓ |
| `load_vlm_weights` | true | true | ✓ |
| `vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | same | ✓ |
| `num_steps` (flow-matching) | 10 | 10 | ✓ |
| `use_cache` | true | true | ✓ |
| `normalization_mapping` | {VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD} | same | ✓ |
| `empty_cameras` | 0 | 0 | ✓ |
| `adapt_to_pi_aloha` | false | false | ✓ |
| `use_delta_joint_actions_aloha` | false | false | ✓ |
| `add_image_special_tokens` | false | false | ✓ |
| `prefix_length` | -1 | -1 (override) | ✓ |
| `pad_language_to` | `longest` | `longest` (override) | ✓ |
| `scheduler_decay_steps` | 100000 | 100000 (override) | ✓ (metadata only; trainer uses its own LambdaLR) |
| `optimizer_lr` / `optimizer_betas` / `optimizer_eps` / `optimizer_weight_decay` / `optimizer_grad_clip_norm` / `scheduler_warmup_steps` / `scheduler_decay_lr` | — | — | SmolVLA optimizer preset is inert; we build AdamW directly in the trainer with these exact values from `configs/memvla_libero.yaml` |

### Drifts closed via `policy.overrides:` (see `configs/memvla_libero.yaml`)

- `n_action_steps`: hub default 50 → baseline 10. **Functional** — eval
  rollouts re-query every 10 actions instead of every 50.
- `prefix_length`: 0 → -1. Cosmetic (both disable the
  `seq_len < self.prefix_length` padding branch in
  `modeling_smolvla.py:619`).
- `pad_language_to`: `max_length` → `longest`. **Functional** — the
  `TokenizerProcessorStep` uses this to decide whether to pad every
  language prompt to 48 tokens (`max_length`) or only to the longest
  in-batch (`longest`). Different attention masks → different loss.
- `scheduler_decay_steps`: 30000 → 100000. Metadata only in this
  pipeline (trainer builds its own cosine scheduler over
  `cfg.total_steps`).

### Drift that cannot be closed safely

- `num_expert_layers`: hub 0 vs baseline-saved -1. Both route to
  `smolvlm_with_expert.py:100`'s `if num_expert_layers > 0:` → false
  branch, meaning "use same layer count as VLM (16)". The architecture
  is identical. The `_ARCHITECTURAL_POLICY_FIELDS` guard in
  `builder.py` blocks overrides of this field to avoid introducing
  shape-breaking changes on other fields guarded the same way.

---

## 2. Base-checkpoint verification (Task 3)

Param counts from `_count_params_by_submodule()`
(`trainer._dump_resolved_policy_config`):

| Submodule | Total | Trainable |
|---|---|---|
| `vlm_backbone_frozen` (VLM text + vision) | 350,165,184 | 0 |
| `action_expert_scratch` (lm_expert + action_out_proj) | 98,268,912 | 98,268,912 |
| `memory_scratch` (FullSeqMemBank) | 23,296,320 | 23,296,320 |
| `other` (proj heads, norms attached to base_policy) | 1,612,080 | 0 |
| **all** | **473,342,496** | **121,565,232** |

Interpretation:
- VLM loaded from `lerobot/smolvla_base` (~350M params), frozen.
- Action expert reinitialized from scratch per
  `builder._reinit_action_expert` (matches spec §5.1, matches
  baseline_v2's from-scratch action-expert protocol).
- Memory modules (retrieval blocks + gate fusion + timestep encoder)
  initialized from scratch.
- `other` is the remaining SmolVLAPolicy projection heads + norms that
  load pretrained weights from `smolvla_base` but are frozen for
  training.

Trainable share: 25.7%, in line with memory-only + expert-from-scratch.

---

## 3. Memory / consolidation training (Task 4)

- `mem_length: 4`, `group_size: 8` → the bank fills at t=4 and then
  consolidates on every subsequent step (t=5, 6, 7 → 3 consolidations
  per group after the initial fill). This exercises the token-merge
  path during training so the training distribution of bank states
  overlaps with eval rollouts (where episodes >> 4 steps).
- `consolidate_type: tome` unchanged.

---

## 4. `delta_timestamps` parity (Task 5)

LIBERO dataset fps (from `LeRobotDatasetMetadata("HuggingFaceVLA/libero").fps`): **10.0 Hz**.

Auto-resolved deltas (via `lerobot.datasets.factory.resolve_delta_timestamps(SmolVLAConfig, ds_meta)`):

- `observation.images.image`: `[0.0]`
- `observation.images.image2`: `[0.0]`
- `observation.state`: `[0.0]`
- `action`: `[0.0, 0.1, 0.2, ..., 4.9]` (n=50)

Identical to the formerly-hardcoded list. Match with baseline ✓ —
baseline used the same `resolve_delta_timestamps` code path.

The yaml no longer hardcodes the 50-entry list; it's recomputed at
loader construction from dataset fps + `policy.action_delta_indices`.
A future re-encode of the dataset at a different fps would
automatically re-resolve — hardcoding would have silently drifted.

---

## 5. Eval-time `mem_bank.reset()` cascade (Task 6)

[scripts/eval_memory_libero.py:89](../../scripts/eval_memory_libero.py#L89) calls `policy.reset()` at the
start of every rollout.

[src/memory_smolvla/policy/memory_smolvla.py:226-234](../../src/memory_smolvla/policy/memory_smolvla.py#L226-L234):

```python
def reset(self) -> None:
    """Reset action queues and memory for a new episode."""
    self.base_policy.reset()
    self.reset_memory()

def reset_memory(self) -> None:
    """Clear the memory bank and the inference timestep counter."""
    self.mem_bank.reset()
    self._infer_timestep = 0
```

[src/memory_smolvla/memory/full_seq_bank.py:86-90](../../src/memory_smolvla/memory/full_seq_bank.py#L86-L90):

```python
def reset(self) -> None:
    """Clear all episode banks. Call at the start of each eval rollout."""
    self.bank = {}
    self.eid_stream = None
    self._last_gate_scale = None
```

Eval-mode behavior in `FullSeqMemBank.process_batch`:
the `if self.training:` branches at L167–188 do NOT fire when the
policy is in `.eval()` mode, so batch-boundary episode clearing is
inert — the only source of truth for bank lifecycle during rollouts is
the `reset()` call at rollout start.

[scripts/eval_memory_libero.py:175](../../scripts/eval_memory_libero.py#L175) calls `policy = policy.cuda().eval()` before
rollouts — `self.training = False` is guaranteed. ✓

---

## 6. Baseline v2 sanity (Task 9)

- Training completion: **100,000 steps** (full target reached).
- Per-suite eval (from `outputs/libero_baseline_v2_eval_per_suite/{suite}/eval_info.json`):

| Suite | Success rate |
|---|---|
| libero_spatial | 84.0% |
| libero_object | 99.0% |
| libero_goal | 96.0% |
| libero_10 (long) | 72.0% |
| **4-suite average** | **87.25%** |

- Range-check vs published SmolVLA paper (87–96% depending on suite):
  ✓ in-range. LIBERO-10 (long-horizon) is the weakest suite, as
  expected — this is the suite where memory is hypothesised to help.

Implication for memvla eval: report per-suite rates rather than the
average, since memory gain concentrates on LIBERO-10 and would be
diluted 4:1 under averaging.

---

## 7. Per-suite eval reporting (Task 10)

[scripts/eval_memory_libero.py](../../scripts/eval_memory_libero.py) now supports `--all-suites`:

- Iterates over `["libero_spatial", "libero_object", "libero_goal", "libero_10"]`.
- Logs per-suite via `wandb.log({"eval/{suite}/success_rate": rate})` when `--wandb` is set.
- Logs 4-suite average as `eval/libero_avg/success_rate`.
- Per-suite mode (`--suite <name>`) preserved for targeted re-runs.

Per-suite rollout-length / failure-mode logging is not implemented;
the existing `per_task[task].successes` list already captures
binary outcome per episode — sufficient to build a failure table
post-hoc without additional instrumentation.

---

## 8. Unresolved / for human review

- **`num_expert_layers: 0 vs -1`** — cosmetic as explained in §1; the
  architectural guard in `builder.py` cannot be relaxed for this field
  alone without weakening the guard for other shape-breaking fields
  (`num_vlm_layers`, `expert_width_multiplier`, etc.). Leaving as-is.
- ~~**Tokenizer `processor.json`**~~ — **Resolved, see §9.** The
  hub-cached tokenizer did bake in `padding="max_length"` despite our
  `pad_language_to` override; fixed via
  `preprocessor_overrides={"tokenizer_processor": {"padding":
  policy.base_policy.config.pad_language_to}}` in both
  [scripts/train.py](../../scripts/train.py) and
  [scripts/eval_memory_libero.py](../../scripts/eval_memory_libero.py).
- **Eval launcher** — we ran per-suite eval separately for baseline_v2
  (current outputs directory already has per-suite JSONs). For memvla,
  `--all-suites` in one invocation will be used going forward.

---

## 9. Tokenizer padding verification

The audit flagged a risk that
`make_pre_post_processors(pretrained_path="lerobot/smolvla_base")`
would deserialize a saved `policy_preprocessor.json` and silently
ignore our `pad_language_to: longest` override. Confirmed real:

| Source | `padding` field in `policy_preprocessor.json` |
|---|---|
| `outputs/libero_baseline_v2/checkpoints/last/pretrained_model` (baseline) | `"longest"` |
| `lerobot/smolvla_base` hub cache | `"max_length"` |

### Diagnostic (`scripts/verify_tokenizer_padding.py`)

Runs the exact loader path training uses — same `build_policy` +
`make_pre_post_processors` invocation — against four prompts of
increasing length, then derives a verdict from the padded width vs the
attention-mask non-pad lengths. Full output at
[outputs/tokenizer_verification_output.txt](../tokenizer_verification_output.txt).

**Pre-fix verdicts:**

| Preprocessor | Padded width | Verdict |
|---|---|---|
| baseline_v2 saved | 27 (= max non-pad) | `longest` ✓ |
| memvla (no override) | 48 (= `tokenizer_max_length`) | `max_length` ✗ |

### Fix applied

[scripts/train.py:173-187](../../scripts/train.py#L173-L187) and
[scripts/eval_memory_libero.py:242-253](../../scripts/eval_memory_libero.py#L242-L253)
now pass
`preprocessor_overrides={"tokenizer_processor": {"padding": policy.base_policy.config.pad_language_to}}`
to `make_pre_post_processors`. The padding value is read from the
resolved `SmolVLAConfig` (post-yaml-override), so future yaml changes
to `pad_language_to` propagate automatically.

**Post-fix verdict:**

| Preprocessor | Padded width | Verdict |
|---|---|---|
| memvla (override applied) | 27 | `longest` ✓ matches baseline |

### Trainer self-check

`MemorySmolVLATrainer.__init__` now calls `_verify_tokenizer_padding()`
after wandb init and before the config dump. It runs a 2-sample batch
through `self.preprocessor.step_through(...)`, stops after the
`TokenizerProcessorStep`, derives the verdict, and:
- logs `tokenizer_padding_verdict=<verdict> ...` at INFO
- writes `tokenizer_padding_verdict: "<verdict>"` into
  `resolved_policy_config.json` alongside `_memory` / `_param_counts`
- posts `config.tokenizer_padding_verdict` to the active wandb run

Confirmed via 2-step smoke run: resolved config shows
`"tokenizer_padding_verdict": "longest"`. Any future drift (e.g. a
future refactor of `make_pre_post_processors` that once again swallows
the override) will surface in both the run's JSON and the wandb config
pane before the first gradient update.

---

Training not launched — awaiting human review of this report.
