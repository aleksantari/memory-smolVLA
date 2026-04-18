# Claude Code task: Verify tokenizer padding behavior before launching memvla training

## Context

The config alignment report (Section 8, bullet 2) flagged a risk: `make_pre_post_processors` may load a cached tokenizer processor json from `smolvla_base` that bakes in `padding="max_length"` regardless of the `pad_language_to: longest` setting in `configs/memvla_libero.yaml`. If this happens, the tokenizer silently ignores our override and the training recipe diverges from what the yaml describes.

This is the single most likely silent-bug scenario remaining before training launch. We need to verify the actual behavior of the preprocessor, not just the configured behavior, before committing to 100k training steps.

The goal of this task is to catch any preprocessor drift and fix it, **without launching training**.

## Task 1. Smoke test the memvla preprocessor with a real batch

Write a standalone diagnostic script `scripts/verify_tokenizer_padding.py` that:

1. Instantiates the memvla policy and preprocessor using the exact same loader path that the trainer uses (`configs/memvla_libero.yaml`). Do not shortcut this by instantiating the tokenizer directly — the point is to verify the path that training actually uses.

2. Constructs a test batch of at least 4 samples with language prompts of **deliberately different lengths**. Suggested prompts (use realistic LIBERO-style strings):
   - Short: `"pick up the cup"` (~5 tokens)
   - Medium: `"move the red block onto the table"` (~9 tokens)
   - Long: `"carefully stack the ceramic bowl on top of the wooden crate near the stove"` (~17 tokens)
   - Very long: pad the prompt to get close to 48 tokens, e.g. `"please carefully and precisely stack the small ceramic bowl on top of the wooden crate that is sitting near the stove on the left side"` (~28 tokens)

3. Runs the full preprocessor chain on this batch as the trainer would, producing the tokenized language inputs and their attention masks.

4. **Prints / logs the following diagnostics:**
   - The raw input strings
   - The token IDs for each sample (full tensor, not truncated)
   - The length of each sample's tokens (non-pad count, derived from attention mask)
   - The total padded length of each sample (tensor shape)
   - The attention mask for each sample
   - Whether all samples padded to the same length (= max_length behavior) or to the longest in the batch (= longest behavior)

5. Prints a verdict at the end:
   - **`VERDICT: longest`** if all 4 samples have a padded length equal to the longest non-pad length in the batch (and that longest value is < 48)
   - **`VERDICT: max_length`** if all 4 samples have a padded length of exactly 48 (the `tokenizer_max_length`)
   - **`VERDICT: ambiguous / other`** if neither pattern holds — include enough detail in the output to diagnose

Do NOT make any assumptions about which is "correct" yet. Just report what's happening.

## Task 2. Also smoke test the baseline preprocessor

Run the same diagnostic against the baseline_v2 checkpoint's preprocessor (`outputs/libero_baseline_v2/checkpoints/last/pretrained_model`).

This matters because baseline_v2 is our reference point. If baseline_v2 silently ran in `max_length` mode despite its config saying `longest`, then memvla running in `longest` mode would actually be drifting from what baseline_v2 *actually did* (as opposed to what baseline_v2's config claimed).

Report the same verdict structure for baseline.

## Task 3. Decide on the fix, if needed

After Tasks 1 and 2, there are four possible outcomes:

| baseline behavior | memvla behavior | Action |
|---|---|---|
| `longest` | `longest` | ✓ No fix needed. Document in report. |
| `max_length` | `max_length` | ✓ No fix needed (both match each other, even if they don't match the yaml). Flag for human review — we may want to understand why the config override is being ignored. |
| `longest` | `max_length` | **Fix needed** — memvla must be brought into `longest` mode to match baseline. Apply `preprocessor_overrides={"tokenizer": {"padding": "longest"}}` or whatever the correct loader-level override is. Re-run Task 1 to confirm the fix worked. |
| `max_length` | `longest` | **Fix needed** — memvla must be brought into `max_length` mode to match baseline. Either set `pad_language_to: max_length` in the yaml, or apply the equivalent preprocessor override. Re-run Task 1 to confirm. |

**Key principle: match baseline exactly.** If baseline was silently in `max_length` despite its config, memvla should match that, not the yaml-declared setting. The ablation's integrity requires that the only intentional difference between the two runs is the memory module — not the tokenizer padding mode.

If a fix is applied, verify that it is persistent across trainer restart by running the verification script a second time after the fix. Cached preprocessor JSONs have a habit of resurrecting.

## Task 4. Add the tokenizer verification to the trainer startup

After any fix is applied and verified, add a lightweight version of the Task 1 diagnostic to the trainer's startup logging, right after the preprocessor is constructed and before the first training step.

Specifically:

1. Inside the trainer init (same place that `_dump_resolved_policy_config` is called), after the preprocessor is built, construct a 2-sample test batch (short + long prompt).
2. Run the preprocessor on this mini-batch.
3. Log the resolved padding mode (verdict) to the same `resolved_policy_config.json` file under a new field `tokenizer_padding_verdict: "longest" | "max_length" | "ambiguous"`.
4. Also log it to stdout / wandb as `config/tokenizer_padding_verdict` so it shows up in wandb's config pane.

This means every future training run will self-declare its actual tokenizer behavior, and we can detect drift without needing to run the standalone script.

## Task 5. Update the alignment report

Append a new section to `outputs/<run_name>/config_alignment_report.md`:

**"9. Tokenizer padding verification"**

Include:
- Memvla verdict (from Task 1)
- Baseline verdict (from Task 2)
- Whether any fix was applied (Task 3)
- Confirmation that the trainer startup now logs the verdict (Task 4)

Mark Task 9 as resolved (or as the specific issue that was found and fixed).

## Deliverables

1. `scripts/verify_tokenizer_padding.py` — the standalone diagnostic script
2. Output of running it against memvla and baseline preprocessors (save to `outputs/tokenizer_verification_output.txt`)
3. Any fix applied to the memvla loader / config (if needed)
4. New trainer startup log entry with `tokenizer_padding_verdict`
5. Updated `config_alignment_report.md` with the new section

## Non-goals

- Do NOT launch training. This task is purely verification and preparatory fix.
- Do NOT modify the baseline — baseline_v2 is immutable reference. We are aligning memvla to whatever baseline actually does, not the other way around.
- Do NOT speculate about whether `longest` or `max_length` is "better" for training outcomes. The SmolVLA paper setting is `longest`; that's a separate question from "what did baseline_v2 actually run." Align to baseline_v2's actual behavior.

## Expected timeline

This should be a 30-minute task. The standalone verification script is ~50 lines. If the verdicts match (both `longest` or both `max_length`), no fix is needed and the task is just diagnostic. If they don't match, the fix is a one-line preprocessor override or yaml change.

Do not proceed to training launch until the updated alignment report confirms the tokenizer padding mode is consistent between baseline and memvla.
