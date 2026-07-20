"""V10A Step 0 — lerobot upstream compatibility audit (BLOCKING).

Asserts the invariants the Coconut monkey-patch depends on, against the actually
installed lerobot. Run it before any V10A code lands; wire into CI so a lerobot
bump fails loudly (the patch is version-fragile).

Findings verified (RevB):
  A. version pin
  C. make_att_2d_masks cumsum property: an appended thought (higher att_mask)
     gets a strictly higher cumsum, so no prefix position can attend it ->
     prefix computation is numerically unchanged by adding thoughts.
  D/E. KV cache is a plain dict[layer]->{key_states,value_states}; fill_kv_cache=False
     concatenates into LOCAL vars and never writes back (no implicit growth).
  F. forward_cross_attn_layer dereferences inputs_embeds[1] -> a [z, None] thought
     pass breaks there, which is why LATENT_THOUGHT needs a forced self-attn dispatch.
"""
from __future__ import annotations
import inspect
import sys

import torch

EXPECTED_LEROBOT = "0.5.1"   # pin; RevB pins ~0.5.x (upstream SHA f7283193)


def _ok(name): print(f"  [PASS] {name}")
def _fail(name, msg): print(f"  [FAIL] {name}: {msg}"); return False


def main() -> int:
    import lerobot
    from lerobot.policies.smolvla import modeling_smolvla as md
    from lerobot.policies.smolvla import smolvlm_with_expert as sm
    ok = True

    # --- A: version -----------------------------------------------------------
    print("A. version")
    if lerobot.__version__ == EXPECTED_LEROBOT:
        _ok(f"lerobot == {EXPECTED_LEROBOT}")
    else:
        ok = _fail("version", f"installed {lerobot.__version__} != pinned {EXPECTED_LEROBOT} "
                              "(patch is version-fragile; re-audit before trusting V10A)")

    # --- C: make_att_2d_masks cumsum visibility (LIVE) ------------------------
    print("C. make_att_2d_masks cumsum property (prefix cannot attend an appended thought)")
    # prefix-lm: 3 prefix tokens (att_mask 0 block) + 1 appended 'thought' (att_mask 1)
    pad = torch.tensor([[1, 1, 1, 1]])          # all valid
    att = torch.tensor([[0, 0, 0, 1]])          # prefix block, then thought starts a new block
    m2d = md.make_att_2d_masks(pad, att)[0]     # (N, N): m2d[q, k] = q can attend k
    prefix_attends_thought = m2d[:3, 3].any().item()
    thought_attends_prefix = m2d[3, :3].all().item()
    prefix_block_full = m2d[:3, :3].all().item()
    if (not prefix_attends_thought) and thought_attends_prefix and prefix_block_full:
        _ok("prefix↛thought, thought→prefix, prefix block mutually visible")
    else:
        ok = _fail("cumsum", f"prefix_attends_thought={prefix_attends_thought} "
                             f"thought_attends_prefix={thought_attends_prefix} "
                             f"prefix_block_full={prefix_block_full}")

    # --- D/E: cache is a dict; fill_kv_cache=False does not write back ---------
    print("D/E. KV cache dict + fill_kv_cache=False non-writeback (source-level)")
    src_attn = " ".join(inspect.getsource(sm.SmolVLMWithExpertModel.forward_attn_layer).split())
    writes_on_true = 'past_key_values[layer_idx] = {' in src_attn
    # fill=False cats the CACHED dict into LOCAL key_states/value_states (the non-writeback pattern):
    cats_cached_into_local = (
        'key_states = torch.cat( [past_key_values[layer_idx]["key_states"]' in src_attn
        or 'key_states = torch.cat([past_key_values[layer_idx]["key_states"]' in src_attn
    )
    # the ONLY assignment back into the cache dict is the fill=True writeback (exactly once):
    no_extra_writeback = src_attn.count('past_key_values[layer_idx] = ') == 1
    if writes_on_true and cats_cached_into_local and no_extra_writeback:
        _ok("fill=True writes dict; fill=False cats to locals, no writeback (must append_delta_kv ourselves)")
    else:
        ok = _fail("cache-writeback", f"writes_on_true={writes_on_true} cats_locally={cats_locally} "
                                      f"no_writeback_in_false={no_writeback_in_false}")

    # --- F: cross-attn layer dereferences inputs_embeds[1] --------------------
    print("F. forward_cross_attn_layer dereferences inputs_embeds[1] ([z, None] breaks -> forced dispatch)")
    src_cross = inspect.getsource(sm.SmolVLMWithExpertModel.forward_cross_attn_layer)
    derefs_expert = "inputs_embeds[1]" in src_cross
    if derefs_expert:
        _ok("cross-attn dereferences inputs_embeds[1]; LATENT_THOUGHT must force forward_attn_layer")
    else:
        ok = _fail("cross-deref", "expected inputs_embeds[1] dereference not found (dispatch assumption changed)")

    # --- dispatch rule (record) ----------------------------------------------
    print("dispatch rule (recorded): self-attn if (fill_kv_cache OR 'cross' not in mode OR "
          "layer_idx % self_attn_every_n_layers == 0), else cross-attn")

    print("\nRESULT:", "ALL PASS — V10A assumptions hold on this lerobot" if ok
          else "FAILURES — re-audit before implementing V10A")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
