"""Full-sequence cognitive memory bank for SmolVLA.

Stores the full VLM output sequence (length ``L``) per timestep, keyed by
episode id. Retrieves via stacked cross-attention blocks with temporal
positional encoding on bank keys. Fuses current and retrieved tokens via
a learned per-token sigmoid gate. Consolidates the bank via token-merge
(ToMe) when capacity is exceeded.

Adapted from MemoryVLA's ``CogMemBank`` (``vla/memory_vla.py``). The only
structural difference is that we pass the full VLM output sequence
(``N = L``) instead of a single pooled token (``N = 1``). Retrieval,
gating, and consolidation are generic over token count.

See ``memory_smolvla_implementation_spec.md`` §3.4.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from memory_smolvla.memory.blocks import (
    CrossTransformerBlock,
    GateFusion,
    TimestepEmbedder,
)

_VALID_CONSOLIDATE = frozenset({"tome", "fifo"})
_VALID_DATALOADER = frozenset({"group", "stream"})


class FullSeqMemBank(nn.Module):
    def __init__(
        self,
        token_size: int,
        mem_length: int = 8,
        retrieval_layers: int = 2,
        use_timestep_pe: bool = True,
        consolidate_type: str = "tome",
        update_fused: bool = False,
        dataloader_type: str = "group",
        group_size: int = 1,
    ) -> None:
        super().__init__()
        if consolidate_type not in _VALID_CONSOLIDATE:
            raise ValueError(
                f"Invalid consolidate_type {consolidate_type!r}. "
                f"Must be one of {sorted(_VALID_CONSOLIDATE)}."
            )
        if dataloader_type not in _VALID_DATALOADER:
            raise ValueError(
                f"Invalid dataloader_type {dataloader_type!r}. "
                f"Must be one of {sorted(_VALID_DATALOADER)}."
            )

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

        self.bank: dict[Any, list[tuple[int | None, Tensor]]] = {}
        self.eid_stream = None
        self._last_gate_scale: Tensor | None = None
        self.bypass: bool = False

    # -- bank lifecycle ---------------------------------------------------

    def reset(self) -> None:
        """Clear all episode banks. Call at the start of each eval rollout."""
        self.bank = {}
        self.eid_stream = None
        self._last_gate_scale = None

    def clear_episode(self, episode_id: Any) -> None:
        self.bank.pop(episode_id, None)

    # -- consolidation ----------------------------------------------------

    @torch.no_grad()
    def _consolidate_with_token_merge(self, episode_id: Any) -> None:
        bank = self.bank.get(episode_id, [])
        T = len(bank)
        if T < 2:
            return

        feats = [feat for (_, feat) in bank]  # each: (L, D)

        sims = []
        for i in range(T - 1):
            f1 = feats[i].flatten(0)
            f2 = feats[i + 1].flatten(0)
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
        episode_id: Any,
        feat: Tensor,
        timestep: int | None,
    ) -> None:
        if episode_id not in self.bank:
            self.bank[episode_id] = []

        self.bank[episode_id].append((timestep, feat.detach().clone()))

        while len(self.bank[episode_id]) > self.mem_length:
            if self.consolidate_type == "fifo":
                self.bank[episode_id] = self.bank[episode_id][-self.mem_length:]
            else:
                self._consolidate_with_token_merge(episode_id)

    # -- main call --------------------------------------------------------

    def process_batch(
        self,
        tokens: Tensor,
        episode_ids: list,
        timesteps: list,
    ) -> Tensor:
        """Retrieve, fuse, and write for each batch item.

        Args:
            tokens: ``(B, L, D)`` current VLM output sequences.
            episode_ids: length-``B`` list of hashable episode ids.
            timesteps: length-``B`` list of integer timesteps within each episode.

        Returns:
            Fused tokens of shape ``(B, L, D)``.
        """
        B, L, D = tokens.shape
        if len(episode_ids) != B or len(timesteps) != B:
            raise ValueError(
                f"episode_ids (len {len(episode_ids)}) and timesteps "
                f"(len {len(timesteps)}) must both have length B={B}."
            )

        if self.bypass:
            self._last_gate_scale = torch.ones(
                (B, L, D), device=tokens.device, dtype=tokens.dtype,
            )
            return tokens

        outputs = []
        gate_scales = []

        if self.training:
            if self.dataloader_type == "group":
                self.bank.clear()
                self.eid_stream = None
            else:
                first_eid = episode_ids[0]
                if self.eid_stream is not None and self.eid_stream != first_eid:
                    self.clear_episode(self.eid_stream)
                self.eid_stream = first_eid

        for i in range(B):
            eid = episode_ids[i]

            if self.training:
                if self.dataloader_type == "group":
                    if i > 0 and i % self.group_size == 0:
                        prev_group_eid = episode_ids[i - self.group_size]
                        self.clear_episode(prev_group_eid)
                else:
                    if i > 0 and episode_ids[i] != episode_ids[i - 1]:
                        self.clear_episode(episode_ids[i - 1])
                        self.eid_stream = episode_ids[i]

            working_mem = tokens[i].unsqueeze(0)  # (1, L, D)

            hist = self.bank.get(eid, [])
            if len(hist) > 0:
                hist_feats = [feat for _, feat in hist]
                episode_mem = torch.stack(hist_feats, dim=0)  # (T, L, D)
                T = episode_mem.shape[0]
                episode_mem_flat = episode_mem.reshape(T * L, D).unsqueeze(0)
                episode_mem_flat = episode_mem_flat.to(
                    dtype=working_mem.dtype, device=working_mem.device
                )

                if self.use_timestep_pe and self.timestep_encoder is not None:
                    hist_ts = torch.tensor(
                        [t if t is not None else 0 for t, _ in hist],
                        device=working_mem.device,
                        dtype=torch.long,
                    )
                    pe = self.timestep_encoder(hist_ts).unsqueeze(0)  # (1, T, D)
                    pe = pe.repeat_interleave(L, dim=1).to(
                        dtype=episode_mem_flat.dtype
                    )
                else:
                    pe = torch.zeros_like(episode_mem_flat)

                query = working_mem
                for block in self.retrieval_blocks:
                    query = block(query, episode_mem_flat + pe, episode_mem_flat)
                retrieved = query
            else:
                retrieved = working_mem

            gate_scale = self.gate_fusion.last_scale(working_mem, retrieved)
            fused = gate_scale * working_mem + (1 - gate_scale) * retrieved
            outputs.append(fused)
            gate_scales.append(gate_scale.detach())

            ts_i = timesteps[i] if self.use_timestep_pe else None
            to_store = fused.squeeze(0) if self.update_fused else tokens[i]
            self._memory_consolidate(eid, to_store, ts_i)

        self._last_gate_scale = torch.cat(gate_scales, dim=0)
        return torch.cat(outputs, dim=0)

    # -- utilities --------------------------------------------------------

    def num_entries(self, episode_id: Any) -> int:
        return len(self.bank.get(episode_id, []))

    def last_gate_scale(self) -> Tensor | None:
        return self._last_gate_scale
