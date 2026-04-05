"""Memory module components: bank, retrieval, gating, temporal PE, and extensions."""

from memory_smolvla.memory.bank import ConsolidatingMemoryBank, MemoryBank
from memory_smolvla.memory.compressor import MemoryCompressor
from memory_smolvla.memory.gating import ResidualGate, SigmoidGate
from memory_smolvla.memory.multi_scale_bank import MultiScaleMemoryBank
from memory_smolvla.memory.retrieval import CrossAttentionRetrieval
from memory_smolvla.memory.temporal_pe import TemporalPositionalEncoding
from memory_smolvla.memory.working_memory import WorkingMemory
from memory_smolvla.memory.write_gate import WriteGate

__all__ = [
    "ConsolidatingMemoryBank",
    "CrossAttentionRetrieval",
    "MemoryBank",
    "MemoryCompressor",
    "MultiScaleMemoryBank",
    "ResidualGate",
    "SigmoidGate",
    "TemporalPositionalEncoding",
    "WorkingMemory",
    "WriteGate",
]
