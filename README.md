# Memory-Augmented Compact VLAs

Adding a temporal memory system to [SmolVLA](https://huggingface.co/papers/2506.01844) (450M-parameter Vision-Language-Action model) for improved multi-step robotic manipulation.

## Architecture

The memory module sits between SmolVLA's frozen VLM backbone (SigLIP + SmolLM2) and the flow-matching action expert:

```
Frozen VLM (SigLIP + SmolLM2)
  → layer N/2 output
  → Memory Module:
      → write to ring buffer memory bank
      → cross-attention retrieval with temporal positional encoding
      → sigmoid-gated fusion
  → Action Expert (flow matching)
  → action chunk
```

The action expert receives the same representation shape it was trained on, just memory-augmented.

## Setup

```bash
pip install -e ".[dev]"
```

For LIBERO evaluation:
```bash
pip install -e ".[dev,libero]"
```

## Training

```bash
python scripts/train.py --config configs/base.yaml
```

## Evaluation

```bash
python scripts/eval.py --checkpoint path/to/checkpoint --config configs/libero_long.yaml
```

## Project Structure

```
src/memory_smolvla/
  memory/          # Memory bank, cross-attention retrieval, gating, temporal PE
  policy/          # MemorySmolVLAPolicy wrapper, feature extraction hooks
  data/            # Episode-sequential data loader
  training/        # Training loop with episode-sequential memory buildup
  eval/            # LIBERO evaluation with success rate + gate activation tracking
```

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed architecture and implementation notes.
