# Memory-SmolVLA Training Tutorial

This tutorial covers how to train and fine-tune the memory-augmented SmolVLA system across all supported training modes.

## Overview

Memory-SmolVLA wraps a frozen [SmolVLA](https://huggingface.co/lerobot/smolvla_base) policy (a 450M-parameter Vision-Language-Action model from LeRobot) with a trainable temporal memory system. The memory intercepts VLM hidden states at a configurable transformer layer, stores them in an episodic memory bank, retrieves relevant past context via cross-attention, and fuses the result back into the representation stream through a learned sigmoid gate. The action expert then cross-attends to memory-augmented context rather than raw VLM output.

Only the memory modules have trainable parameters by default — the VLM backbone is always frozen. Different training modes selectively unfreeze the action expert for variable-depth VLM experiments.

### Architecture at a glance

```
Observation + Language
        |
   VLM layers 0 .. injection_layer
        |
   [Memory Callback]
     |-- Read from memory bank (past timesteps)
     |-- Cross-attention retrieval + temporal PE
     |-- Sigmoid-gated fusion (alpha starts near 0)
     |-- Write current features to bank (detached)
        |
   VLM layers injection_layer+1 .. N-1
        |
   Final LayerNorm
        |
   Action Expert (flow-matching head)
        |
   Predicted action chunk
```

### Trainable vs frozen components

| Component | `memory_only` | `expert_finetune` | `expert_scratch` | `expert_only_scratch` |
|---|---|---|---|---|
| VLM backbone (SmolLM2) | Frozen | Frozen | Frozen | Frozen |
| CrossAttentionRetrieval | **Trained** | **Trained** | **Trained** | Frozen |
| SigmoidGate | **Trained** | **Trained** | **Trained** | Frozen |
| memory_proj (Linear) | **Trained** | **Trained** | **Trained** | Frozen |
| Action expert (lm_expert) | Frozen | **Trained** | **Trained** | **Trained** |
| Output projection | Frozen | **Trained** | **Trained** | **Trained** |

---

## Installation

```bash
# Clone and install in editable mode
git clone <repo-url> memory-smolvla
cd memory-smolvla
pip install -e ".[dev]"

# For LIBERO evaluation support
pip install -e ".[dev,libero]"
```

**Requirements:** Python >= 3.10, CUDA-capable GPU, PyTorch >= 2.2.1.

**HuggingFace authentication** (needed for private datasets):

```bash
# Option 1: environment variable
export HF_TOKEN=hf_your_token_here

# Option 2: persistent login (recommended)
huggingface-cli login
```

---

## Training Modes

Each training mode controls which parameters receive gradients, which checkpoint strategy is used, and how data is loaded.

### 1. `memory_only` — Train memory on top of pretrained SmolVLA

**When to use:** You have the standard 16-layer pretrained SmolVLA and want to add temporal memory without touching the action expert. This is the fastest mode and the primary use case.

**What's trained:**
- CrossAttentionRetrieval (cross-attention + layer norm)
- SigmoidGate (2-layer MLP)
- memory_proj (zero-initialized Linear)

**What's frozen:** VLM backbone, action expert, output projection (all from pretrained checkpoint)

**VLM layers:** Must be 16 (pretrained checkpoint constraint)

**Data loading:** Episode-sequential (B=1, frames in temporal order per episode)

**Config:** `configs/memory_only.yaml`

```yaml
_base_: base.yaml

policy:
  training_mode: memory_only
  base_checkpoint: lerobot/smolvla_base   # Pretrained 16-layer SmolVLA
  num_vlm_layers: 16
  injection_layer: 8                       # Memory fires at layer 8 of 16

trainer:
  training_mode: memory_only
  total_steps: 100000
  memory_lr: 1.0e-4
  warmup_steps: 1000
```

**Run:**

```bash
python scripts/train.py --config configs/memory_only.yaml
```

---

### 2. `expert_finetune` — Joint memory + expert fine-tuning

**When to use:** You want to jointly adapt both the memory system and the pretrained action expert. Useful when the target task distribution differs significantly from SmolVLA's pretraining data.

**What's trained:** All memory modules + action expert + output projection

**What's frozen:** VLM backbone only

**VLM layers:** Must be 16 (pretrained checkpoint constraint)

**Data loading:** Episode-sequential (B=1)

**Config:** Create `configs/expert_finetune.yaml`:

```yaml
_base_: base.yaml

policy:
  training_mode: expert_finetune
  base_checkpoint: lerobot/smolvla_base
  num_vlm_layers: 16
  injection_layer: 8

trainer:
  training_mode: expert_finetune
  total_steps: 150000
  memory_lr: 1.0e-4
  expert_lr: 5.0e-5       # Lower LR for pretrained expert to avoid catastrophic forgetting
  warmup_steps: 1500
```

**Run:**

```bash
python scripts/train.py --config configs/expert_finetune.yaml
```

---

### 3. `expert_scratch` — Variable VLM depth with memory (8 layers)

**When to use:** You want to study how VLM depth affects memory-augmented policy learning. With fewer than 16 layers, the pretrained action expert is incompatible (it was trained to cross-attend to 16-layer representations), so the expert must be trained from scratch.

**What's trained:** All memory modules + action expert (randomly initialized) + output projection

**What's frozen:** VLM backbone (pretrained weights loaded, then truncated to 8 layers)

**VLM layers:** 8 (or any count you choose — the VLM is truncated at init time)

**Data loading:** Episode-sequential (B=1)

**Config:** `configs/expert_scratch_8layers.yaml`

```yaml
_base_: base.yaml

policy:
  training_mode: expert_scratch
  num_vlm_layers: 8
  injection_layer: 4       # Mid-VLM: layer 4 of 8
  bank_max_size: 16

trainer:
  training_mode: expert_scratch
  total_steps: 200000      # More steps needed — expert starts from random init
  memory_lr: 1.0e-4
  expert_lr: 5.0e-5
  warmup_steps: 2000
  checkpoint_every: 10000
  wandb_run_name: expert_scratch_8layers
```

**Run:**

```bash
python scripts/train.py --config configs/expert_scratch_8layers.yaml
```

---

### 4. `expert_scratch` — Variable VLM depth with memory (24 layers)

Same mode as above, but with a deeper VLM. More layers give the VLM richer representations, but the action expert must still be trained from scratch since 24 != 16.

**Config:** `configs/expert_scratch_24layers.yaml`

```yaml
_base_: base.yaml

policy:
  training_mode: expert_scratch
  num_vlm_layers: 24
  injection_layer: 12      # Mid-VLM: layer 12 of 24
  bank_max_size: 16

trainer:
  training_mode: expert_scratch
  total_steps: 200000
  memory_lr: 1.0e-4
  expert_lr: 5.0e-5
  warmup_steps: 2000
  checkpoint_every: 10000
  wandb_run_name: expert_scratch_24layers
```

**Run:**

```bash
python scripts/train.py --config configs/expert_scratch_24layers.yaml
```

---

### 5. `expert_only_scratch` — Ablation baseline (no memory)

**When to use:** Ablation control. Trains only the action expert with no memory system, so you can measure how much the memory modules contribute at a given VLM depth.

**What's trained:** Action expert + output projection only

**What's frozen:** VLM backbone + all memory modules (retrieval, gate, memory_proj are created but explicitly frozen)

**VLM layers:** Any count (8 in the provided config)

**Data loading:** Standard random batching (`DataLoader` with `EpisodeAwareSampler`, B=`batch_size`). No episode ordering needed since there is no memory state.

**Config:** `configs/ablation_8layers.yaml`

```yaml
_base_: base.yaml

policy:
  training_mode: expert_only_scratch
  num_vlm_layers: 8

trainer:
  training_mode: expert_only_scratch
  total_steps: 200000
  expert_lr: 5.0e-5
  warmup_steps: 2000
  checkpoint_every: 10000
  wandb_run_name: ablation_8layers_no_memory
  batch_size: 32
  num_workers: 4
```

**Run:**

```bash
python scripts/train.py --config configs/ablation_8layers.yaml
```

---

## Datasets

### Using the community dataset

The default configs use the community dataset built for SmolVLA development:

```yaml
dataset:
  repo_ids:
    - "danaaubakirova/svla_so100_task1_v3"
```

This is a public HuggingFace dataset — no authentication needed.

### Using your own HuggingFace datasets

Replace `repo_ids` with your dataset's repo ID:

```yaml
dataset:
  repo_ids:
    - "your-username/your-robot-dataset"
```

For **private datasets**, authenticate first:

```bash
# Pass token at runtime
HF_TOKEN=hf_... python scripts/train.py --config your_config.yaml

# Or via CLI flag
python scripts/train.py --config your_config.yaml --hf-token hf_...

# Or login once (persists)
huggingface-cli login
```

### Using multiple datasets

List multiple repo IDs to concatenate them. Episodes from all datasets are interleaved (shuffled across datasets in sequential mode):

```yaml
dataset:
  repo_ids:
    - "danaaubakirova/svla_so100_task1_v3"
    - "your-username/your-custom-dataset"
```

### Delta timestamps

`delta_timestamps` specifies which time offsets to include for each feature. The default loads only the current frame:

```yaml
dataset:
  delta_timestamps:
    observation.state: [0.0]
    action: [0.0]
```

To include the previous frame's state as well:

```yaml
  delta_timestamps:
    observation.state: [-0.1, 0.0]   # Previous + current
    action: [0.0]
```

---

## Key Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `total_steps` | 100,000 | Total gradient update steps |
| `memory_lr` | 1e-4 | Learning rate for memory modules (retrieval, gate, memory_proj) |
| `expert_lr` | 1e-4 | Learning rate for action expert + output projection |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping norm (0 to disable) |
| `warmup_steps` | 1,000 | Linear warmup steps before cosine decay |
| `grad_accum_steps` | 1 | Gradient accumulation steps per optimizer update |
| `checkpoint_every` | 5,000 | Save checkpoint every N steps |
| `log_every` | 100 | Log metrics every N steps |
| `device` | `cuda` | PyTorch device |
| `batch_size` | 32 | Batch size (only used in `expert_only_scratch` random mode) |

### Separate learning rate groups

The trainer creates separate optimizer parameter groups with independent learning rates:

- **Memory group** (`memory_lr`): retrieval, gate, memory_proj — used in all modes except `expert_only_scratch`
- **Expert group** (`expert_lr`): lm_expert, action_out_proj — used in `expert_scratch`, `expert_finetune`, `expert_only_scratch`

In `memory_only` mode, only the memory group exists. A lower `expert_lr` (e.g. 5e-5) is recommended when fine-tuning a pretrained expert to prevent catastrophic forgetting.

### Learning rate schedule

All modes use **cosine decay with linear warmup**:

```
LR
 ^
 |    /----\
 |   /      \
 |  /        \
 | /          \___
 +--+---------+---> step
  warmup    total_steps
```

### Gradient accumulation

For episode-sequential modes, each "sample" is a single frame (B=1). With `grad_accum_steps=4`, the optimizer updates every 4 frames, giving an effective batch of 4 temporally-ordered frames from the same episode.

For random mode (`expert_only_scratch`), the effective batch size is `batch_size * grad_accum_steps`.

---

## Resuming Training

To resume from a checkpoint:

```bash
python scripts/train.py --config configs/memory_only.yaml --resume checkpoints/step_0050000.pt
```

This restores:
- Policy weights (all modules)
- Optimizer state (AdamW momentum)
- LR scheduler position
- Step counter (training continues from where it left off)

---

## Monitoring

### Weights & Biases

Set `wandb_project` in your config to enable logging:

```yaml
trainer:
  wandb_project: memory-smolvla
  wandb_run_name: my-experiment      # Optional, auto-generated if null
```

To disable wandb, set `wandb_project: null`.

### Key metrics to watch

| Metric | What it tells you |
|---|---|
| `train/loss` | Flow-matching MSE loss — should decrease steadily |
| `train/gate_alpha_mean` | Average sigmoid gate activation. Starts near 0.007 (identity init). Rising alpha means the memory system is learning to contribute. |
| `train/gate_alpha_std` | Gate activation variance. Increasing std suggests the model is selectively using memory for some tokens but not others (good sign). |
| `train/memory_bank_size` | Current number of entries in the memory bank. Grows within each episode, resets at boundaries. |
| `train/lr_group_0` | Learning rate of the first parameter group (memory or expert, depending on mode) |

### Interpreting gate alpha

- **alpha near 0** (early training): Memory is not contributing — the model behaves like vanilla SmolVLA. This is expected and by design (zero-init identity start).
- **alpha rising to 0.1-0.5**: Memory is being used. The model has learned to retrieve useful temporal context.
- **alpha near 1.0**: Memory dominates — the model heavily relies on past context. Rare, and may indicate overfitting to temporal shortcuts.

---

## Evaluation

### LIBERO rollout evaluation

Evaluate a trained checkpoint on LIBERO manipulation tasks:

```bash
python scripts/eval.py \
    --checkpoint checkpoints/step_0100000.pt \
    --config configs/memory_only.yaml \
    --tasks libero_10 \
    --n-rollouts 10 \
    --max-steps 500 \
    --output-dir eval_results
```

This runs rollouts in the LIBERO simulator, measuring success rate per task and recording gate activation timelines.

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--config` | (required) | YAML config (needed to reconstruct the policy) |
| `--tasks` | `libero_10` | Task suite or comma-separated task names |
| `--n-rollouts` | 10 | Rollouts per task |
| `--max-steps` | 500 | Maximum steps per episode |
| `--output-dir` | `eval_results` | Directory for results JSON |
| `--wandb-project` | None | Log eval metrics to wandb |

### Gate activation analysis

Visualize how gate alpha evolves over episode timesteps:

```bash
python scripts/analyze_gates.py \
    --results eval_results/results.json \
    --output plots/
```

Produces:
- Per-task gate activation timelines (PNG)
- Summary bar chart of average gate alpha across tasks

---

## Custom Configs

### Config inheritance

All configs use `_base_` to inherit from `configs/base.yaml`. Override only what differs:

```yaml
# configs/my_experiment.yaml
_base_: base.yaml

policy:
  training_mode: expert_scratch
  num_vlm_layers: 12                # Custom layer count
  injection_layer: 6                # Mid-VLM injection

dataset:
  repo_ids:
    - "my-username/my-dataset"      # Custom dataset

trainer:
  training_mode: expert_scratch
  total_steps: 150000
  wandb_run_name: 12layer_experiment
```

### Choosing the injection layer

The injection layer determines where in the VLM the memory system intercepts hidden states. General guidelines:

| VLM layers | Recommended injection layer | Rationale |
|---|---|---|
| 8 | 4 | Mid-VLM (N/2) — balanced between low-level and high-level features |
| 12 | 6 | Mid-VLM (N/2) |
| 16 | 8 | Mid-VLM (N/2) — default for pretrained SmolVLA |
| 24 | 12 | Mid-VLM (N/2) |

Valid range: `0` to `num_vlm_layers - 1`. Lower values inject earlier (lower-level features), higher values inject later (higher-level, more semantic features). N/2 is the default starting point; tuning this is an open research question.

### Quick smoke test

Use `--steps` to cap training at a small number of steps for debugging:

```bash
python scripts/train.py --config configs/memory_only.yaml --steps 50
```

---

## Experiment Matrix

For a complete ablation study across VLM depths, run these configs:

| Experiment | Config | VLM layers | Memory | Expert | Purpose |
|---|---|---|---|---|---|
| Memory-only (baseline) | `memory_only.yaml` | 16 | Trained | Pretrained (frozen) | Primary use case |
| Expert fine-tune | `expert_finetune.yaml` | 16 | Trained | Pretrained (fine-tuned) | Joint adaptation |
| 8-layer + memory | `expert_scratch_8layers.yaml` | 8 | Trained | From scratch | Shallow VLM + memory |
| 8-layer no memory | `ablation_8layers.yaml` | 8 | Frozen | From scratch | Ablation control |
| 24-layer + memory | `expert_scratch_24layers.yaml` | 24 | Trained | From scratch | Deep VLM + memory |

Comparing "8-layer + memory" vs "8-layer no memory" directly measures the memory system's contribution at a given VLM depth. Comparing across VLM depths (8 vs 16 vs 24) reveals how temporal memory interacts with representation quality.
