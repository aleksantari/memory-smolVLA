# Continuing memvla_libero training — handoff instructions

This is a handoff snapshot from Aleks's machine. Training was paused at step **32,800 / 100,000**; the last on-disk checkpoint is at **step 30,000**. These steps cover everything you need to resume on your own GPU.

## 1. Prerequisites

- **GPU:** CUDA-capable GPU with ≥ 10 GB VRAM. Training currently uses ~8.5 GB with `batch=32, bfloat16`. A 16 GB RTX 4080 is plenty.
- **Python / conda:** Miniconda or Anaconda.
- **Disk:** ~60 GB free (LIBERO dataset cache ~40 GB + repo + checkpoints).

## 2. Clone the repo

```bash
git clone https://github.com/Aleksantari/memory-smolvla.git
cd memory-smolvla
git checkout dev
```

All the work-in-progress changes (tokenizer padding fix, multi-worker DataLoader, etc.) live on `dev`.

## 3. Create the conda env

```bash
conda create -n lerobot python=3.11 -y
conda activate lerobot

# Install PyTorch matching your CUDA. For an RTX 4080 (Ada, sm_89),
# the default PyTorch stable wheels work — no cu128 nightly needed.
pip install torch torchvision

# Install LeRobot from the sibling repo path used in development.
# Easiest: clone LeRobot next to this repo and install editable.
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla]"
cd ../memory-smolvla

# Install this project in editable mode.
pip install -e ".[dev]"
```

## 4. Authenticate with HuggingFace

The LIBERO dataset and checkpoint live on HF. Create a **read** token at https://huggingface.co/settings/tokens, then:

```bash
huggingface-cli login   # paste the token
# or: export HF_TOKEN=hf_xxx in your shell
```

## 5. Download the step-30k checkpoint

```bash
mkdir -p checkpoints/memvla_libero
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='aleksantari/memvla-libero-ckpts',
    filename='step_0030000.pt',
    local_dir='checkpoints/memvla_libero',
)
"
```

You should see `checkpoints/memvla_libero/step_0030000.pt` (≈ 1.5 GB).

## 6. (Optional) Prime the LIBERO dataset cache

First run downloads ~40 GB of LIBERO parquet + videos from `HuggingFaceVLA/libero`. You can either just let training handle it (network-bound for the first few minutes) or pre-fetch:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(repo_id='HuggingFaceVLA/libero')
"
```

## 7. DO NOT CHANGE THE CONFIG

**Critical.** Everything in [configs/memvla_libero.yaml](configs/memvla_libero.yaml) is frozen — this is a controlled experiment against the baseline_v2 reference run. Any change invalidates the comparison and the remaining 70k training steps are wasted.

**Locked — do NOT modify under any circumstance:**

- `trainer.num_groups` (4) and `trainer.group_size` (8) — effective batch size is **32**. Changing either changes the loss signal and gate statistics.
- `trainer.num_workers` (8) — affects RNG stream ordering across workers. Changing it changes which frames land in which batch and silently desyncs the run.
- `trainer.seed` (1000), `trainer.memory_lr`, `trainer.expert_lr`, `trainer.weight_decay`, `trainer.adam_betas`, `trainer.adam_eps`, `trainer.max_grad_norm`, `trainer.warmup_steps`, `trainer.total_steps` — any change breaks parity with baseline_v2.
- `trainer.use_amp` (true), `trainer.amp_dtype` (bfloat16) — the optimizer state in the checkpoint assumes bfloat16 autocast; switching to fp16 or fp32 will NaN the loss or mismatch state shapes.
- `policy.*` block — architectural (injection layer, mem length, consolidate type, pad_language_to, etc.). All locked.
- `dataset.*` block — augmentation pipeline is part of the experiment.

**If your GPU can't fit batch 32 in VRAM, do NOT shrink the batch.** Stop and ping Aleks instead — we'll figure out a different hand-off (e.g. finish the run on the 5090, or use gradient checkpointing). Reducing the batch is a silent experiment-killer that's hard to notice until eval numbers come back wrong.

Aleks's machine uses the config as-is with ~8.5 GB VRAM and a 32-thread CPU. A 16 GB RTX 4080 has 2× VRAM headroom. A modern ≥ 8-core CPU can handle `num_workers=8` — if it can't keep up, the symptom is dataloading stalls, not correctness; let the GPU wait rather than tuning knobs.

## 8. Resume training

```bash
python scripts/train.py \
  --config configs/memvla_libero.yaml \
  --resume checkpoints/memvla_libero/step_0030000.pt 2>&1 | tee outputs/memvla_libero/train.log
```

The `--resume` flag loads policy + optimizer + scheduler state, so the run picks up at step 30,001 with the correct LR schedule. Remaining work: 70,000 steps. Expected wall time on a 4080: ~20 hours (≈ 1.0 s/step, vs 0.5 s/step on the 5090).

### Expected startup log

On a clean resume you'll see:
- `Loaded dataset HuggingFaceVLA/libero: N episodes, M frames`
- `tokenizer_padding_verdict: "longest"` in the resolved config JSON — this **must** be `"longest"`; if it says `"max_length"` something is wrong with the tokenizer override path.
- `Resumed from checkpoint: .../step_0030000.pt at step 30000`
- Loss should pick up in the ~0.015–0.035 range, `gate_mean ≈ 0.48`, `gate_std ≈ 0.49`.

### wandb

The trainer will start a new wandb run (not resume the old one) unless you pass an explicit run id. If you want the charts continuous, tell me what wandb entity/project is set up on your side and we can wire `WANDB_RUN_ID` before launching.

## 9. Save and share new checkpoints

New checkpoints land in `checkpoints/memvla_libero/step_00XXXXX.pt` every 5,000 steps. When training completes (or if you pause it), upload the final checkpoint back to the same HF repo so Aleks can pull it for eval:

```bash
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='checkpoints/memvla_libero/final.pt',
    path_in_repo='final.pt',
    repo_id='aleksantari/memvla-libero-ckpts',
    repo_type='model',
)
"
```

## 10. Stopping safely

If you need to stop mid-run, `Ctrl-C` in the training terminal. The trainer does **not** do a mid-step checkpoint on SIGINT — progress since the last 5k checkpoint is lost. Plan pauses around step boundaries where possible.

## Troubleshooting

**Rule of thumb: if something's wrong, stop and ping Aleks — do NOT tune the config to make the error go away.**

- **`CUDA out of memory`** → stop, ping Aleks. Do not reduce batch size. With 16 GB VRAM it should fit; an OOM here means the env or the checkpoint is wrong (e.g. fp32 instead of bf16), not that the batch is too big.
- **DataLoader workers stall / hang** → stop, ping Aleks. Do not change `num_workers`; changing it desyncs the RNG stream and invalidates the experiment. Likely root causes: too few CPU cores, not enough RAM for the prefetch buffer, or a bad LeRobot install.
- **Loss explodes after resume** → the optimizer state didn't load. Check the checkpoint file size matches (~1.5 GB) and your PyTorch major version matches Aleks's (`torch.__version__` at training time — ask if unsure). Do not continue training if loss spikes — stop immediately.
- **`ModuleNotFoundError: lerobot.*`** → you installed the wrong LeRobot fork or forgot `pip install -e ".[smolvla]"` inside it.
- **Training starts but `tokenizer_padding_verdict != "longest"`** → stop, ping Aleks. The tokenizer padding override isn't wired up correctly; continuing trains against a different token distribution than baseline_v2.

## Questions

Ping Aleks on Slack / Telegram. The full design spec is at `memory_smolvla_implementation_spec.md` in the repo root.
