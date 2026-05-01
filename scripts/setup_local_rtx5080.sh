#!/usr/bin/env bash
# One-shot local setup for RTX 5080 (Blackwell, sm_120) on Linux.
#
# Prerequisites:
#   * Ubuntu 22.04+ or equivalent
#   * NVIDIA driver 570+ already installed (verify with `nvidia-smi`)
#   * conda available on PATH
#   * git access to github.com:aleksantari/memory-smolVLA.git
#
# Usage:
#   bash scripts/setup_local_rtx5080.sh
#
# What this script does:
#   1. Creates conda env `smolvla` with Python 3.10
#   2. Installs PyTorch 2.7.0 + CUDA 12.8 wheels (required for sm_120)
#   3. Installs LIBERO sim deps (libosmesa6, robosuite, mujoco)
#   4. Installs the project in editable mode + LIBERO benchmark
#   5. Verifies CUDA + GPU detection
#   6. Runs a 100-step smoke test
#
# Re-running is safe (mostly idempotent — pip will skip already-installed pkgs).

set -euo pipefail

# ----- 0. Sanity checks -------------------------------------------------------
echo "===== 0. Sanity checks ====="
command -v conda >/dev/null || { echo "conda not found on PATH. Install Miniconda first."; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found. Install NVIDIA driver first."; exit 1; }

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU detected: $GPU_NAME"
echo "Driver:       $DRIVER"

if [ "$DRIVER" -lt 570 ]; then
    echo "WARNING: NVIDIA driver $DRIVER < 570. RTX 5080 (Blackwell) needs 570+."
    echo "         Install/update the driver first or this script will fail."
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

# ----- 1. Conda env -----------------------------------------------------------
echo
echo "===== 1. Conda env ====="
if conda env list | grep -qE "^smolvla\s"; then
    echo "Env 'smolvla' already exists. Using it."
else
    conda create -n smolvla python=3.10 -y
fi

# Activate (sourcing conda.sh works inside non-interactive shells)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smolvla
echo "Active env: $CONDA_DEFAULT_ENV"

# ----- 2. PyTorch (Blackwell-compatible) --------------------------------------
echo
echo "===== 2. PyTorch 2.7+ with CUDA 12.8 ====="
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "0")
echo "Currently installed torch: $TORCH_VER"

if [[ ! "$TORCH_VER" =~ ^2\.[7-9] ]] && [[ ! "$TORCH_VER" =~ ^[3-9] ]]; then
    echo "Installing torch==2.7.0 + CUDA 12.8 wheels..."
    pip install --upgrade --force-reinstall \
        torch==2.7.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "torch is already $TORCH_VER. Skipping reinstall."
fi

# ----- 3. LIBERO sim deps -----------------------------------------------------
echo
echo "===== 3. LIBERO sim dependencies ====="
echo "Installing apt packages (will prompt for sudo password)..."
sudo apt-get install -y libosmesa6 ffmpeg

echo "Installing pinned mujoco/robosuite..."
pip install -q robosuite==1.4.1 mujoco==3.6.0 scipy

echo "Installing LIBERO benchmark suite..."
pip install -q git+https://github.com/Lifelong-Robot-Learning/LIBERO.git

# ----- 4. Project install -----------------------------------------------------
echo
echo "===== 4. Project install ====="
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "claude/feature/v5-all-fixes" ]; then
    echo "Currently on branch: $CURRENT_BRANCH"
    echo "Switching to claude/feature/v5-all-fixes (the V5 merged branch)..."
    git fetch origin
    git checkout claude/feature/v5-all-fixes
    git pull --ff-only
fi

pip install -e ".[dev]"
pip install -q wandb

# ----- 5. Verification --------------------------------------------------------
echo
echo "===== 5. Verification ====="
python - <<'PY'
import torch
print(f"torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"Device: {dev}")
    print(f"Compute capability: {cap[0]}.{cap[1]}  (sm_{cap[0]}{cap[1]})")
    if cap >= (12, 0):
        print("Blackwell GPU detected. PyTorch 2.7+ is required.")
    # Try to allocate a tensor on GPU
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.T
    print(f"GPU sanity check: tensor op OK ({y.shape})")
else:
    print("ERROR: CUDA not available. Check driver installation.")
    raise SystemExit(1)
PY

# ----- 6. Environment variables -----------------------------------------------
echo
echo "===== 6. Environment variables ====="
echo "Add these to your shell profile or run before training:"
echo
echo "    export MUJOCO_GL=osmesa"
echo "    export PYOPENGL_PLATFORM=osmesa"
echo "    export HF_TOKEN=hf_...     # for HuggingFaceVLA/libero dataset"
echo
read -rp "Set MUJOCO_GL=osmesa for this session? [Y/n] " ans
if [[ ! "$ans" =~ ^[Nn]$ ]]; then
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    echo "MUJOCO_GL set to osmesa for this shell session."
fi

# ----- 7. Smoke test ----------------------------------------------------------
echo
echo "===== 7. Smoke test (100 training steps, ~3 min) ====="
read -rp "Run smoke test now? Requires HF_TOKEN to be set. [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "HF_TOKEN not set. Skipping smoke test — set it and run:"
        echo "    python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml --steps 100"
    else
        python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml --steps 100
    fi
else
    echo "Skipped. Run manually when ready:"
    echo "    python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml --steps 100"
fi

echo
echo "===== Setup complete ====="
echo
echo "Next steps:"
echo "  1. Read HANDOFF.md and NEXT_AGENT_BRIEF.md"
echo "  2. Set HF_TOKEN if you haven't"
echo "  3. Run smoke test (above) if you haven't"
echo "  4. Kick off Run 0:"
echo "       python scripts/train.py --config configs/libero_v5_run0_diagnostic.yaml"
echo "  5. Update RUN_LOG.md as you go."
