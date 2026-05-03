#!/usr/bin/env bash
# Sanity check the WSL eval pipeline using lerobot's official lerobot-eval CLI
# on the upstream HuggingFaceVLA/smolvla_libero baseline. If this reads ~99% on
# libero_object, the pipeline is good and our custom eval script is the bug.

set -u

cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
export HF_HUB_DOWNLOAD_TIMEOUT=300

PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/lerobot_baseline_check.log
mkdir -p logs results/lerobot_baseline_check

echo "=== lerobot-eval baseline started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"

# 1 ep × 10 tasks = 10 episodes for libero_object suite.
"$PY" -m lerobot.scripts.lerobot_eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.device=cuda \
    --policy.use_amp=false \
    --env.type=libero \
    --env.task=libero_object \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --output_dir=results/lerobot_baseline_check \
    >> "$LOG" 2>&1

RC=$?
echo "=== lerobot-eval baseline finished rc=$RC at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
