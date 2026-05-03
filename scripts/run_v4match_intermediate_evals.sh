#!/usr/bin/env bash
# Eval v5_v4match intermediate checkpoints to confirm gradual drift from
# overcooked expert_lr=1e-4 (vs baseline 70% on libero_object).
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v4match_intermediate.log
mkdir -p logs

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for STEP in 0005000 0015000 0025000; do
    CKPT="checkpoints/v5_v4match/step_${STEP}.pt"
    echo "=== START step ${STEP}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint "$CKPT" \
        --config configs/libero_v5_v4match.yaml \
        --suite libero_object \
        --n-episodes 1 \
        --output-dir results/v4match_intermediate \
        >> "$LOG" 2>&1
    echo "=== END step ${STEP}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
