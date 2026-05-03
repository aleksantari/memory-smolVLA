#!/usr/bin/env bash
# Quick eval (1 ep × 10 tasks) across all 4 LIBERO suites for v5_meanpool_v4hp.
# Uses the FIXED eval pipeline (eval_memory_libero_v2.py).
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v5_meanpool_v4hp_full_sweep.log
mkdir -p logs results/v5_meanpool_v4hp_full

echo "=== sweep started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START suite=${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --output-dir results/v5_meanpool_v4hp_full \
        >> "$LOG" 2>&1
    echo "=== END suite=${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== sweep finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
