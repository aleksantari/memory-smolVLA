#!/usr/bin/env bash
# Quick canary on v5_meanpool_v4hp with memory_proj zeroed (memory pathway
# neutralized). Compares to v5_meanpool_v4hp (memory on, but already trained
# to ~5% strength) and base smolvla_libero (no memory).
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/zeroproj_canary.log
mkdir -p logs results/v5_meanpool_v4hp_zeroproj_1ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp_zeroproj/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --output-dir results/v5_meanpool_v4hp_zeroproj_1ep \
        >> "$LOG" 2>&1
    echo "=== END ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
