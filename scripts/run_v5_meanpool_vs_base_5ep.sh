#!/usr/bin/env bash
# Tight head-to-head: v5_meanpool_v4hp vs base smolvla_libero, 5 ep/task across
# all 4 LIBERO suites. Total ~5-7 h.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v5_vs_base_5ep.log
mkdir -p logs results/v5_meanpool_v4hp_5ep results/base_smolvla_5ep

echo "=== sweep started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"

# 1) v5_meanpool_v4hp — 5 ep × 10 tasks per suite
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START v5 ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 5 \
        --output-dir results/v5_meanpool_v4hp_5ep \
        >> "$LOG" 2>&1
    echo "=== END v5 ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done

# 2) base smolvla_libero (no checkpoint) — same 5 ep × 10 tasks per suite
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START base ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 5 \
        --output-dir results/base_smolvla_5ep \
        >> "$LOG" 2>&1
    echo "=== END base ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done

echo "=== sweep finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
