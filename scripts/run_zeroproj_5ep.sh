#!/usr/bin/env bash
# Tight 5 ep × 10 task sweep on the memory_proj-zeroed v5_meanpool checkpoint.
# Compare directly to v5_meanpool 5ep and base 5ep numbers.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/zeroproj_5ep.log
mkdir -p logs results/v5_meanpool_v4hp_zeroproj_5ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp_zeroproj/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 5 \
        --output-dir results/v5_meanpool_v4hp_zeroproj_5ep \
        >> "$LOG" 2>&1
    echo "=== END ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
