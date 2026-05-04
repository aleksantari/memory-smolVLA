#!/usr/bin/env bash
# 1ep × 10 task canary on v5_compressor_v4hp across all 4 suites.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v5_compressor_canary.log
mkdir -p logs results/v5_compressor_v4hp_1ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_compressor_v4hp/final.pt \
        --config configs/libero_v5_compressor_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --output-dir results/v5_compressor_v4hp_1ep \
        >> "$LOG" 2>&1
    echo "=== END ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
