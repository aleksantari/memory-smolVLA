#!/usr/bin/env bash
# Tight 5 ep × 10 task sweep on v5_meanpool_v4hp with step_increment=1.
# Direct head-to-head vs the no-fix 5ep numbers (results/v5_meanpool_v4hp_5ep/)
# and base 5ep (results/base_smolvla_5ep/) — the rest of Phase 2b's apples-to-apples.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v5_meanpool_stepinc1_5ep.log
mkdir -p logs results/v5_meanpool_v4hp_stepinc1_5ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 5 \
        --step-increment 1 \
        --output-dir results/v5_meanpool_v4hp_stepinc1_5ep \
        >> "$LOG" 2>&1
    echo "=== END ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
