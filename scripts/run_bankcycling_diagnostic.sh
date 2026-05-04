#!/usr/bin/env bash
# Phase 2b core diagnostic: re-evaluate v5_meanpool_v4hp with
# `--step-increment 1` (matching training's frame-by-frame increment),
# overriding the inference default of 50. If this configuration writes to
# the bank only when current_time % write_stride == 0 (with write_stride=50,
# that's once per 50 callbacks instead of once per callback), the bank
# coverage at inference matches training's 800-frame span.
#
# Compare to the v5_meanpool 5ep numbers:
#   spatial 64 / object 86 / goal 80 / libero_10 30 / overall 65
# Big libero_10 swing here = bank cycling was the bug. Stable libero_10 =
# memory architecture is the bug, not the timing.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/bankcycling_diagnostic.log
mkdir -p logs results/v5_meanpool_v4hp_stepinc1_1ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
        --config configs/libero_v5_meanpool_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --step-increment 1 \
        --output-dir results/v5_meanpool_v4hp_stepinc1_1ep \
        >> "$LOG" 2>&1
    echo "=== END ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done
echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
