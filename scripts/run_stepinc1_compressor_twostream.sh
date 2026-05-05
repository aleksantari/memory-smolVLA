#!/usr/bin/env bash
# Cross-check: apply the step_increment=1 fix to compressor + two_stream
# and verify libero_10 jumps similarly to v5_meanpool's 30 -> 50 jump.
# 1 ep x 10 tasks per suite per variant. ~80 min total.
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/stepinc1_compressor_twostream.log
mkdir -p logs results/v5_compressor_v4hp_stepinc1_1ep results/v5_two_stream_v4hp_stepinc1_1ep

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"

# 1) compressor with step_inc=1
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START compressor ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_compressor_v4hp/final.pt \
        --config configs/libero_v5_compressor_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --step-increment 1 \
        --output-dir results/v5_compressor_v4hp_stepinc1_1ep \
        >> "$LOG" 2>&1
    echo "=== END compressor ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done

# 2) two_stream with step_inc=1
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== START two_stream ${SUITE}  $(date -u +%H:%M:%S) ===" >> "$LOG"
    "$PY" scripts/eval_memory_libero_v2.py \
        --checkpoint checkpoints/v5_two_stream_v4hp/final.pt \
        --config configs/libero_v5_two_stream_v4hp.yaml \
        --suite "$SUITE" \
        --n-episodes 1 \
        --step-increment 1 \
        --output-dir results/v5_two_stream_v4hp_stepinc1_1ep \
        >> "$LOG" 2>&1
    echo "=== END two_stream ${SUITE}  rc=$?  $(date -u +%H:%M:%S) ===" >> "$LOG"
done

echo "=== finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
