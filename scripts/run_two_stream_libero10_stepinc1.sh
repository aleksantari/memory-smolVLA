#!/usr/bin/env bash
# Resume just the missing two_stream libero_10 + step_inc=1 test that got
# killed by Windows sleep (DC sleep was still on with 10min timeout — fixed).
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/two_stream_libero10_stepinc1.log
mkdir -p logs

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
"$PY" scripts/eval_memory_libero_v2.py \
    --checkpoint checkpoints/v5_two_stream_v4hp/final.pt \
    --config configs/libero_v5_two_stream_v4hp.yaml \
    --suite libero_10 \
    --n-episodes 1 \
    --step-increment 1 \
    --output-dir results/v5_two_stream_v4hp_stepinc1_1ep \
    >> "$LOG" 2>&1
echo "=== finished rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
