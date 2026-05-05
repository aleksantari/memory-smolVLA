#!/usr/bin/env bash
# Resume just libero_10 of the v5_meanpool step_inc=1 5ep sweep.
# (Other suites already saved in results/v5_meanpool_v4hp_stepinc1_5ep/.)
set -u
cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=/home/santari/LIBERO
PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/v5_meanpool_libero10_stepinc1_5ep.log
mkdir -p logs

echo "=== started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"
"$PY" scripts/eval_memory_libero_v2.py \
    --checkpoint checkpoints/v5_meanpool_v4hp/final.pt \
    --config configs/libero_v5_meanpool_v4hp.yaml \
    --suite libero_10 \
    --n-episodes 5 \
    --step-increment 1 \
    --output-dir results/v5_meanpool_v4hp_stepinc1_5ep \
    >> "$LOG" 2>&1
echo "=== finished rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
