#!/usr/bin/env bash
# Quick sim eval sweep: 2 v5 checkpoints × 4 LIBERO suites × 1 ep/task.
# Run via WSL (osmesa headless rendering). Writes per-suite JSONs to results/
# and a sequential progress log to logs/sim_sweep.log.

set -u  # don't fail on missing dirs etc., but do error on undefined vars

cd "/mnt/c/Users/Tarik Metin/memory-smolVLA"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/home/santari/LIBERO

PY=/home/santari/miniconda3/envs/smolvla/bin/python
LOG=logs/sim_sweep.log
mkdir -p logs

echo "=== sim sweep started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOG"

for CKPT_NAME in v5_run0_diagnostic v5_run1_kitchen_sink; do
    case "$CKPT_NAME" in
        v5_run0_diagnostic)   CFG=configs/libero_v5_run0_diagnostic.yaml ;;
        v5_run1_kitchen_sink) CFG=configs/libero_v5_run1_kitchen_sink.yaml ;;
    esac
    for SUITE in libero_spatial libero_object libero_goal libero_10; do
        echo "=== START $CKPT_NAME / $SUITE  $(date -u +%H:%M:%S) ===" >> "$LOG"
        "$PY" scripts/eval_memory_libero.py \
            --checkpoint "checkpoints/$CKPT_NAME/final.pt" \
            --config "$CFG" \
            --suite "$SUITE" \
            --n-episodes 1 \
            --output-dir "results/${CKPT_NAME}_quick" \
            >> "$LOG" 2>&1
        RC=$?
        echo "=== END $CKPT_NAME / $SUITE  rc=$RC  $(date -u +%H:%M:%S) ===" >> "$LOG"
    done
done

echo "=== sim sweep finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$LOG"
