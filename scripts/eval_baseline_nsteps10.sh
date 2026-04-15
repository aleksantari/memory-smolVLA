#!/usr/bin/env bash
# Re-evaluate the LIBERO baseline checkpoint at n_action_steps=10 across all 4 suites.
# Produces per-suite JSON under outputs/libero_baseline_eval_n10/<suite>/.
set -euo pipefail

export MUJOCO_GL=egl

CKPT="outputs/libero_baseline/checkpoints/last/pretrained_model"
OUT_BASE="outputs/libero_baseline_eval_n10"
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p "$OUT_BASE"

for suite in "${SUITES[@]}"; do
  echo "=========================================="
  echo "Evaluating $suite"
  echo "=========================================="
  lerobot-eval \
    --policy.path="$CKPT" \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task="$suite" \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --output_dir="$OUT_BASE/$suite" \
    2>&1 | tee "$OUT_BASE/${suite}.log"
done

echo "All evals complete. Results under $OUT_BASE/"
