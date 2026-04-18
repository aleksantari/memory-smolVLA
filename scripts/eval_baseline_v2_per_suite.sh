#!/usr/bin/env bash
# Re-run v2 baseline eval per-suite on the 100K checkpoint.
# Each suite gets its own lerobot-eval call and its own JSON under
# outputs/libero_baseline_v2_eval_per_suite/<suite>/ so the per_group
# aggregation log line is never truncated.
set -euo pipefail

export MUJOCO_GL=egl

LEROBOT_EVAL="/home/santari/miniconda3/envs/lerobot/bin/lerobot-eval"
CKPT="outputs/libero_baseline_v2/checkpoints/100000/pretrained_model"
OUT_BASE="outputs/libero_baseline_v2_eval_per_suite"
SUITES=(libero_spatial libero_object libero_goal libero_10)

mkdir -p "$OUT_BASE"

for suite in "${SUITES[@]}"; do
  echo "=========================================="
  echo "Evaluating $suite"
  echo "=========================================="
  "$LEROBOT_EVAL" \
    --policy.path="$CKPT" \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task="$suite" \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --output_dir="$OUT_BASE/$suite" \
    2>&1 | tee "$OUT_BASE/${suite}.log"
done

echo "All per-suite evals complete. Results under $OUT_BASE/"
