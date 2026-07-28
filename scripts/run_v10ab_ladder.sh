#!/usr/bin/env bash
# V10 A/B ladder: Coconut latent-reasoning arms on the clean dual-injection base.
#   V10A (K=1): warm-start V10, train cached -> eval
#   V10B (K=2): warm-start V10, train cached -> eval
# Reads V10=74 (libero_10) as the baseline to beat; Gate = V10B - V10A > +5pp.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
set -a; source "$HOME/.v7_env" 2>/dev/null; set +a

run_arm() {
  local name="$1" cfg="$2" ckpt="checkpoints/memvla_libero_$1"
  echo "[ab] === $name train ($(date -u +%H:%M:%S)) ==="
  bash scripts/run_supervised.sh "$cfg" "$ckpt"
  if [ ! -f "$ckpt/final.pt" ]; then echo "[ab] $name no final.pt — stopping."; return 1; fi
  echo "[ab] === $name eval ($(date -u +%H:%M:%S)) ==="
  MUJOCO_GL=glfw DISPLAY=:0 python scripts/eval_memory_libero.py \
    --checkpoint "$ckpt/final.pt" --config "$cfg" \
    --all-suites --n-episodes 10 --ensemble \
    --output-dir "results/${name}_eval" || echo "[ab] $name eval failed (non-fatal)"
}

run_arm v10a configs/memvla_libero_v10a.yaml && \
run_arm v10b configs/memvla_libero_v10b.yaml
echo "[ab] done $(date -u +%H:%M:%S). Compare libero_10: V10=74 vs v10a vs v10b."
