#!/usr/bin/env bash
# V10A Step 6 — sequential arm launcher (RevB).
#
# Trains the two Coconut arms back-to-back, each with the same crash/auto-resume
# supervision as the V9 run. GPU-GATED: waits for the V9 training to finish
# (final.pt present OR no train.py process) before starting, so it never contends
# with the running control on the single GB10.
#
# Arms (headline comparison, Gate 1 = K2 − K1 > +5pp on libero_10):
#   Arm 2  configs/memvla_libero_v10a_k1.yaml   (K=1)
#   Arm 3  configs/memvla_libero_v10a_k2.yaml   (K=2, headline)
# Arm 1 (control) is the already-running V9.
#
# Usage: nohup bash scripts/run_v10a_arms.sh > .train_v10a.log 2>&1 &

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
set -a; source "$HOME/.v7_env" 2>/dev/null; set +a
export HF_HUB_DISABLE_PROGRESS_BARS=1

V9_CKPT_DIR="$ROOT/checkpoints/memvla_libero_v9"
MAX_FAILS=12

wait_for_gpu() {
  echo "[v10a] waiting for the GPU (V9 to finish)..."
  while true; do
    if [ -f "$V9_CKPT_DIR/final.pt" ]; then
      echo "[v10a] V9 final.pt present — GPU free."; return 0
    fi
    if ! pgrep -f "[t]rain.py --config" >/dev/null 2>&1; then
      echo "[v10a] no train.py running — GPU free."; return 0
    fi
    sleep 300
  done
}

run_arm() {
  local config="$1" ckpt_dir="$2"
  local fails=0 attempt=0 ck newck rc
  latest() { ls -1 "$ckpt_dir"/step_*.pt 2>/dev/null | sort | tail -1; }
  echo "[v10a] === arm: $config -> $ckpt_dir ==="
  while true; do
    attempt=$((attempt+1))
    [ -f "$ckpt_dir/final.pt" ] && { echo "[v10a] $config: final.pt present — done."; return 0; }
    ck="$(latest)"; local resume=""
    [ -n "$ck" ] && { resume="--resume $ck"; echo "[v10a] attempt $attempt resume $ck"; } \
                 || echo "[v10a] attempt $attempt fresh start"
    # shellcheck disable=SC2086
    python scripts/train.py --config "$config" $resume
    rc=$?
    echo "[v10a] train.py exited rc=$rc at $(date -u +%H:%M:%S)"
    [ -f "$ckpt_dir/final.pt" ] && { echo "[v10a] done after exit."; return 0; }
    [ "$rc" -eq 0 ] && { echo "[v10a] clean exit w/o final.pt — stopping arm."; return 1; }
    newck="$(latest)"
    if [ "$newck" != "$ck" ] && [ -n "$newck" ]; then fails=0; else fails=$((fails+1)); fi
    [ "$fails" -ge "$MAX_FAILS" ] && { echo "[v10a] $fails no-progress fails — giving up."; return 1; }
    echo "[v10a] restart in 15s (fails=$fails)"; sleep 15
  done
}

wait_for_gpu
run_arm configs/memvla_libero_v10a_k1.yaml "$ROOT/checkpoints/memvla_libero_v10a_k1" \
  && run_arm configs/memvla_libero_v10a_k2.yaml "$ROOT/checkpoints/memvla_libero_v10a_k2"
echo "[v10a] arms launcher exiting at $(date -u +%H:%M:%S)."
