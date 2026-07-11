#!/usr/bin/env bash
# Supervised launcher for the V7 week-long run on the DGX Spark.
#
# Runs scripts/train.py to completion, auto-resuming from the latest checkpoint
# if the process dies (e.g. an intermittent pyav decode worker crash over ~6
# days). W&B resumes the same run (trainer uses resume="allow" + fixed name).
# Stops when final.pt appears or after too many consecutive failures.
#
# Usage: nohup bash scripts/run_v7_supervised.sh configs/memvla_libero_v7.yaml > .train_v7.log 2>&1 &

set -u
CONFIG="${1:-configs/memvla_libero_v7.yaml}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# secrets / build env (WANDB_API_KEY etc.)
set -a; source "$HOME/.v7_env" 2>/dev/null; set +a
export HF_HUB_DISABLE_PROGRESS_BARS=1

CKPT_DIR="$ROOT/checkpoints/memvla_libero_v7"
MAX_FAILS=12
fails=0
attempt=0

latest_ckpt() {
  ls -1 "$CKPT_DIR"/step_*.pt 2>/dev/null | sort | tail -1
}

while true; do
  attempt=$((attempt+1))
  if [ -f "$CKPT_DIR/final.pt" ]; then
    echo "[supervisor] final.pt present — run complete."; break
  fi
  RESUME_ARG=""
  ck="$(latest_ckpt)"
  if [ -n "$ck" ]; then
    RESUME_ARG="--resume $ck"
    echo "[supervisor] attempt $attempt: resuming from $ck"
  else
    echo "[supervisor] attempt $attempt: fresh start"
  fi

  # shellcheck disable=SC2086
  python scripts/train.py --config "$CONFIG" $RESUME_ARG
  rc=$?
  echo "[supervisor] train.py exited rc=$rc at $(date -u +%H:%M:%S)"

  if [ -f "$CKPT_DIR/final.pt" ]; then
    echo "[supervisor] final.pt present after exit — done."; break
  fi
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] clean exit without final.pt (unexpected) — stopping."; break
  fi

  # crash: require forward progress (a new checkpoint) to reset the fail counter
  newck="$(latest_ckpt)"
  if [ "$newck" != "$ck" ] && [ -n "$newck" ]; then
    fails=0
  else
    fails=$((fails+1))
  fi
  if [ "$fails" -ge "$MAX_FAILS" ]; then
    echo "[supervisor] $fails consecutive failures with no progress — giving up."; break
  fi
  echo "[supervisor] restarting in 15s (consecutive no-progress fails=$fails)"
  sleep 15
done
echo "[supervisor] exiting."
