#!/usr/bin/env bash
# Generic supervised launcher — runs a config to completion with crash/auto-resume,
# resuming from the latest checkpoint if train.py dies (e.g. a pyav worker crash).
# Stops on final.pt or too many consecutive no-progress failures.
#
# Usage: nohup bash scripts/run_supervised.sh <config.yaml> <checkpoint_dir> > <log> 2>&1 &
set -u
CONFIG="${1:?usage: run_supervised.sh <config> <ckpt_dir>}"
CKPT_DIR="${2:?usage: run_supervised.sh <config> <ckpt_dir>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
set -a; source "$HOME/.v7_env" 2>/dev/null; set +a
export HF_HUB_DISABLE_PROGRESS_BARS=1
[ "$(realpath "$CKPT_DIR")" = "$CKPT_DIR" ] || CKPT_DIR="$ROOT/$CKPT_DIR"

MAX_FAILS=12; fails=0; attempt=0
latest() { ls -1 "$CKPT_DIR"/step_*.pt 2>/dev/null | sort | tail -1; }
while true; do
  attempt=$((attempt+1))
  [ -f "$CKPT_DIR/final.pt" ] && { echo "[sup] final.pt present — done."; break; }
  ck="$(latest)"; resume=""
  [ -n "$ck" ] && { resume="--resume $ck"; echo "[sup] attempt $attempt resume $ck"; } \
               || echo "[sup] attempt $attempt fresh start ($CONFIG)"
  # shellcheck disable=SC2086
  python scripts/train.py --config "$CONFIG" $resume
  rc=$?
  echo "[sup] train.py exited rc=$rc at $(date -u +%H:%M:%S)"
  [ -f "$CKPT_DIR/final.pt" ] && { echo "[sup] done after exit."; break; }
  [ "$rc" -eq 0 ] && { echo "[sup] clean exit w/o final.pt — stopping."; break; }
  newck="$(latest)"
  if [ "$newck" != "$ck" ] && [ -n "$newck" ]; then fails=0; else fails=$((fails+1)); fi
  [ "$fails" -ge "$MAX_FAILS" ] && { echo "[sup] $fails no-progress fails — giving up."; break; }
  echo "[sup] restart in 15s (fails=$fails)"; sleep 15
done
echo "[sup] exiting."
