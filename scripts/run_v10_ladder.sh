#!/usr/bin/env bash
# V10 pipeline orchestrator (cached):
#   1. wait for the prefix cache precompute to finish
#   2. train V10 cached & supervised (resumes from its latest checkpoint)
#   3. eval V10 (all suites + ensemble)
# STOPS before the A/B Coconut arms — scientific gate: confirm dual-injection V10
# beats V9 (libero_10=59) before spending GPU on latent-reasoning arms.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
set -a; source "$HOME/.v7_env" 2>/dev/null; set +a
CACHE="$ROOT/cache/prefix_libero_maxlen"

echo "[ladder] waiting for prefix cache ($CACHE) ..."
while true; do
  if [ -f "$CACHE/meta.json" ] && ! pgrep -f "[p]recompute_prefix" >/dev/null 2>&1; then
    echo "[ladder] cache ready ($(date -u +%H:%M:%S))"; break
  fi
  sleep 60
done

echo "[ladder] training V10 (cached, supervised) ..."
bash scripts/run_supervised.sh configs/memvla_libero_v10.yaml checkpoints/memvla_libero_v10

if [ ! -f checkpoints/memvla_libero_v10/final.pt ]; then
  echo "[ladder] V10 did not finish (no final.pt) — stopping."; exit 1
fi

echo "[ladder] evaluating V10 ($(date -u +%H:%M:%S)) ..."
MUJOCO_GL=glfw DISPLAY=:0 python scripts/eval_memory_libero.py \
  --checkpoint checkpoints/memvla_libero_v10/final.pt \
  --config configs/memvla_libero_v10.yaml \
  --all-suites --n-episodes 10 --ensemble \
  --output-dir results/v10_eval || echo "[ladder] eval failed (non-fatal)"

echo "[ladder] V10 trained + evaluated. A/B (Coconut) arms GATED on review of"
echo "[ladder] results/v10_eval vs V9 (libero_10=59). Launch with:"
echo "[ladder]   bash scripts/run_supervised.sh configs/memvla_libero_v10a.yaml checkpoints/memvla_libero_v10a"
echo "[ladder] done $(date -u +%H:%M:%S)."
