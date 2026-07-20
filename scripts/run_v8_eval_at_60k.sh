#!/usr/bin/env bash
# Wait for V8 to finish (60k / final.pt), then auto-run the full eval on the
# freed GPU: mem-on, bypass, and mem-on + ensembling. Robust (runs detached like
# training, not a tool-watcher). Launch: nohup bash scripts/run_v8_eval_at_60k.sh > .v8_eval60k.log 2>&1 &
set -u
cd ~/memory-smolVLA
source .venv/bin/activate
set -a; source ~/.v7_env 2>/dev/null; set +a
export MUJOCO_GL=glfw DISPLAY=:0 HF_HUB_DISABLE_PROGRESS_BARS=1

CKPT=checkpoints/memvla_libero_v8/final.pt
CFG=configs/memvla_libero_v8.yaml
OUT=results/v8_eval

echo "[v8eval] waiting for 60k (final.pt) + GPU free ... $(date -u +%H:%M)"
while true; do
  if [ -f "$CKPT" ] && ! pgrep -f "train.py --config configs/memvla_libero_v8" >/dev/null; then
    break
  fi
  sleep 300
done
echo "[v8eval] final.pt present, training done -> starting eval $(date -u +%H:%M)"
sleep 30  # let the GPU settle

echo "[v8eval] === MEM-ON $(date -u +%H:%M) ==="
python scripts/eval_memory_libero.py --checkpoint "$CKPT" --config "$CFG" \
  --all-suites --n-episodes 10 --output-dir "$OUT"

echo "[v8eval] === BYPASS $(date -u +%H:%M) ==="
python scripts/eval_memory_libero.py --checkpoint "$CKPT" --config "$CFG" \
  --all-suites --n-episodes 10 --bypass-memory --output-dir "$OUT"

echo "[v8eval] === MEM-ON + ENSEMBLE $(date -u +%H:%M) ==="
python scripts/eval_memory_libero.py --checkpoint "$CKPT" --config "$CFG" \
  --all-suites --n-episodes 10 --ensemble --query-every 10 --ensemble-decay 0.1 \
  --output-dir "${OUT}_ensemble"

echo "[v8eval] ALL_DONE $(date -u +%H:%M)"
