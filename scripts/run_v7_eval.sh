#!/usr/bin/env bash
# Full V7 LIBERO eval on the DGX Spark: mem-on then bypass, all 4 suites.
set -u
cd ~/memory-smolVLA
source .venv/bin/activate
set -a; source ~/.v7_env 2>/dev/null; set +a
export MUJOCO_GL=glfw DISPLAY=:0 HF_HUB_DISABLE_PROGRESS_BARS=1
CK=checkpoints/memvla_libero_v7/final.pt
CFG=configs/memvla_libero_v7.yaml
NEP="${1:-10}"
OUT="${2:-results/v7_eval_flip}"

echo "[eval] === MEM-ON (all suites, ${NEP} ep/task) $(date -u +%H:%M) ==="
python scripts/eval_memory_libero.py --checkpoint "$CK" --config "$CFG" \
  --all-suites --n-episodes "$NEP" --output-dir "$OUT"
echo "[eval] MEM-ON rc=$? $(date -u +%H:%M)"

echo "[eval] === BYPASS (all suites, ${NEP} ep/task) $(date -u +%H:%M) ==="
python scripts/eval_memory_libero.py --checkpoint "$CK" --config "$CFG" \
  --all-suites --n-episodes "$NEP" --bypass-memory --output-dir "$OUT"
echo "[eval] BYPASS rc=$? $(date -u +%H:%M)"
echo "[eval] ALL_DONE $(date -u +%H:%M)"
