#!/usr/bin/env bash
# Stage 1 — embedding-dim ablation, 4 dims x 5 seeds = 20 runs.
# Fixed base config: hidden=256, 2 layers, dropout=0.3, Adam lr=1e-3,
# batch=64, 8 epochs, grad_clip=1.0. Tag: stage1_emb<dim>_s<seed>.

set -u
PYENV="/c/Users/datam/OneDrive/Desktop/2024Spring/CDS492/agnews-dl/python.exe"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOG_DIR="outputs/lstm/_logs"
mkdir -p "$LOG_DIR"

EMBED_DIMS=(32 64 128 256)
SEEDS=(42 43 44 45 46)

total=$(( ${#EMBED_DIMS[@]} * ${#SEEDS[@]} ))
i=0
t_overall=$(date +%s)

for embed in "${EMBED_DIMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    tag="stage1_emb${embed}_s${seed}"
    out="outputs/lstm/${tag}"
    log="${LOG_DIR}/${tag}.log"
    if [ -f "${out}/metrics.json" ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  SKIP (exists)"
      continue
    fi
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  START"
    "$PYENV" -u train_lstm.py \
      --embed-dim "$embed" \
      --seed "$seed" \
      --tag "$tag" \
      > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  FAIL rc=${rc} (${dt}s) — see ${log}"
      exit $rc
    fi
    test_line=$(grep -E '^TEST:' "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  OK (${dt}s) ${test_line}"
  done
done

echo "[$(date +%H:%M:%S)] Stage 1 COMPLETE — total $(( $(date +%s) - t_overall ))s across ${total} runs"
