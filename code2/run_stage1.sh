#!/usr/bin/env bash
# Stage 1 — embedding-dim ablation, 4 dims x 5 model-seeds = 20 runs.
# Fixed base config: hidden=256, 2 layers, dropout=0.3, Adam lr=1e-3,
# batch=64, 8 epochs, grad_clip=1.0. Tag: stage1_emb<dim>_s<model_seed>.
# data_seed is FIXED at 42 (identical train/val split every run); only model_seed
# (weight init / shuffle / dropout) varies across {42..46}, so the per-cell sigma
# measures TRAINING robustness on a fixed split, not split variance.

set -u
# Override with PYENV=/path/to/python on machines where the conda env python is
# not on PATH. Default assumes `conda activate agnews-dl` already done.
PYENV="${PYENV:-python}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required for torch deterministic GPU matmuls

LOG_DIR="outputs/lstm/_logs"
mkdir -p "$LOG_DIR"

EMBED_DIMS=(32 64 128 256)
SEEDS=(42 43 44 45 46)   # MODEL seeds (init/shuffle/dropout); data_seed stays fixed at 42

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
      --data-seed 42 \
      --model-seed "$seed" \
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
