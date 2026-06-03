#!/usr/bin/env bash
# Stage 3 — hidden_size ablation, 4 values x 5 seeds = 20 runs, at Stage-1+2 winners.
# Stage-1 winner: embed_dim=256 (val_f1 0.9200 +/- 0.0013).
# Stage-2 winner: dropout=0.3  (val_f1 0.9200 +/- 0.0013).
# hidden mapping per the plan: max=1024 -> 0.25/0.5/0.75/0.9 = {256, 512, 768, 922}.
# Other fixed config: 2 layers, Adam lr=1e-3, batch=64, 8 epochs, grad_clip=1.0.
# Tag: stage3_hid<H>_s<seed>  (e.g. stage3_hid256_s42).

set -u
# Override with PYENV=/path/to/python on machines where the conda env python is
# not on PATH. Default assumes `conda activate agnews-dl` already done.
PYENV="${PYENV:-python}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOG_DIR="outputs/lstm/_logs"
mkdir -p "$LOG_DIR"

EMBED_DIM=256
DROPOUT=0.3
HIDDENS=(256 512 768 922)
SEEDS=(42 43 44 45 46)

total=$(( ${#HIDDENS[@]} * ${#SEEDS[@]} ))
i=0
t_overall=$(date +%s)

for hid in "${HIDDENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    tag="stage3_hid${hid}_s${seed}"
    out="outputs/lstm/${tag}"
    log="${LOG_DIR}/${tag}.log"
    if [ -f "${out}/metrics.json" ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  SKIP (exists)"
      continue
    fi
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  START"
    "$PYENV" -u train_lstm.py \
      --embed-dim "$EMBED_DIM" \
      --dropout "$DROPOUT" \
      --hidden-size "$hid" \
      --seed "$seed" \
      --tag "$tag" \
      > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  FAIL rc=${rc} (${dt}s) -- see ${log}"
      exit $rc
    fi
    test_line=$(grep -E '^TEST:' "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  OK (${dt}s) ${test_line}"
  done
done

echo "[$(date +%H:%M:%S)] Stage 3 COMPLETE -- total $(( $(date +%s) - t_overall ))s across ${total} runs"
