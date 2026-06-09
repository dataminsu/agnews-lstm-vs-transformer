#!/usr/bin/env bash
# Word-order ablation — Transformer side: 3 conditions x 5 seeds = 15 runs.
#
# Fixed baseline config = script defaults (embed=128, nhead=4, 2 layers,
# ff=256, dropout=0.3, Adam lr=1e-3, batch=64, 8 epochs). --seed varies model
# init/dropout across {42..46} (the existing Transformer multi-seed convention);
# data_seed stays fixed at 42, and the token-order perturbation is keyed on it, so
# all 5 seeds in a condition see IDENTICAL perturbed data.
#
# Resumable: a run whose metrics.json already exists is skipped. Run from the
# code/ root (the script cd's into transformer/ itself):
#   conda activate agnews-dl
#   bash run_word_order_transformer.sh
# On a 2-GPU box you can run this and the LSTM script at the same time:
#   CUDA_VISIBLE_DEVICES=0 bash run_word_order_lstm.sh &
#   CUDA_VISIBLE_DEVICES=1 bash run_word_order_transformer.sh &
set -u
PYENV="${PYENV:-python}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$(dirname "$0")/transformer" || exit 1
LOG_DIR="outputs/transformer/_logs"
mkdir -p "$LOG_DIR"

PAIRS=("orig:original" "local:local_shuffle" "full:full_shuffle")
SEEDS=(42 43 44 45 46)

total=$(( ${#PAIRS[@]} * ${#SEEDS[@]} ))
i=0
t_overall=$(date +%s)

for pair in "${PAIRS[@]}"; do
  short="${pair%%:*}"
  cond="${pair##*:}"
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    tag="wo_${short}_s${seed}"
    out="outputs/transformer/${tag}"
    log="${LOG_DIR}/${tag}.log"
    if [ -f "${out}/metrics.json" ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  SKIP (exists)"
      continue
    fi
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  START (${cond})"
    # --save-plots only on the original-order runs (loss curves + confusion
    # matrices for the report figures); shuffled runs still write confusion_matrix.npy.
    extra=""
    [ "$short" = "orig" ] && extra="--save-plots"
    "$PYENV" -u train_transformer.py \
      --order-condition "$cond" \
      --perturb-window 5 \
      --seed "$seed" \
      --tag "$tag" \
      $extra \
      > "$log" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  FAIL rc=${rc} (${dt}s) — see ${log}"
      exit $rc
    fi
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  OK (${dt}s) $(grep -E '^TEST:' "$log" | tail -1)"
  done
done

echo "[$(date +%H:%M:%S)] Transformer word-order COMPLETE — $(( $(date +%s) - t_overall ))s across ${total} runs"
