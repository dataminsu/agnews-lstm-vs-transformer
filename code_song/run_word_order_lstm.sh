#!/usr/bin/env bash
# Word-order ablation — LSTM side: 3 conditions x 5 model-seeds = 15 runs.
#
# Fixed baseline config = script defaults (embed=128, hidden=256, 2 layers,
# dropout=0.3, Adam lr=1e-3, batch=64, 8 epochs). data_seed is FIXED at 42 so the
# train/val split is identical every run; only model_seed (init/shuffle/dropout)
# varies across {42..46}. The token-order perturbation is keyed on data_seed, so
# all 5 seeds in a condition see IDENTICAL perturbed data (controlled variable,
# not augmentation).
#
# Resumable: a run whose metrics.json already exists is skipped. Per-run stdout
# goes to outputs/lstm/_logs/<tag>.log. Run from the code/ root:
#   conda activate agnews-dl
#   bash run_word_order_lstm.sh
# Override the interpreter on machines without the env on PATH:
#   PYENV=/path/to/envs/agnews-dl/bin/python bash run_word_order_lstm.sh
set -u
PYENV="${PYENV:-python}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # required for torch deterministic GPU matmuls

LOG_DIR="outputs/lstm/_logs"
mkdir -p "$LOG_DIR"

# "<short>:<--order-condition value>"  (short name keeps tags/folders tidy)
PAIRS=("orig:original" "local:local_shuffle" "full:full_shuffle")
SEEDS=(42 43 44 45 46)   # MODEL seeds; data_seed stays fixed at 42

total=$(( ${#PAIRS[@]} * ${#SEEDS[@]} ))
i=0
t_overall=$(date +%s)

for pair in "${PAIRS[@]}"; do
  short="${pair%%:*}"
  cond="${pair##*:}"
  for seed in "${SEEDS[@]}"; do
    i=$((i + 1))
    tag="wo_${short}_s${seed}"
    out="outputs/lstm/${tag}"
    log="${LOG_DIR}/${tag}.log"
    if [ -f "${out}/metrics.json" ]; then
      echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  SKIP (exists)"
      continue
    fi
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  START (${cond})"
    "$PYENV" -u train_lstm.py \
      --order-condition "$cond" \
      --perturb-window 5 \
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
    echo "[$(date +%H:%M:%S)] (${i}/${total}) ${tag}  OK (${dt}s) $(grep -E '^TEST:' "$log" | tail -1)"
  done
done

echo "[$(date +%H:%M:%S)] LSTM word-order COMPLETE — $(( $(date +%s) - t_overall ))s across ${total} runs"

# Quick per-model readout (combined two-model table comes from summarize_word_order.py).
"$PYENV" summarize_multiseed.py --tag-prefix wo_ --sweep order_condition \
  --save-md ablation_word_order_lstm.md || true
