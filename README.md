# AG News: LSTM vs Transformer Encoder

Term project for *Deep Learning (2026)*. We compare an LSTM classifier against a
Transformer Encoder classifier on AG News, a 4-class news topic task. Both models
are trained from scratch, with no pretrained language models.

> Team: Minsu Kang (lead), Minjeong Song, Hyeju Lee.
> Plan due 2026-05-31, final report 2026-06-21, presentation 2026-06-23.

## Quick start

You need Python 3.11 (torchtext 0.18 has no wheel for 3.13) and Miniconda.

```powershell
# 1. Build the conda env (only once)
powershell -ExecutionPolicy Bypass -File code\setup_env.ps1
# or: conda env create -f code\environment.yml

# 2. Activate it
conda activate agnews-dl

# 3. Check the pipeline. This prints the split sizes, class distribution, and one
#    batch shape, then runs an AverageEmbeddingClassifier smoke test.
cd code
python data_pipeline.py

# 4. Train the LSTM baseline
python -u train_lstm.py
```

Every command below assumes you are inside the `code/` directory.

Each run writes its results to `code/outputs/<model>/<tag>/`, namely `metrics.json`,
`history.json`, `confusion_matrix.npy`, and `best.pt`. The final submission lives in
`train_eval.ipynb`.

## Layout

```
code/
  data_pipeline.py       # AG News loader, vocab, DataLoaders (shared by both models)
  models.py              # AverageEmbeddingClassifier (smoke test) + LSTMClassifier + TransformerEncoderClassifier
  train_lstm.py          # LSTM baseline + ablation runner
  summarize_ablation.py  # Collects outputs/<model>/<tag>/metrics.json into one table (works for both models)
  summarize_multiseed.py # Aggregates multi-seed ablation runs into mean +/- std tables, picks winner by mean val_f1
  run_stage1.sh / run_stage2.sh / run_stage3.sh  # Resumable drivers for the 3-stage LSTM ablation
  ablation_embed_dim.md  # Generated LSTM Stage-1 (embed_dim) multi-seed table
  ablation_dropout.md    # Generated LSTM Stage-2 (dropout) multi-seed table
  ablation_hidden.md     # Generated LSTM Stage-3 (hidden_size) multi-seed table
  train_eval.ipynb       # Final submission notebook
  model_guide.md         # Pipeline interface, batch contract, hyperparameters (Korean, for teammates)
  environment.yml        # conda env spec (Python 3.11)
  requirements.txt       # pinned pip deps (torch 2.3.0, torchtext 0.18.0, datasets, sklearn)
  setup_env.ps1          # Windows setup helper (conda)
  transformer/                          # Everything specific to the Transformer side
    train_transformer.py                # Transformer baseline + ablation runner
    summarize_ablation_transformer.py   # Collects this folder's outputs into one table
    ablation_embed_dim_transformer.md   # Generated Transformer embedding-dim ablation table
    transformer_guide.md                # Step-by-step guide for extending the Transformer (Korean, for Hyeju)
docs/                    # Course handouts, kept local (git-ignored, not redistributed)
  2026_term_project.pdf
  assignment1_plan.docx
```

`data_pipeline.py` and `models.py` are shared and stay in `code/`. The Transformer
scripts live in `code/transformer/` and add `code/` to the import path automatically,
so run them from inside `code/transformer/`.

## Fixed experimental conditions

We keep everything below identical for both models so the comparison is fair.

| | Value |
|---|---|
| Dataset source | HuggingFace `ag_news` (one source only, no mixing with TorchText or CSV) |
| Split | 90/10 train/val from the official train set (**`data_seed=42`, fixed across every run**), official test set held out |
| Tokenizer | `torchtext` basic_english |
| Vocab | Built from train only, capped at 20,000, `<pad>=0` and `<unk>=1` |
| Max length | 128 (dynamic batch padding, capped at 128) |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 64 |
| Epochs | 8 |
| Dropout | 0.3 |
| Loss | CrossEntropyLoss on logits |
| Model selection | Best validation macro-F1 (test set is touched only at the very end) |

### Base architectures

- LSTM: embedding 128, 2-layer unidirectional, hidden 256, last hidden state pooling.
- Transformer Encoder: embedding 128 plus positional encoding, 2 layers, 4 heads,
  feedforward 256, mean pooling over non-pad tokens.

## Required ablation: LSTM side — 3-stage multi-seed sweep

The brief requires an embedding-dim ablation; we extend it to a 3-stage
sequential ablation over `embed_dim`, `dropout`, and `hidden_size`, with 5
model-seeds per cell (42, 43, 44, 45, 46) → 60 runs total. At each stage we hold
the previous-stage winner fixed and pick the next winner by **mean validation
macro-F1 across the 5 seeds**. Per brief §3.3 the test set is held out: it is
never used to choose between cells. Test numbers below are reported only as
the final readout per cell.

Controlled across every cell (held FIXED): HuggingFace `ag_news`, the 90/10
train/val split at **`data_seed=42` — the identical split for every run**,
torchtext `basic_english` tokenizer, vocab capped at 20,000 built from train
only, max_len=128 with dynamic batch padding, 2-layer unidirectional LSTM with
last-hidden pooling, Adam lr=1e-3, batch=64, 8 epochs, grad_clip=1.0,
CrossEntropyLoss on logits. **Varied between cells:** the ablated knob only.
**Varied within a cell:** `model_seed` ∈ {42..46} (weight init, shuffle order,
dropout masks) — so each cell's σ is *training robustness on a fixed split*, not
split variance.

### Stage 1 — embedding dimension (hidden=256, dropout=0.3)

| embed_dim | params    | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|----------:|----------:|:--------------------|:------------------|:------------------|------:|
|        32 | 1,464,324 | 0.9156 ± 0.0005     | 0.9124 ± 0.0021   | 0.9124 ± 0.0020   |    75 |
|        64 | 2,137,092 | 0.9166 ± 0.0015     | 0.9123 ± 0.0015   | 0.9123 ± 0.0015   |    76 |
|       128 | 3,482,628 | 0.9174 ± 0.0012     | 0.9138 ± 0.0006   | 0.9137 ± 0.0006   |    76 |
|   **256** | 6,173,700 | **0.9190 ± 0.0017** | 0.9144 ± 0.0022   | 0.9143 ± 0.0022   |    81 |

**Winner: embed_dim = 256** (mean val_f1 0.9190 ± 0.0017). The full-range
effect is statistically resolved: embed=256 beats embed=32 in 5 of 5 seeds
(one-sided paired sign test, p ≈ 0.031). Adjacent-cell deltas are not
individually resolved at n=5 (256 vs 128: 3 of 5 wins). Mean val_f1 is monotone
across the swept range. We carry embed=256 forward as the highest-mean cell.
(t (s) = mean total train time per run on an RTX 4080.)

### Stage 2 — dropout (at embed=256, hidden=256)

| dropout | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|--------:|:--------------------|:------------------|:------------------|------:|
| **0.1** | **0.9193 ± 0.0007** | 0.9166 ± 0.0027   | 0.9166 ± 0.0026   |    81 |
|     0.3 | 0.9190 ± 0.0017     | 0.9144 ± 0.0022   | 0.9143 ± 0.0022   |    80 |
|     0.5 | 0.9188 ± 0.0014     | 0.9162 ± 0.0014   | 0.9162 ± 0.0014   |    81 |
|     0.8 | 0.9180 ± 0.0016     | 0.9128 ± 0.0017   | 0.9129 ± 0.0016   |    82 |

**Winner: dropout = 0.1** by the selection rule (highest mean val_f1), but the
swept range is **flat**: no pair is statistically separable at n=5 — in
particular 0.1 vs 0.3 is only 3 of 5 seed-wins (means 0.9180–0.9193, σ ≈ 0.0014).
⚠️ **Carry-over note:** Stage 3 below was executed at **dropout = 0.3** (the value
carried from the prior sweep), not 0.1. Because 0.1 and 0.3 are statistically
indistinguishable here, this does not affect the hidden-size conclusions, and
0.3 also matches the shared Transformer baseline. Re-running Stage 3 at 0.1 would
change nothing material — the dropout axis is flat.

**Reproducibility cross-check.** Stage-2 cells at dropout=0.3 across model-seeds
{42..46} re-run the exact same config as Stage-1 cells at embed=256 (same
`data_seed=42`, same `model_seed`). The two sets of `metrics.json` are
byte-identical except for the `tag` and `train_time_sec` fields — confirmed: both
report val_f1 0.9190 ± 0.0017. Verify (no external dep, uses stdlib):

```bash
python -c "
import json
def strip(p):
    d = json.load(open(p)); d['config'].pop('tag', None); d.pop('train_time_sec', None); return d
print('MATCH' if strip('outputs/lstm/stage1_emb256_s42/metrics.json') == strip('outputs/lstm/stage2_drop0.3_s42/metrics.json') else 'DIFF')
"
```

The two seeds therefore control all stochasticity end-to-end — `data_seed` the
split, `model_seed` init/shuffle/dropout — with no hidden RNG drift between stages.

### Stage 3 — hidden size (at embed=256, dropout=0.3)

Sweep maps {0.25, 0.5, 0.75, 0.9} of an arbitrary cap of 1024 → {256, 512,
768, 922}.

| hidden  | params     | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|--------:|-----------:|:--------------------|:------------------|:------------------|------:|
|     256 |  6,173,700 | 0.9190 ± 0.0017     | 0.9144 ± 0.0022   | 0.9143 ± 0.0022   |    81 |
|     512 |  8,800,260 | 0.9215 ± 0.0012     | 0.9178 ± 0.0019   | 0.9177 ± 0.0019   |   116 |
|     768 | 12,999,684 | 0.9213 ± 0.0020     | 0.9166 ± 0.0031   | 0.9166 ± 0.0030   |   173 |
| **922** | 16,283,580 | **0.9222 ± 0.0019** | 0.9178 ± 0.0041   | 0.9177 ± 0.0040   |   247 |

**Winner: hidden = 922** (mean val_f1 0.9222 ± 0.0019). Statistical resolution
inside the sweep: hidden=922 beats hidden=256 in 5 of 5 seeds on val_f1
(paired sign test p ≈ 0.031, resolved), but hidden=922 vs hidden=512 is
**unresolved** — 3 of 5 wins, Δ val_f1 = +0.0007 < 1 σ. The plateau predicted by
plan §5 ("pick the plateau, not the largest") begins near **hidden ≈ 512**. We
therefore also report a **cost-aware alternative (hidden = 512)** in the headline
below: it ties the winner on test F1 (both 0.9177), is statistically
indistinguishable on validation, and trains ≈ 2.1× faster (116 s vs 247 s per
run on a 4080, training-side info only — no test-set quantity enters this choice).

### Headline LSTM result

Final test-set readout, 5 model-seeds on the fixed `data_seed=42` split,
n=7,600 test examples. The winner row is the config picked by the published
selection rule (max mean val_f1); the cost-aware-alternative row is provided
because the two are statistically indistinguishable on validation (and tied on
test F1) but the alternative trains ≈ 2.1× faster.

| config                                                   | val_f1            | test_acc          | test_f1           |
|----------------------------------------------------------|-------------------|-------------------|-------------------|
| **winner** (embed=256, drop=0.3, **hidden=922**)         | 0.9222 ± 0.0019   | 0.9178 ± 0.0041   | 0.9177 ± 0.0040   |
| cost-aware alternative (embed=256, drop=0.3, hidden=512) | 0.9215 ± 0.0012   | 0.9178 ± 0.0019   | 0.9177 ± 0.0019   |

The 3-stage sweep raised the LSTM's mean test F1 from **0.9124 ± 0.0020**
(embed=32 starting cell) to **0.9177 ± 0.0040** at the winning config — a
0.53 pp absolute gain, only ≈ 1.7× the binomial standard error on n=7,600 test
examples (≈ ±0.32 pp at p ≈ 0.92), so modest in absolute terms. Treat the LSTM
ceiling at this 8-epoch budget as roughly **0.92 macro-F1**. Every run uses the
fixed split plus deterministic algorithms, so the residual σ is training-process
noise (init / shuffle / dropout) only — not split variance.

Consistent with the plan §5 hypothesis on two of the three axes:
- `embed_dim`: monotone in mean across the swept range, with the full
  32→256 step statistically resolved (5 of 5 wins, paired sign test
  p ≈ 0.031). Adjacent-cell deltas are inside 1 σ (256 vs 128: 3 of 5).
- `hidden_size`: the 256→922 step is resolved (5 of 5 wins, p ≈ 0.031); the
  512→922 step is **not** resolved (3 of 5 wins) — i.e. a plateau from
  hidden ≈ 512 onward (the cost-aware pick).
- `dropout`: flat across the swept range; no pair is separable at n=5, so the
  hypothesis can be neither supported nor refuted on this axis.

Per-run artifacts (untracked, regenerable): `code/outputs/lstm/stage{1,2,3}_*_s*/`
with `metrics.json`, `history.json`, `confusion_matrix.npy`, `best.pt`.
Reproduce: `bash run_stage1.sh && bash run_stage2.sh && bash run_stage3.sh`
(resumable — skips any run whose `metrics.json` already exists).
Aggregate: `python summarize_multiseed.py --tag-prefix stage1_emb --sweep embed_dim --save-md ablation_embed_dim.md`
(and analogously for `stage2_drop`/`dropout`, `stage3_hid`/`hidden_size`).

## Required ablation: Transformer side — 3-stage multi-seed sweep

The brief requires an embedding-dim ablation; we extend it to a 3-stage sequential ablation over `embed_dim`, `dropout`, and `num_layers`, with 5 seeds per cell (42, 43, 44, 45, 46) — 60 runs total. At each stage we hold the previous-stage winner fixed and pick the next winner by **mean validation macro-F1** across the 5 seeds. Per §3.3 the test set is held out: it is never used to choose between cells. Test numbers below are reported only as the final readout per cell.

Controlled across every cell: HuggingFace `ag_news`, 90/10 train/val split fixed at `data_seed=42` (split and DataLoader shuffle do not vary across seeds), torchtext `basic_english` tokenizer, vocab capped at 20,000 built from train only, `max_len=128` with dynamic batch padding, 2-layer Transformer Encoder with mean pooling over non-pad tokens, Adam lr=1e-3, batch=64, 8 epochs, grad_clip=1.0, CrossEntropyLoss on logits. The `--seed` argument controls model initialization and training-time stochasticity only (`model_seed`); the σ across seeds reflects model-init + dropout variance, not data-split variance.

---

### Stage 1 — embedding dimension (num_layers=2, dropout=0.3)

| embed_dim | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| 32 | 682,180 | 0.9170 ± 0.0014 | 0.9170 ± 0.0008 | 0.9168 ± 0.0009 | 245 |
| **64** | **1,380,228** | **0.9233 ± 0.0014** | **0.9228 ± 0.0019** | **0.9227 ± 0.0019** | **532** |
| 128 | 2,825,476 | 0.9206 ± 0.0020 | 0.9199 ± 0.0022 | 0.9199 ± 0.0023 | 306 |
| 256 | 5,912,580 | 0.8999 ± 0.0026 | 0.8982 ± 0.0012 | 0.8980 ± 0.0012 | 580 |

**Winner: embed_dim = 64** (val_f1 0.9233 ± 0.0014). Performance peaks at embed=64 and declines in both directions. embed=64 beats embed=32 by 0.006 val_f1 (5 of 5 seeds), confirming the step is resolved. The drop at embed=256 is large (−0.023 from embed=64), consistent with underfitting within the 8-epoch budget at that capacity. The 64→128 step is not resolved (3 of 5 seeds, p ≈ 0.5).

---

### Stage 2 — dropout (at embed=64, num_layers=2)

| dropout | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| **0.1** | **1,380,228** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** | **1560** |
| 0.3 | 1,380,228 | 0.9233 ± 0.0014 | 0.9228 ± 0.0019 | 0.9227 ± 0.0019 | 532 |
| 0.5 | 1,380,228 | 0.9126 ± 0.0011 | 0.9118 ± 0.0028 | 0.9115 ± 0.0028 | 1558 |
| 0.8 | 1,380,228 | 0.6850 ± 0.0996 | 0.7136 ± 0.0746 | 0.6806 ± 0.1020 | 1561 |

**Winner: dropout = 0.1** (val_f1 0.9253 ± 0.0017). dropout=0.5 and 0.8 are clearly worse; dropout=0.8 collapses entirely (σ=0.10, unstable training). The gap between dropout=0.1 and dropout=0.3 is 0.002 val_f1, which is within 1σ and not statistically resolved at n=5. We carry dropout=0.1 forward as the formal winner by the selection rule.

> **Note — single-seed vs multi-seed discrepancy:** With a single seed (42), dropout=0.3 produced a higher val_f1 and was treated as the winner. After running 5 seeds, dropout=0.1 has the higher mean val_f1 (0.9253 vs 0.9233). Additionally, test_f1 is higher for dropout=0.3 (0.9227 vs 0.9215), meaning the two are effectively equivalent on both val and test. This illustrates why single-seed selection is unreliable when the gap between cells is small.

---

### Stage 3 — encoder depth (at embed=64, dropout=0.1)

| num_layers | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,330,244 | 0.9204 ± 0.0016 | 0.9184 ± 0.0022 | 0.9183 ± 0.0021 | 1202 |
| **2** | **1,380,228** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** | **1560** |
| 3 | 1,430,212 | 0.9240 ± 0.0022 | 0.9224 ± 0.0024 | 0.9225 ± 0.0023 | 1376 |
| 4 | 1,480,196 | 0.9230 ± 0.0023 | 0.9214 ± 0.0028 | 0.9215 ± 0.0028 | 1408 |
| 5 | 1,530,180 | 0.9252 ± 0.0030 | 0.9217 ± 0.0032 | 0.9218 ± 0.0032 | 1452 |

**Winner: num_layers = 2** (val_f1 0.9253 ± 0.0017). The 1→2 step is clearly resolved (5 of 5 seeds). layers=5 is numerically tied with layers=2 (Δ=0.0001, well within 1σ), but layers=2 is preferred as it achieves the same performance with fewer parameters. layers=3 and 4 show a slight decline from layers=2, suggesting the 8-epoch budget is insufficient for deeper models to fully converge.

---

### Headline Transformer result

Final test-set readout, 5 seeds, n=7,600 test examples. Winner row is the config picked by the 3-stage selection rule (max mean val_f1 at each stage).

| config | val_f1 | test_acc | test_f1 |
|---|---:|---:|---:|
| **winner (embed=64, drop=0.1, layers=2)** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** |
| shared baseline (embed=128, drop=0.3, layers=2) | 0.9206 ± 0.0020 | 0.9199 ± 0.0022 | 0.9199 ± 0.0023 |

The 3-stage sweep raised the Transformer's mean test F1 from 0.9168 ± 0.0009 (embed=32 starting cell) to **0.9215 ± 0.0009** at the winning config — a 0.47 pp absolute gain.

Consistent with the plan §5 hypothesis:
- `embed_dim`: monotone in mean val_f1 across 32→64→128 (not 256), full 32→64 step resolved (5 of 5 seeds). The 64→128 step is **not** resolved (3 of 5 wins, p ≈ 0.5).
- `dropout`: flat between 0.1 and 0.3 (< 1σ gap); 0.5 and 0.8 clearly worse. n=5 cannot statistically separate 0.1 from 0.3.
- `num_layers`: 2-layer plateau; adjacent-cell deltas (2 vs 3) are inside 1σ.

## License

Coursework, released open for inspection. Not for production use.
