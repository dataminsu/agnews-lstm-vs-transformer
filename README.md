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
| Split | 90/10 train/val from the official train set (seed 42), official test set held out |
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
seeds per cell (42, 43, 44, 45, 46) → 60 runs total. At each stage we hold the
previous-stage winner fixed and pick the next winner by **mean validation
macro-F1 across the 5 seeds**. Per brief §3.3 the test set is held out: it is
never used to choose between cells. Test numbers below are reported only as
the final readout per cell.

Controlled across every cell: HuggingFace `ag_news`, 90/10 train/val split
**re-seeded per `--seed`** (so for any single cell the σ across seeds
aggregates split, init, and shuffle variance — not init/shuffle alone),
torchtext `basic_english` tokenizer, vocab capped at 20,000 built from train
only, max_len=128 with dynamic batch padding, 2-layer unidirectional LSTM with
last-hidden pooling, Adam lr=1e-3, batch=64, 8 epochs, grad_clip=1.0,
CrossEntropyLoss on logits.

### Stage 1 — embedding dimension (hidden=256, dropout=0.3)

| embed_dim | params    | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|----------:|----------:|:--------------------|:------------------|:------------------|------:|
|        32 | 1,464,324 | 0.9165 ± 0.0014     | 0.9136 ± 0.0022   | 0.9135 ± 0.0023   |   233 |
|        64 | 2,137,092 | 0.9170 ± 0.0014     | 0.9141 ± 0.0023   | 0.9141 ± 0.0023   |   255 |
|       128 | 3,482,628 | 0.9182 ± 0.0014     | 0.9149 ± 0.0017   | 0.9148 ± 0.0016   |   250 |
|   **256** | 6,173,700 | **0.9200 ± 0.0013** | 0.9170 ± 0.0032   | 0.9169 ± 0.0032   |   269 |

**Winner: embed_dim = 256** (mean val_f1 0.9200 ± 0.0013). The full-range
effect is statistically resolved: embed=256 beats embed=32 in 5 of 5 seeds
(one-sided paired sign test, p ≈ 0.031). Adjacent-cell deltas are not
individually resolved at n=5 (256 vs 128: 4 of 5 wins, p ≈ 0.19). We carry
embed=256 forward as the highest-mean cell.

### Stage 2 — dropout (at embed=256, hidden=256)

| dropout | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|--------:|:--------------------|:------------------|:------------------|------:|
|     0.1 | 0.9195 ± 0.0009     | 0.9168 ± 0.0017   | 0.9168 ± 0.0016   |   268 |
| **0.3** | **0.9200 ± 0.0013** | 0.9170 ± 0.0032   | 0.9169 ± 0.0032   |   264 |
|     0.5 | 0.9190 ± 0.0015     | 0.9163 ± 0.0026   | 0.9161 ± 0.0026   |   267 |
|     0.8 | 0.9192 ± 0.0019     | 0.9151 ± 0.0006   | 0.9151 ± 0.0004   |   267 |

**Winner: dropout = 0.3** by the selection rule, but the swept range is
**flat**: all four cells sit inside one σ of each other (means 0.9190–0.9200,
σ ≈ 0.0012). With n=5 seeds we cannot statistically separate these dropout
values at this max_len / embed_dim / hidden_size / budget. We carry 0.3
because it has the highest mean and matches the shared Transformer baseline.

**Reproducibility cross-check.** Stage-2 cells at dropout=0.3 across seeds
{42..46} re-run the same config as Stage-1 cells at embed=256 across the same
seeds. The two sets of `metrics.json` are byte-identical except for the `tag`
and `train_time_sec` fields. Verify (no external dep, uses stdlib):

```bash
python -c "
import json
def strip(p):
    d = json.load(open(p)); d['config'].pop('tag', None); d.pop('train_time_sec', None); return d
print('MATCH' if strip('outputs/lstm/stage1_emb256_s42/metrics.json') == strip('outputs/lstm/stage2_drop0.3_s42/metrics.json') else 'DIFF')
"
```

Seeding therefore controls all stochasticity end-to-end and there is no hidden
RNG drift between stages.

### Stage 3 — hidden size (at embed=256, dropout=0.3)

Sweep maps {0.25, 0.5, 0.75, 0.9} of an arbitrary cap of 1024 → {256, 512,
768, 922}.

| hidden  | params     | val_f1 mean±std     | test_acc mean±std | test_f1 mean±std  | t (s) |
|--------:|-----------:|:--------------------|:------------------|:------------------|------:|
|     256 |  6,173,700 | 0.9200 ± 0.0013     | 0.9170 ± 0.0032   | 0.9169 ± 0.0032   |   266 |
|     512 |  8,800,260 | 0.9216 ± 0.0011     | 0.9185 ± 0.0024   | 0.9184 ± 0.0023   |   298 |
|     768 | 12,999,684 | 0.9228 ± 0.0017     | 0.9207 ± 0.0014   | 0.9206 ± 0.0015   |   389 |
| **922** | 16,283,580 | **0.9229 ± 0.0013** | 0.9206 ± 0.0018   | 0.9205 ± 0.0017   |   588 |

**Winner: hidden = 922** (mean val_f1 0.9229 ± 0.0013). Statistical resolution
inside the sweep: hidden=768 beats hidden=256 in 5 of 5 seeds on val_f1
(paired sign test p ≈ 0.031, resolved), but hidden=922 vs hidden=768 is
**unresolved** — 3 of 5 wins, p ≈ 0.5, Δ val_f1 = +0.0001 ≪ 1 σ. The plateau
predicted by plan §5 ("pick the plateau, not the largest") begins near
**hidden ≈ 768**. We therefore also report a **cost-aware alternative
(hidden = 768)** in the headline below: it is statistically indistinguishable
from the winner on validation and trains ≈ 34% faster (389 s vs 588 s per
epoch, training-side info only — no test-set quantity enters this choice).

### Headline LSTM result

Final test-set readout, 5 seeds, n=7,600 test examples. The winner row is the
config picked by the published selection rule (max mean val_f1); the
cost-aware-alternative row is provided because the two are statistically
indistinguishable on validation but the alternative trains ≈ 34% faster.

| config                                                  | val_f1            | test_acc          | test_f1           |
|---------------------------------------------------------|-------------------|-------------------|-------------------|
| **winner** (embed=256, drop=0.3, **hidden=922**)        | 0.9229 ± 0.0013   | 0.9206 ± 0.0018   | 0.9205 ± 0.0017   |
| cost-aware alternative (embed=256, drop=0.3, hidden=768) | 0.9228 ± 0.0017   | 0.9207 ± 0.0014   | 0.9206 ± 0.0015   |

The 3-stage sweep raised the LSTM's mean test F1 from **0.9135 ± 0.0023**
(embed=32 starting cell) to **0.9205 ± 0.0017** at the winning config — a
0.70 pp absolute gain, several cell-stds wide but only ≈ 2× the binomial
standard error on n=7,600 test examples (≈ ±0.32 pp at p ≈ 0.92), so modest
in absolute terms. Treat the LSTM ceiling at this 8-epoch budget as roughly
**0.92 macro-F1**.

Consistent with the plan §5 hypothesis on two of the three axes:
- `embed_dim`: monotone in mean across the swept range, with the full
  32→256 step statistically resolved (5 of 5 wins, paired sign test
  p ≈ 0.031). Adjacent-cell deltas are inside 1 σ.
- `hidden_size`: the 256→768 step is resolved (5 of 5 wins, p ≈ 0.031); the
  768→922 step is **not** resolved (3 of 5 wins, p ≈ 0.5) — i.e. a plateau.
- `dropout`: flat across the swept range; n=5 cannot support or refute the
  hypothesis on this axis.

Per-run artifacts (untracked, regenerable): `code/outputs/lstm/stage{1,2,3}_*_s*/`
with `metrics.json`, `history.json`, `confusion_matrix.npy`, `best.pt`.
Reproduce: `bash run_stage1.sh && bash run_stage2.sh && bash run_stage3.sh`
(resumable — skips any run whose `metrics.json` already exists).
Aggregate: `python summarize_multiseed.py --tag-prefix stage1_emb --sweep embed_dim --save-md ablation_embed_dim.md`
(and analogously for `stage2_drop`/`dropout`, `stage3_hid`/`hidden_size`).

## Required ablation: Transformer side, embedding dimension 64 / 128 / 256

Same hypothesis as above (plan section 5): accuracy should improve with
representation size but with diminishing returns, so we want the plateau rather than
the largest model. Everything else stays at the base config (2 encoder layers, 4
attention heads, feedforward 256, dropout 0.3, Adam lr 1e-3, batch 64, 8 epochs,
seed 42, mean pooling).

| tag      | embed | params    | best ep | val F1 | **test acc** | **test F1** | t (s) |
|----------|------:|----------:|--------:|-------:|-------------:|------------:|------:|
| embed64  |    64 | 1,380,228 |       8 | 0.9230 |   **0.9243** |  **0.9244** | 180.4 |
| baseline |   128 | 2,825,476 |       8 | 0.9214 |       0.9208 |      0.9207 |  98.3 |
| embed256 |   256 | 5,912,580 |       8 | 0.8979 |       0.8970 |      0.8968 | 191.8 |

What the Transformer numbers say. The hypothesis did not hold here. Test F1 is highest
at embed 64 (0.9244), a bit lower at embed 128 (0.9207), and falls off sharply at
embed 256 (0.8968). Validation F1 follows the same shape (0.9230, 0.9214, 0.8979), so
this is not just test-set overfitting. Under the fixed training budget and
hyperparameters, the larger embedding seems harder to train, not better. The best
accuracy-per-parameter point is again embed 64, which reaches the top test F1 with
about 49% of the baseline's parameters. So for this setup a bigger representation did
not help: the smallest embedding was both the most accurate and the most efficient.

Per-run artifacts: `code/transformer/outputs/<tag>/` with `metrics.json`,
`history.json`, `confusion_matrix.npy`, `best.pt`.
Reproduce (from `code/transformer/`): `python -u train_transformer.py --embed-dim 64 --tag embed64` (then 128 and 256).
Aggregate: `python summarize_ablation_transformer.py --save-md ablation_embed_dim_transformer.md`.

## License

Coursework, released open for inspection. Not for production use.
