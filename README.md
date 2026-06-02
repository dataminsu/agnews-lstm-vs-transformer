# AG News — LSTM vs Transformer Encoder

Term project for *Deep Learning (2026)*: a controlled comparison of an LSTM
classifier and a Transformer Encoder classifier on the AG News 4-class news
classification task. Both models trained **from scratch** — no pretrained LMs.

> Team: Minsu Kang (lead), Minjeong Song, Hyeju Lee.
> Plan due **2026-05-31**, final report **2026-06-21**, presentation **2026-06-23**.

## Quick start

Python 3.11 (torchtext 0.18 has no 3.13 wheel). Miniconda required.

```powershell
# 1. Build the conda env (one-time)
powershell -ExecutionPolicy Bypass -File code\setup_env.ps1
# or: conda env create -f code\environment.yml

# 2. Activate it
conda activate agnews-dl

# 3. Verify the pipeline (prints split sizes, class distribution, batch shape,
#    runs AverageEmbeddingClassifier smoke test).
cd code
python data_pipeline.py

# 4. Train the LSTM baseline
python -u train_lstm.py
```

Outputs land in `code/outputs/lstm/<tag>/` (`metrics.json`, `history.json`,
`confusion_matrix.npy`, `best.pt`). Final submission notebook is
`code/train_eval.ipynb`.

## Layout

```
code/
  data_pipeline.py     # AG News loader, vocab, DataLoaders (shared by both models)
  models.py            # AverageEmbeddingClassifier (smoke test), LSTMClassifier, TransformerEncoderClassifier
  train_lstm.py        # LSTM baseline + ablation runner
  train_eval.ipynb     # Final submission notebook
  model_guide.md       # Interface / batch contract / hyperparameters for the model teammate
  environment.yml      # conda env spec (Python 3.11)
  requirements.txt     # pinned pip deps (torch 2.3.0 + torchtext 0.18.0 + datasets + sklearn)
  setup_env.ps1        # Windows setup helper (conda)
  train_transformer.py # Transformer baseline + ablation runner
docs/
  2026_term_project.pdf
  assignment1_plan.docx
```

## Fixed experimental conditions

| | Value |
|---|---|
| Dataset source | HuggingFace `ag_news` (single source — no mixing with TorchText/CSV) |
| Split | 90/10 train/val from official train (seed 42) + official test set held out |
| Tokenizer | `torchtext` basic_english |
| Vocab | Built from train only, capped at 20,000, `<pad>=0` / `<unk>=1` |
| Max length | 128 (dynamic batch padding, capped at 128) |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 64 |
| Epochs | 8 |
| Dropout | 0.3 |
| Loss | CrossEntropyLoss on logits |
| Model selection | Best validation macro-F1 (test set held until final eval) |

### Base architectures

- **LSTM**: embed 128, 2-layer unidirectional, hidden 256, last hidden-state pooling.
- **Transformer Encoder**: embed 128 + positional encoding, 2 layers, 4 heads, FF 256, mean-pool over non-pad tokens.

## Required ablation — LSTM side, embedding dimension `64 / 128 / 256`

Hypothesis (plan §5): representation size improves accuracy with diminishing returns; the chosen config is the one at the accuracy plateau, **not the largest**. All other knobs fixed at the base config (hidden=256, 2 layers, dropout=0.3, Adam lr=1e-3, batch=64, 8 epochs, seed=42).

| tag      | embed | params     | best ep | val F1 | **test acc** | **test F1** | t (s) |
|----------|------:|-----------:|--------:|-------:|-------------:|------------:|------:|
| embed64  |    64 |  2,137,092 |       4 | 0.9165 |   **0.9175** |  **0.9174** | 301.5 |
| baseline |   128 |  3,482,628 |       6 | 0.9165 |       0.9172 |      0.9172 | 303.1 |
| embed256 |   256 |  6,173,700 |       4 | 0.9204 |       0.9138 |      0.9137 | 316.3 |

**Reading (LSTM only).** Test F1 is essentially flat between embed=64 (0.9174) and embed=128 (0.9172), and **drops** at embed=256 (0.9137). Validation F1 keeps rising with size (0.9165 → 0.9165 → 0.9204), so the largest model has the best *seen* checkpoint but generalizes worse — a textbook overfit signature visible only because we held the test set back. The accuracy-per-parameter optimum on the LSTM side of this sweep is **embed=64** (≈62% of the baseline's params for an indistinguishable test score). The Transformer-side sweep is owned by a teammate and may show a different plateau.

Raw per-run artifacts (untracked, regenerable): `code/outputs/lstm/<tag>/{metrics.json, history.json, confusion_matrix.npy, best.pt}`.
Reproduce: `python -u code/train_lstm.py --embed-dim 64 --tag embed64` (and 128/256). Aggregate: `python code/summarize_ablation.py --save-md code/ablation_embed_dim.md`.


## Required ablation — Transformer Encoder side, embedding dimension `64 / 128 / 256`

Hypothesis (plan §5): representation size improves accuracy with diminishing returns; the chosen config is the one at the accuracy plateau, **not the largest**. All other knobs fixed at the base config (2 Transformer encoder layers, 4 attention heads, feedforward dimension 256, dropout=0.3, Adam lr=1e-3, batch=64, 8 epochs, seed=42, mean pooling).

| tag      | embed | params    | best ep | val F1 | **test acc** | **test F1** | t (s) |
|----------|------:|----------:|--------:|-------:|-------------:|------------:|------:|
| embed64  |    64 | 1,380,228 |       8 | 0.9230 |   **0.9243** |  **0.9244** | 180.4 |
| baseline |   128 | 2,825,476 |       8 | 0.9214 |       0.9208 |      0.9207 |  98.3 |
| embed256 |   256 | 5,912,580 |       8 | 0.8979 |       0.8970 |      0.8968 | 191.8 |

**Reading (Transformer Encoder only).** The hypothesis was not supported on the Transformer side. Test F1 is highest at embed=64 (0.9244), slightly lower at embed=128 (0.9207), and drops sharply at embed=256 (0.8968). Validation F1 shows the same pattern (0.9230 → 0.9214 → 0.8979), so the larger embedding does not merely overfit the test set; it appears harder to optimize or less effective under the fixed training budget and hyperparameters. The accuracy-per-parameter optimum on the Transformer side is **embed=64**, using about 49% of the baseline model's parameters while achieving the best test F1. Thus, for this setup, increasing representation size did not improve performance; the smallest embedding dimension was both the most accurate and the most parameter-efficient.

Raw per-run artifacts: `code_Transformer/outputs/transformer/<tag>/{metrics.json, history.json, confusion_matrix.npy, best.pt}`.
Reproduce: `python -u train_transformer.py --embed-dim 64 --tag embed64` (and 128/256).


## License

Coursework — released open for inspection. No production use.
