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
  ablation_embed_dim.md  # Generated LSTM embedding-dim ablation table
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

## Required ablation: LSTM side, embedding dimension 64 / 128 / 256

The plan (section 5) expected accuracy to rise with representation size but with
diminishing returns, so the config we pick should sit at the plateau rather than
being the largest one. Everything else stays at the base config (hidden 256, 2
layers, dropout 0.3, Adam lr 1e-3, batch 64, 8 epochs, seed 42).

| tag      | embed | params     | best ep | val F1 | **test acc** | **test F1** | t (s) |
|----------|------:|-----------:|--------:|-------:|-------------:|------------:|------:|
| embed64  |    64 |  2,137,092 |       4 | 0.9165 |   **0.9175** |  **0.9174** | 301.5 |
| baseline |   128 |  3,482,628 |       6 | 0.9165 |       0.9172 |      0.9172 | 303.1 |
| embed256 |   256 |  6,173,700 |       4 | 0.9204 |       0.9138 |      0.9137 | 316.3 |

What the LSTM numbers say. Test F1 barely moves between embed 64 (0.9174) and embed
128 (0.9172), and it actually drops at embed 256 (0.9137). Validation F1 keeps
climbing with size (0.9165, 0.9165, 0.9204), so the biggest model has the best
validation checkpoint but generalizes worse. That gap is the usual sign of
overfitting, and we only caught it because the test set was kept aside. On the LSTM
side the best accuracy-per-parameter point is embed 64: it uses about 62% of the
baseline's parameters for a test score we can't tell apart. The Transformer sweep is
owned by a teammate and may land on a different plateau.

Per-run artifacts (untracked, regenerable): `code/outputs/lstm/<tag>/` with
`metrics.json`, `history.json`, `confusion_matrix.npy`, `best.pt`.
Reproduce: `python -u train_lstm.py --embed-dim 64 --tag embed64` (then 128 and 256).
Aggregate: `python summarize_ablation.py --save-md ablation_embed_dim.md`.

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
