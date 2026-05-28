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

## Required ablation

Embedding dimension sweep `64 / 128 / 256`, identical for both models. Hypothesis: representation size improves accuracy with diminishing returns.

## License

Coursework — released open for inspection. No production use.
