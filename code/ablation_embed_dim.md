# lstm multi-seed ablation: embed_dim sweep

Tag prefix: `stage1_emb`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| embed_dim | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 32 | 5 | 1,464,324 | 0.9156 +/- 0.0005 | 0.9124 +/- 0.0021 | 0.9124 +/- 0.0020 | 75.4 |  |
| 64 | 5 | 2,137,092 | 0.9166 +/- 0.0015 | 0.9123 +/- 0.0015 | 0.9123 +/- 0.0015 | 76.4 |  |
| 128 | 5 | 3,482,628 | 0.9174 +/- 0.0012 | 0.9138 +/- 0.0006 | 0.9137 +/- 0.0006 | 76.4 |  |
| 256 | 5 | 6,173,700 | 0.9190 +/- 0.0017 | 0.9144 +/- 0.0022 | 0.9143 +/- 0.0022 | 81.2 | <-- best val_f1 |

**Winner:** embed_dim=256 (val_f1 0.9190 +/- 0.0017, n=5 seeds).
