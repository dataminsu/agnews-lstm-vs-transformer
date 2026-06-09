# lstm multi-seed ablation: embed_dim sweep

Tag prefix: `stage1_emb`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| embed_dim | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 32 | 5 | 1,464,324 | 0.9155 +/- 0.0013 | 0.9128 +/- 0.0023 | 0.9128 +/- 0.0022 | 195.7 |  |
| 64 | 5 | 2,137,092 | 0.9165 +/- 0.0011 | 0.9139 +/- 0.0029 | 0.9138 +/- 0.0028 | 213.0 |  |
| 128 | 5 | 3,482,628 | 0.9176 +/- 0.0018 | 0.9137 +/- 0.0025 | 0.9137 +/- 0.0024 | 174.5 |  |
| 256 | 5 | 6,173,700 | 0.9184 +/- 0.0012 | 0.9149 +/- 0.0017 | 0.9148 +/- 0.0017 | 193.3 | <-- best val_f1 |

**Winner:** embed_dim=256 (val_f1 0.9184 +/- 0.0012, n=5 seeds).
