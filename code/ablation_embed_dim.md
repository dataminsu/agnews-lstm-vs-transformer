# lstm multi-seed ablation: embed_dim sweep

Tag prefix: `stage1_emb`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| embed_dim | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 32 | 5 | 1,464,324 | 0.9165 +/- 0.0014 | 0.9136 +/- 0.0022 | 0.9135 +/- 0.0023 | 233.3 |  |
| 64 | 5 | 2,137,092 | 0.9170 +/- 0.0014 | 0.9141 +/- 0.0023 | 0.9141 +/- 0.0023 | 255.2 |  |
| 128 | 5 | 3,482,628 | 0.9182 +/- 0.0014 | 0.9149 +/- 0.0017 | 0.9148 +/- 0.0016 | 250.3 |  |
| 256 | 5 | 6,173,700 | 0.9200 +/- 0.0013 | 0.9170 +/- 0.0032 | 0.9169 +/- 0.0032 | 269.2 | <-- best val_f1 |

**Winner:** embed_dim=256 (val_f1 0.9200 +/- 0.0013, n=5 seeds).
