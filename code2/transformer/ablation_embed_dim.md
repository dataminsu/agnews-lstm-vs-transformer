# transformer multi-seed ablation: embed_dim sweep

Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| embed_dim | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 32 | 5 | 682,180 | 0.9170 +/- 0.0014 | 0.9170 +/- 0.0008 | 0.9168 +/- 0.0009 | 244.9 |  |
| 64 | 5 | 1,380,228 | 0.9233 +/- 0.0014 | 0.9228 +/- 0.0019 | 0.9227 +/- 0.0019 | 532.1 | <-- best val_f1 |
| 128 | 5 | 2,825,476 | 0.9206 +/- 0.0020 | 0.9199 +/- 0.0022 | 0.9199 +/- 0.0023 | 306.0 |  |
| 256 | 5 | 5,912,580 | 0.8999 +/- 0.0026 | 0.8982 +/- 0.0012 | 0.8980 +/- 0.0012 | 580.3 |  |

**Winner:** embed_dim=64 (val_f1 0.9233 +/- 0.0014, n=5 seeds).
