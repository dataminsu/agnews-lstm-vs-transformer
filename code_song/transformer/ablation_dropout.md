# transformer multi-seed ablation: dropout sweep

Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| dropout | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 0.1 | 5 | 1,380,228 | 0.9253 +/- 0.0017 | 0.9216 +/- 0.0009 | 0.9215 +/- 0.0009 | 1559.5 | <-- best val_f1 |
| 0.3 | 5 | 1,380,228 | 0.9233 +/- 0.0014 | 0.9228 +/- 0.0019 | 0.9227 +/- 0.0019 | 532.1 |  |
| 0.5 | 5 | 1,380,228 | 0.9126 +/- 0.0011 | 0.9118 +/- 0.0028 | 0.9115 +/- 0.0028 | 1557.5 |  |
| 0.8 | 5 | 1,380,228 | 0.6850 +/- 0.0996 | 0.7136 +/- 0.0746 | 0.6806 +/- 0.1020 | 1561.2 |  |

**Winner:** dropout=0.1 (val_f1 0.9253 +/- 0.0017, n=5 seeds).
