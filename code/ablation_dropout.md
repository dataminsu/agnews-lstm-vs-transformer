# lstm multi-seed ablation: dropout sweep

Tag prefix: `stage2_drop`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| dropout | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 0.1 | 5 | 6,173,700 | 0.9193 +/- 0.0007 | 0.9166 +/- 0.0027 | 0.9166 +/- 0.0026 | 80.9 | <-- best val_f1 |
| 0.3 | 5 | 6,173,700 | 0.9190 +/- 0.0017 | 0.9144 +/- 0.0022 | 0.9143 +/- 0.0022 | 80.3 |  |
| 0.5 | 5 | 6,173,700 | 0.9188 +/- 0.0014 | 0.9162 +/- 0.0014 | 0.9162 +/- 0.0014 | 80.7 |  |
| 0.8 | 5 | 6,173,700 | 0.9180 +/- 0.0016 | 0.9128 +/- 0.0017 | 0.9129 +/- 0.0016 | 81.5 |  |

**Winner:** dropout=0.1 (val_f1 0.9193 +/- 0.0007, n=5 seeds).
