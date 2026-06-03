# lstm multi-seed ablation: dropout sweep

Tag prefix: `stage2_drop`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| dropout | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 0.1 | 5 | 6,173,700 | 0.9195 +/- 0.0009 | 0.9168 +/- 0.0017 | 0.9168 +/- 0.0016 | 267.6 |  |
| 0.3 | 5 | 6,173,700 | 0.9200 +/- 0.0013 | 0.9170 +/- 0.0032 | 0.9169 +/- 0.0032 | 264.3 | <-- best val_f1 |
| 0.5 | 5 | 6,173,700 | 0.9190 +/- 0.0015 | 0.9163 +/- 0.0026 | 0.9161 +/- 0.0026 | 267.2 |  |
| 0.8 | 5 | 6,173,700 | 0.9192 +/- 0.0019 | 0.9151 +/- 0.0006 | 0.9151 +/- 0.0004 | 267.3 |  |

**Winner:** dropout=0.3 (val_f1 0.9200 +/- 0.0013, n=5 seeds).
