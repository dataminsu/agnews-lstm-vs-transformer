# lstm multi-seed ablation: dropout sweep

Tag prefix: `stage2_drop`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| dropout | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 0.1 | 5 | 6,173,700 | 0.9193 +/- 0.0023 | 0.9172 +/- 0.0018 | 0.9171 +/- 0.0019 | 232.5 | <-- best val_f1 |
| 0.3 | 5 | 6,173,700 | 0.9184 +/- 0.0012 | 0.9149 +/- 0.0017 | 0.9148 +/- 0.0017 | 236.1 |  |
| 0.5 | 5 | 6,173,700 | 0.9192 +/- 0.0020 | 0.9144 +/- 0.0023 | 0.9144 +/- 0.0023 | 179.6 |  |
| 0.8 | 5 | 6,173,700 | 0.9185 +/- 0.0011 | 0.9142 +/- 0.0025 | 0.9142 +/- 0.0025 | 200.1 |  |

**Winner:** dropout=0.1 (val_f1 0.9193 +/- 0.0023, n=5 seeds).
