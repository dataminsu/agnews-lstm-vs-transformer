# lstm multi-seed ablation: hidden_size sweep

Tag prefix: `stage3_hid`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| hidden_size | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 256 | 5 | 6,173,700 | 0.9193 +/- 0.0023 | 0.9172 +/- 0.0018 | 0.9171 +/- 0.0019 | 177.3 |  |
| 512 | 5 | 8,800,260 | 0.9209 +/- 0.0009 | 0.9190 +/- 0.0014 | 0.9189 +/- 0.0014 | 267.0 |  |
| 768 | 5 | 12,999,684 | 0.9215 +/- 0.0012 | 0.9202 +/- 0.0016 | 0.9201 +/- 0.0016 | 281.9 |  |
| 922 | 5 | 16,283,580 | 0.9226 +/- 0.0013 | 0.9189 +/- 0.0017 | 0.9189 +/- 0.0017 | 385.1 | <-- best val_f1 |

**Winner:** hidden_size=922 (val_f1 0.9226 +/- 0.0013, n=5 seeds).
