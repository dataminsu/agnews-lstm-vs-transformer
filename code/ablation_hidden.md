# lstm multi-seed ablation: hidden_size sweep

Tag prefix: `stage3_hid`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| hidden_size | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 256 | 5 | 6,173,700 | 0.9190 +/- 0.0017 | 0.9144 +/- 0.0022 | 0.9143 +/- 0.0022 | 80.7 |  |
| 512 | 5 | 8,800,260 | 0.9215 +/- 0.0012 | 0.9178 +/- 0.0019 | 0.9177 +/- 0.0019 | 115.6 |  |
| 768 | 5 | 12,999,684 | 0.9213 +/- 0.0020 | 0.9166 +/- 0.0031 | 0.9166 +/- 0.0030 | 172.8 |  |
| 922 | 5 | 16,283,580 | 0.9222 +/- 0.0019 | 0.9178 +/- 0.0041 | 0.9177 +/- 0.0040 | 247.4 | <-- best val_f1 |

**Winner:** hidden_size=922 (val_f1 0.9222 +/- 0.0019, n=5 seeds).
