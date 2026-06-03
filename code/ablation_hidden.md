# lstm multi-seed ablation: hidden_size sweep

Tag prefix: `stage3_hid`. Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| hidden_size | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 256 | 5 | 6,173,700 | 0.9200 +/- 0.0013 | 0.9170 +/- 0.0032 | 0.9169 +/- 0.0032 | 265.9 |  |
| 512 | 5 | 8,800,260 | 0.9216 +/- 0.0011 | 0.9185 +/- 0.0024 | 0.9184 +/- 0.0023 | 298.0 |  |
| 768 | 5 | 12,999,684 | 0.9228 +/- 0.0017 | 0.9207 +/- 0.0014 | 0.9206 +/- 0.0015 | 388.8 |  |
| 922 | 5 | 16,283,580 | 0.9229 +/- 0.0013 | 0.9206 +/- 0.0018 | 0.9205 +/- 0.0017 | 587.9 | <-- best val_f1 |

**Winner:** hidden_size=922 (val_f1 0.9229 +/- 0.0013, n=5 seeds).
