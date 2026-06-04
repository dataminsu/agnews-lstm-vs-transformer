# transformer multi-seed ablation: num_layers_d01 sweep

Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).

| num_layers | n | params | val_f1 (mean +/- std) | test_acc (mean +/- std) | test_f1 (mean +/- std) | t(s) mean | winner |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 1,330,244 | 0.9204 +/- 0.0016 | 0.9184 +/- 0.0022 | 0.9183 +/- 0.0021 | 1202.1 |  |
| 2 | 5 | 1,380,228 | 0.9253 +/- 0.0017 | 0.9216 +/- 0.0009 | 0.9215 +/- 0.0009 | 1559.5 | <-- best val_f1 |
| 3 | 5 | 1,430,212 | 0.9240 +/- 0.0022 | 0.9224 +/- 0.0024 | 0.9225 +/- 0.0023 | 1375.6 |  |
| 4 | 5 | 1,480,196 | 0.9230 +/- 0.0023 | 0.9214 +/- 0.0028 | 0.9215 +/- 0.0028 | 1407.6 |  |
| 5 | 5 | 1,530,180 | 0.9252 +/- 0.0030 | 0.9217 +/- 0.0032 | 0.9218 +/- 0.0032 | 1452.0 |  |

**Winner:** num_layers=2 (val_f1 0.9253 +/- 0.0017, n=5 seeds).
