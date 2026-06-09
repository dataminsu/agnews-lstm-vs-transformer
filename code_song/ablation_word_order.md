# Word Order Perturbation ablation (combined)

Mean +/- std across 5 model-seeds per condition. Delta F1 and Order Sensitivity = (F1_orig - F1_cond)/F1_orig are relative to each model's own original-order mean test_f1. The token-order perturbation is keyed on data_seed, so all seeds in a condition see identical perturbed data.

| model | order_condition | n | val_f1 (mean+/-std) | test_acc (mean+/-std) | test_f1 (mean+/-std) | Delta F1 | Order Sens. |
|---|---|---|---|---|---|---|---|
| LSTM | original | 5 | 0.9181 +/- 0.0013 | 0.9154 +/- 0.0031 | 0.9153 +/- 0.0030 | - | - |
| LSTM | local_shuffle | 5 | 0.9129 +/- 0.0022 | 0.9079 +/- 0.0022 | 0.9079 +/- 0.0023 | -0.0075 | +0.0082 |
| LSTM | full_shuffle | 5 | 0.9077 +/- 0.0010 | 0.9051 +/- 0.0044 | 0.9050 +/- 0.0043 | -0.0103 | +0.0113 |
| Transformer | original | 5 | 0.9214 +/- 0.0003 | 0.9225 +/- 0.0014 | 0.9224 +/- 0.0014 | - | - |
| Transformer | local_shuffle | 5 | 0.9212 +/- 0.0011 | 0.9216 +/- 0.0013 | 0.9214 +/- 0.0014 | -0.0009 | +0.0010 |
| Transformer | full_shuffle | 5 | 0.9204 +/- 0.0010 | 0.9189 +/- 0.0010 | 0.9188 +/- 0.0010 | -0.0036 | +0.0039 |
