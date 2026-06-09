# Word-order ablation — class-wise F1 + Business<->Sci/Tech confusion

Per-class macro-F1 mean±std over 5 seeds. `B->S%` = share of true Business predicted as Sci/Tech; `S->B%` = true Sci/Tech predicted as Business (row-normalized, the most lexically overlapping pair).

### LSTM — per-class macro-F1 (mean +/- std over 5 seeds)

| order_condition | World | Sports | Business | Sci/Tech | B->S% | S->B% |
|---|---|---|---|---|---|---|
| original | 0.921±0.004 | 0.968±0.003 | 0.882±0.002 | 0.890±0.005 | 8.4 | 6.4 |
| local_shuffle | 0.913±0.004 | 0.961±0.002 | 0.875±0.002 | 0.883±0.003 | 8.9 | 6.8 |
| full_shuffle | 0.909±0.005 | 0.961±0.002 | 0.870±0.003 | 0.881±0.008 | 9.4 | 6.8 |

### Transformer — per-class macro-F1 (mean +/- std over 5 seeds)

| order_condition | World | Sports | Business | Sci/Tech | B->S% | S->B% |
|---|---|---|---|---|---|---|
| original | 0.927±0.002 | 0.971±0.002 | 0.891±0.004 | 0.900±0.003 | 8.2 | 5.3 |
| local_shuffle | 0.928±0.004 | 0.972±0.002 | 0.887±0.002 | 0.899±0.004 | 8.9 | 5.1 |
| full_shuffle | 0.923±0.002 | 0.970±0.002 | 0.884±0.002 | 0.898±0.003 | 8.9 | 5.2 |
