# LSTM ablation: embed_dim sweep

| tag | embed_dim | hidden | L | bidir | drop | params | best_ep | val_f1 | test_acc | test_f1 | t(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| embed64 | 64 | 256 | 2 | False | 0.3 | 2,137,092 | 4 | 0.9165 | 0.9175 | 0.9174 | 301.5 |
| baseline | 128 | 256 | 2 | False | 0.3 | 3,482,628 | 6 | 0.9165 | 0.9172 | 0.9172 | 303.1 |
| embed256 | 256 | 256 | 2 | False | 0.3 | 6,173,700 | 4 | 0.9204 | 0.9138 | 0.9137 | 316.3 |
