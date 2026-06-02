# Transformer ablation: embed_dim sweep

| tag | embed_dim | layers | heads | ff | pool | drop | params | best_ep | val_f1 | test_acc | test_f1 | t(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| embed64 | 64 | 2 | 4 | 256 | mean | 0.3 | 1,380,228 | 8 | 0.9230 | 0.9243 | 0.9244 | 180.4 |
| baseline | 128 | 2 | 4 | 256 | mean | 0.3 | 2,825,476 | 8 | 0.9214 | 0.9208 | 0.9207 | 98.3 |
| embed256 | 256 | 2 | 4 | 256 | mean | 0.3 | 5,912,580 | 8 | 0.8979 | 0.8970 | 0.8968 | 191.8 |
