# AG News: LSTM vs Transformer Encoder

Term project for *Deep Learning (2026)*. We compare an LSTM classifier against a
Transformer Encoder classifier on AG News, a 4-class news topic task. Both models
are trained from scratch, with no pretrained language models.

> Team: Minsu Kang (lead), Minjeong Song, Hyeju Lee.
> Plan due 2026-05-31, final report 2026-06-21, presentation 2026-06-23.

## Quick start

You need Python 3.11 (torchtext 0.18 has no wheel for 3.13) and Miniconda.

```powershell
# 1. Build the conda env (only once)
powershell -ExecutionPolicy Bypass -File code\setup_env.ps1
# or: conda env create -f code\environment.yml

# 2. Activate it
conda activate agnews-dl

# 3. Check the pipeline. This prints the split sizes, class distribution, and one
#    batch shape, then runs an AverageEmbeddingClassifier smoke test.
cd code
python data_pipeline.py

# 4. Train the LSTM baseline
python -u train_lstm.py
```

Every command below assumes you are inside the `code/` directory.

Each run writes its results to `code/outputs/<model>/<tag>/`, namely `metrics.json`,
`history.json`, `confusion_matrix.npy`, and `best.pt`. The final submission lives in
`train_eval.ipynb`.

## Layout

```
code/
  data_pipeline.py       # AG News loader, vocab, DataLoaders (shared by both models)
  models.py              # AverageEmbeddingClassifier (smoke test) + LSTMClassifier + TransformerEncoderClassifier
  train_lstm.py          # LSTM baseline + ablation runner
  summarize_ablation.py  # Collects outputs/<model>/<tag>/metrics.json into one table (works for both models)
  summarize_multiseed.py # Aggregates multi-seed ablation runs into mean +/- std tables, picks winner by mean val_f1
  run_stage1.sh / run_stage2.sh / run_stage3.sh  # Resumable drivers for the 3-stage LSTM ablation
  ablation_embed_dim.md  # Generated LSTM Stage-1 (embed_dim) multi-seed table
  ablation_dropout.md    # Generated LSTM Stage-2 (dropout) multi-seed table
  ablation_hidden.md     # Generated LSTM Stage-3 (hidden_size) multi-seed table
  train_eval.ipynb       # Final submission notebook
  model_guide.md         # Pipeline interface, batch contract, hyperparameters (Korean, for teammates)
  environment.yml        # conda env spec (Python 3.11)
  requirements.txt       # pinned pip deps (torch 2.3.0, torchtext 0.18.0, datasets, sklearn)
  setup_env.ps1          # Windows setup helper (conda)
  transformer/                          # Everything specific to the Transformer side
    train_transformer.py                # Transformer baseline + ablation runner
    summarize_ablation_transformer.py   # Collects this folder's outputs into one table
    ablation_embed_dim_transformer.md   # Generated Transformer embedding-dim ablation table
    transformer_guide.md                # Step-by-step guide for extending the Transformer (Korean, for Hyeju)
docs/                    # Course handouts, kept local (git-ignored, not redistributed)
  2026_term_project.pdf
  assignment1_plan.docx
```

`data_pipeline.py` and `models.py` are shared and stay in `code/`. The Transformer
scripts live in `code/transformer/` and add `code/` to the import path automatically,
so run them from inside `code/transformer/`.

## Fixed experimental conditions

We keep everything below identical for both models so the comparison is fair.

| | Value |
|---|---|
| Dataset source | HuggingFace `ag_news` (one source only, no mixing with TorchText or CSV) |
| Split | 90/10 train/val from the official train set (**`data_seed=42`, fixed across every run**), official test set held out |
| Tokenizer | `torchtext` basic_english |
| Vocab | Built from train only, capped at 20,000, `<pad>=0` and `<unk>=1` |
| Max length | 128 (dynamic batch padding, capped at 128) |
| Optimizer | Adam, lr 1e-3 |
| Batch size | 64 |
| Epochs | 8 |
| Dropout | 0.3 |
| Loss | CrossEntropyLoss on logits |
| Model selection | Best validation macro-F1 (test set is touched only at the very end) |

### Base architectures

- LSTM: embedding 128, 2-layer unidirectional, hidden 256, last hidden state pooling.
- Transformer Encoder: embedding 128 plus positional encoding, 2 layers, 4 heads,
  feedforward 256, mean pooling over non-pad tokens.

## LSTM 실험 결과 — 3단계 순차 Ablation

과제는 임베딩 차원 ablation을 요구하며, 이를 `embed_dim` → `dropout` → `hidden_size`
**3단계 순차 ablation**으로 확장했다. 각 셀은 **model_seed 5개(42–46)** 로 학습해
평균±표준편차로 보고하며(총 60런), 각 단계의 우승은 **검증 macro-F1 평균**으로 고른
뒤 다음 단계에 고정한다. test set은 셀 선택에 절대 쓰지 않고 **최종 확인용**으로만
1회 평가한다.

**모든 셀에서 고정(공정 비교):** HuggingFace `ag_news`, 90/10 train/val 분할
(`data_seed=42` 동일 분할), `basic_english` 토크나이저, vocab 20,000(train만으로 구축),
max_len=128 동적 패딩, 2층 단방향 LSTM(마지막 hidden 풀링), Adam lr=1e-3, batch=64,
8 epoch, grad_clip=1.0, CrossEntropyLoss. **셀 간 변화:** 해당 축 하나만.
**셀 내 변화:** `model_seed`(가중치 초기화·셔플·드롭아웃) → 각 셀의 σ는 고정 분할 위
**학습 과정 견고성**을 의미(분할 변동이 아님). t(s)는 RTX 3080 Ti 기준 런당 평균 학습 시간.

### Stage 1 — 임베딩 차원 (hidden=256, dropout=0.3)

| embed_dim | 파라미터 | val_f1 (평균±σ) | test_acc | test_f1 | t(s) |
|----------:|----------:|:--------------------|:------------------|:------------------|------:|
|        32 | 1,464,324 | 0.9155 ± 0.0013     | 0.9128 ± 0.0023   | 0.9128 ± 0.0022   |   196 |
|        64 | 2,137,092 | 0.9165 ± 0.0011     | 0.9139 ± 0.0029   | 0.9138 ± 0.0028   |   213 |
|       128 | 3,482,628 | 0.9176 ± 0.0018     | 0.9137 ± 0.0025   | 0.9137 ± 0.0024   |   175 |
|   **256** | 6,173,700 | **0.9184 ± 0.0012** | 0.9149 ± 0.0017   | 0.9148 ± 0.0017   |   193 |

**① 최고 성능:** `embed_dim = 256` (val_f1 0.9184, test_f1 0.9148) → 다음 단계로 고정.

**② 종합·이유:** 32 → 256으로 갈수록 평균 성능이 **단조 증가**한다(0.9155 → 0.9184). 임베딩
차원이 커질수록 토큰 의미를 담는 표현 공간이 넓어져 분류 단서가 풍부해지기 때문이다. 다만
인접 셀 차이(예: 128→256)는 표준편차(σ≈0.0012~0.0018) 안이라 통계적으로 또렷하지 않고,
**32→256 전 구간 차이(+0.0029)만 분명**하다. 이 8 epoch 예산에서는 256에서 이미 수확체감이 시작된다.

### Stage 2 — 드롭아웃 (embed=256, hidden=256)

| dropout | val_f1 (평균±σ) | test_acc | test_f1 | t(s) |
|--------:|:--------------------|:------------------|:------------------|------:|
| **0.1** | **0.9193 ± 0.0023** | 0.9172 ± 0.0018   | 0.9171 ± 0.0019   |   233 |
|     0.3 | 0.9184 ± 0.0012     | 0.9149 ± 0.0017   | 0.9148 ± 0.0017   |   236 |
|     0.5 | 0.9192 ± 0.0020     | 0.9144 ± 0.0023   | 0.9144 ± 0.0023   |   180 |
|     0.8 | 0.9185 ± 0.0011     | 0.9142 ± 0.0025   | 0.9142 ± 0.0025   |   200 |

**① 최고 성능:** `dropout = 0.1` (val_f1 0.9193).

**② 종합·이유:** 0.1~0.8 구간이 **거의 평평**하다(전체 범위 0.9184~0.9193, σ 안 → 통계적으로
무차별). 특히 0.1과 0.5가 사실상 동률(0.9193 vs 0.9192)이다. 모델 규모(6.2M)와 8 epoch
예산에서는 심한 과적합이 일어나지 않아 드롭아웃의 정규화 효과가 미미하고, 0.8처럼 과하면
오히려 정보 손실로 소폭 하락한다. 즉 **이 문제에서 드롭아웃은 민감하지 않은 축**이다.
이번 실험에서는 **Stage 3를 이 우승값 0.1에서 실행**해 "이전 단계 우승값을 고정한다"는
3단계 프로토콜을 일관되게 지켰다.

### Stage 3 — 은닉 크기 (embed=256, dropout=0.1)

은닉 크기는 임의 상한 1024의 {0.25, 0.5, 0.75, 0.9} = {256, 512, 768, 922}로 스윕했다.
(Stage 2 우승값 **dropout=0.1**을 고정 — 프로토콜 일관성.)

| hidden  | 파라미터 | val_f1 (평균±σ) | test_acc | test_f1 | t(s) |
|--------:|-----------:|:--------------------|:------------------|:------------------|------:|
|     256 |  6,173,700 | 0.9193 ± 0.0023     | 0.9172 ± 0.0018   | 0.9171 ± 0.0019   |   177 |
|     512 |  8,800,260 | 0.9209 ± 0.0009     | 0.9190 ± 0.0014   | 0.9189 ± 0.0014   |   267 |
|     768 | 12,999,684 | 0.9215 ± 0.0012     | **0.9202 ± 0.0016** | **0.9201 ± 0.0016** |   282 |
| **922** | 16,283,580 | **0.9226 ± 0.0013** | 0.9189 ± 0.0017   | 0.9189 ± 0.0017   |   385 |

**① 최고 성능:** `hidden = 922` (val_f1 0.9226 — 선택 규칙은 val 기준).

**② 종합·이유:** 256 → 512에서 뚜렷이 향상(+0.0016 val)되고, **512~922 구간은 평평
(plateau)**하다. 즉 표현 용량이 512 부근에서 이미 포화되어 그 이상 키워도 평균 성능은
거의 늘지 않는다. 흥미롭게도 **val은 922가 최고지만 test는 768이 최고(0.9201)**이고,
512는 test에서 922와 동률(0.9189)이다. 따라서 922가 (val 기준) 명목 우승이나,
**hidden=512가 비용효율 최적** — test가 922와 같으면서 파라미터는 0.54배, 학습은 약
1.4배 빠르다(267s vs 385s). 큰 모델일수록 과적합 신호(낮은 best epoch)도 함께 나타난다.

### 최종 선택 모델

| 설정 | val_f1 | test_acc | test_f1 |
|------|-------------------|-------------------|-------------------|
| **우승** (embed=256, **drop=0.1**, **hidden=922**)     | 0.9226 ± 0.0013 | 0.9189 ± 0.0017 | 0.9189 ± 0.0017 |
| 비용효율 대안 (embed=256, drop=0.1, hidden=512)        | 0.9209 ± 0.0009 | 0.9190 ± 0.0014 | 0.9189 ± 0.0014 |

3단계 sweep으로 LSTM 평균 test F1을 **0.9128(시작 셀 embed=32) → 0.9189(우승)** 으로
+0.61pp 끌어올렸다(절대값으로는 완만한 향상). **8 epoch 예산에서 이 LSTM의 천장은 대략
macro-F1 0.92** 수준이며, 잔차 σ(±0.0013~0.0023)는 고정 분할 위 학습 과정(초기화·셔플·
드롭아웃) 노이즈일 뿐 분할 변동이 아니다. 세 축 중 `embed_dim`과 `hidden_size`는
"키울수록 좋아지다 plateau"라는 가설과 일치했고, `dropout`은 평평해 가설을 지지도 반박도
하지 못했다. (모든 수치는 RTX 3080 Ti, 5-seed 재실행 기준.)

### ③ 최종 모델 Class별 성능 분석

우승 설정(embed=256, **dropout=0.1**, **hidden=922**) 단일 시드(seed=42) 학습 모델의 test set
(클래스당 1,900개, 총 7,600개) 성능이다. 이 시드의 test macro-F1은 **0.9209**로 5-시드
평균(0.9189 ± 0.0017)과 일치 범위 안에 있다(best epoch=3).

| 클래스 | Precision | Recall | F1 | n |
|--------|----------:|-------:|------:|------:|
| World    | 93.62 | 92.74 | 93.18 | 1,900 |
| Sports   | 96.47 | 97.74 | **97.10** | 1,900 |
| Business | 90.79 | 86.63 | **88.66** | 1,900 |
| Sci/Tech | 87.63 | 91.32 | 89.43 | 1,900 |
| **macro** | **92.13** | **92.11** | **92.09** | 7,600 |

Confusion matrix (행=정답, 열=예측):

|            | →World | →Sports | →Business | →Sci/Tech |
|------------|-------:|--------:|----------:|----------:|
| **World**    | 1762 |   25 |   49 |   64 |
| **Sports**   |   19 | 1857 |   20 |    4 |
| **Business** |   58 |   19 | 1646 |  **177** |
| **Sci/Tech** |   43 |   24 |   **98** | 1735 |

**분석:**
- **Sports가 압도적으로 쉽다(F1 97.1).** 경기·선수·점수·리그명 등 스포츠 어휘가 다른
  주제와 거의 겹치지 않아 오분류가 1,900개 중 43개뿐이다.
- **World도 양호(F1 93.2).** 다만 국제 정치·경제 기사가 Business·Sci/Tech와 일부 겹쳐
  World→Sci/Tech 64건, →Business 49건의 오분류가 나온다.
- **Business와 Sci/Tech가 가장 어렵고 서로 가장 많이 혼동된다(F1 88.7 / 89.4).**
  Business→Sci/Tech 177건, Sci/Tech→Business 98건으로 **전체 오분류의 최대 축**이다.
  이유는 두 주제가 **어휘를 공유**하기 때문이다 — 기술기업(애플·구글 등)·제품·주가·실적·
  시장 같은 단어가 양쪽 기사에 모두 등장해, 마지막 hidden state 하나만으로는
  "기업의 사업 소식(Business)"과 "기술 그 자체(Sci/Tech)"를 가르기 어렵다.
  Business의 recall(86.6)이 가장 낮은 것도 상당수가 Sci/Tech로 새기 때문이다.
- 종합하면 LSTM의 천장(macro-F1 ≈ 0.92)을 누르는 주 병목은 **Business ↔ Sci/Tech 경계**이며,
  추가 개선(어텐션 풀링·키워드 가중 등)은 이 두 클래스에 집중하는 것이 가장 효율적이다.

**어떤 단어에서 혼동하나 (실제 오분류 분석):** 우승 모델의 test 오분류를 직접 뜯어보면, 두
클래스의 오류는 거의 전부 **테크 기업 기사**에서 난다. 같은 기업·제품이 양쪽 주제에 등장하므로,
기사가 *재무 표현*에 기대면 Business로, *제품·기술 용어*에 기대면 Sci/Tech로 읽힌다. (괄호 안 숫자는
해당 방향 177/98개 오분류 기사 중 그 단어가 등장한 기사 수.)

- **Business → Sci/Tech (177건)** — 끌고 간 단어: `company`(28) `internet`(18) `computer`(17)
  `software`(15) `online`(12) `web`(10) `chip`(10) `microsoft`(11) `technology`(11) — 즉 **제품·기술 어휘**.
  - *"Intel to delay … a video display **chip** …"* → Intel의 제품 출시 연기(실은 기업 소식)인데 `chip`·Intel에 끌려 Sci/Tech로 오분류.
  - *"Yahoo! Ups Ante for Small Businesses — **Web** hosting, domain-name price cuts"* → 가격 정책(Business)이지만 `Web`·Yahoo에 끌림.
  - *"Ohio Sues Best Buy … the **electronics** retailer …"* → 소송(Business)인데 `electronics`에 끌림.
- **Sci/Tech → Business (98건)** — 끌고 간 단어: `billion`(10) `quarter`(9) `sales`(7) `prices`(7)
  `shares`(6) `stock`(5) `percent`(6) `oracle`(7) `peoplesoft`(6) — 즉 **재무 어휘**.
  - *"Intuit Posts Wider **Loss** … maker of … software TurboTax … wider **quarterly** loss"* → 소프트웨어 업체(Sci/Tech)지만 `loss`·`quarterly`에 끌려 Business로.
  - *"Rivals Try to Turn Tables on Charles Schwab … low **prices** … discount **stock** broker"* → `prices`·`stock`에 끌림.
  - *Oracle·PeopleSoft* 인수전 기사들 → `billion`·`shares`에 끌림.

→ 즉 혼동의 축은 *개별 단어*가 아니라 **두 주제가 공유하는 어휘(기업명·제품·시장 용어)** 그 자체다.
마지막 hidden state 하나로 "같은 회사 기사가 어느 프레임인가"를 가르기 어렵다 — 어텐션/키워드
가중이 이 경계에서 가장 큰 이득을 줄 지점이다.

재현용 산출물(untracked, 재생성 가능): `code/outputs/lstm/stage{1,2,3}_*_s*/`
(`metrics.json`, `history.json`, `confusion_matrix.npy`, `best.pt`).
재실행: `bash run_stage1.sh && bash run_stage2.sh && bash run_stage3.sh`
(resumable — `metrics.json`이 이미 있으면 건너뜀). 집계:
`python summarize_multiseed.py --tag-prefix stage1_emb --sweep embed_dim --save-md ablation_embed_dim.md`
(이하 `stage2_drop`/`dropout`, `stage3_hid`/`hidden_size` 동일).

## Required ablation: Transformer side — 3-stage multi-seed sweep

The brief requires an embedding-dim ablation; we extend it to a 3-stage sequential ablation over `embed_dim`, `dropout`, and `num_layers`, with 5 seeds per cell (42, 43, 44, 45, 46) — 60 runs total. At each stage we hold the previous-stage winner fixed and pick the next winner by **mean validation macro-F1** across the 5 seeds. Per §3.3 the test set is held out: it is never used to choose between cells. Test numbers below are reported only as the final readout per cell.

Controlled across every cell: HuggingFace `ag_news`, 90/10 train/val split fixed at `data_seed=42` (split and DataLoader shuffle do not vary across seeds), torchtext `basic_english` tokenizer, vocab capped at 20,000 built from train only, `max_len=128` with dynamic batch padding, 2-layer Transformer Encoder with mean pooling over non-pad tokens, Adam lr=1e-3, batch=64, 8 epochs, grad_clip=1.0, CrossEntropyLoss on logits. The `--seed` argument controls model initialization and training-time stochasticity only (`model_seed`); the σ across seeds reflects model-init + dropout variance, not data-split variance.

---

### Stage 1 — embedding dimension (num_layers=2, dropout=0.3)

| embed_dim | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| 32 | 682,180 | 0.9170 ± 0.0014 | 0.9170 ± 0.0008 | 0.9168 ± 0.0009 | 245 |
| **64** | **1,380,228** | **0.9233 ± 0.0014** | **0.9228 ± 0.0019** | **0.9227 ± 0.0019** | **532** |
| 128 | 2,825,476 | 0.9206 ± 0.0020 | 0.9199 ± 0.0022 | 0.9199 ± 0.0023 | 306 |
| 256 | 5,912,580 | 0.8999 ± 0.0026 | 0.8982 ± 0.0012 | 0.8980 ± 0.0012 | 580 |

**Winner: embed_dim = 64** (val_f1 0.9233 ± 0.0014). Performance peaks at embed=64 and declines in both directions. embed=64 beats embed=32 by 0.006 val_f1 (5 of 5 seeds), confirming the step is resolved. The drop at embed=256 is large (−0.023 from embed=64), consistent with underfitting within the 8-epoch budget at that capacity. The 64→128 step is not resolved (3 of 5 seeds, p ≈ 0.5).

---

### Stage 2 — dropout (at embed=64, num_layers=2)

| dropout | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| **0.1** | **1,380,228** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** | **1560** |
| 0.3 | 1,380,228 | 0.9233 ± 0.0014 | 0.9228 ± 0.0019 | 0.9227 ± 0.0019 | 532 |
| 0.5 | 1,380,228 | 0.9126 ± 0.0011 | 0.9118 ± 0.0028 | 0.9115 ± 0.0028 | 1558 |
| 0.8 | 1,380,228 | 0.6850 ± 0.0996 | 0.7136 ± 0.0746 | 0.6806 ± 0.1020 | 1561 |

**Winner: dropout = 0.1** (val_f1 0.9253 ± 0.0017). dropout=0.5 and 0.8 are clearly worse; dropout=0.8 collapses entirely (σ=0.10, unstable training). The gap between dropout=0.1 and dropout=0.3 is 0.002 val_f1, which is within 1σ and not statistically resolved at n=5. We carry dropout=0.1 forward as the formal winner by the selection rule.

> **Note — single-seed vs multi-seed discrepancy:** With a single seed (42), dropout=0.3 produced a higher val_f1 and was treated as the winner. After running 5 seeds, dropout=0.1 has the higher mean val_f1 (0.9253 vs 0.9233). Additionally, test_f1 is higher for dropout=0.3 (0.9227 vs 0.9215), meaning the two are effectively equivalent on both val and test. This illustrates why single-seed selection is unreliable when the gap between cells is small.

---

### Stage 3 — number of layers (at embed=64, dropout=0.1)

| num_layers | params | val_f1 mean±std | test_acc mean±std | test_f1 mean±std | t (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,330,244 | 0.9204 ± 0.0016 | 0.9184 ± 0.0022 | 0.9183 ± 0.0021 | 1202 |
| **2** | **1,380,228** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** | **1560** |
| 3 | 1,430,212 | 0.9240 ± 0.0022 | 0.9224 ± 0.0024 | 0.9225 ± 0.0023 | 1376 |
| 4 | 1,480,196 | 0.9230 ± 0.0023 | 0.9214 ± 0.0028 | 0.9215 ± 0.0028 | 1408 |
| 5 | 1,530,180 | 0.9252 ± 0.0030 | 0.9217 ± 0.0032 | 0.9218 ± 0.0032 | 1452 |

**Winner: num_layers = 2** (val_f1 0.9253 ± 0.0017). The 1→2 step is clearly resolved (5 of 5 seeds). layers=5 is numerically tied with layers=2 (Δ=0.0001, well within 1σ), but layers=2 is preferred as it achieves the same performance with fewer parameters. layers=3 and 4 show a slight decline from layers=2, suggesting the 8-epoch budget is insufficient for deeper models to fully converge.

---

### Headline Transformer result

Final test-set readout, 5 seeds, n=7,600 test examples. Winner row is the config picked by the 3-stage selection rule (max mean val_f1 at each stage).

| config | val_f1 | test_acc | test_f1 |
|---|---:|---:|---:|
| **winner (embed=64, drop=0.1, layers=2)** | **0.9253 ± 0.0017** | **0.9216 ± 0.0009** | **0.9215 ± 0.0009** |
| shared baseline (embed=128, drop=0.3, layers=2) | 0.9206 ± 0.0020 | 0.9199 ± 0.0022 | 0.9199 ± 0.0023 |

The 3-stage sweep raised the Transformer's mean test F1 from 0.9168 ± 0.0009 (embed=32 starting cell) to **0.9215 ± 0.0009** at the winning config — a 0.47 pp absolute gain.

Consistent with the plan §5 hypothesis:
- `embed_dim`: monotone in mean val_f1 across 32→64→128 (not 256), full 32→64 step resolved (5 of 5 seeds). The 64→128 step is **not** resolved (3 of 5 wins, p ≈ 0.5).
- `dropout`: flat between 0.1 and 0.3 (< 1σ gap); 0.5 and 0.8 clearly worse. n=5 cannot statistically separate 0.1 from 0.3.
- `num_layers`: 2-layer plateau; adjacent-cell deltas (2 vs 3) are inside 1σ.

## License

Coursework, released open for inspection. Not for production use.
