# Does Word Order Matter? Comparing LSTM and Transformer Encoder Models on AG News


---

## 1. Introduction

본 프로젝트는 동일한 텍스트 분류 과제(AG News, 4-class 뉴스 주제 분류)에서 **LSTM 분류기**와
**Transformer Encoder 분류기**를 처음부터(from scratch) 학습시켜 비교한다. 두 모델 모두 사전학습
언어모델을 사용하지 않는다.

본 보고서의 목적은 단순히 높은 정확도를 얻는 것이 아니라, **순환적 시퀀스 모델링(LSTM)과
self-attention 기반 모델링(Transformer)이 분류 성능·학습 거동·수렴 양상·실패 사례·실험 설정
민감도에서 어떻게 다른지**를 분석하는 것이다.

**Main research question.** 동일한 데이터·전처리·어휘·학습 예산에서 두 아키텍처는 (a) 어느 쪽이 더
높은 macro-F1을 얻는가, (b) 같은 성능을 얻기 위해 얼마나 많은 파라미터를 필요로 하는가, 그리고
(c) AG News 분류가 실제로 *단어 순서*에 의존하는가 아니면 *주제 키워드*만으로 풀리는가를 묻는다.
(c)는 본 보고서의 핵심 ablation(§4.2, Word Order Perturbation)으로 검증한다.

---

## 2. Dataset

| 항목 | 값 |
|---|---|
| 데이터 소스 / 접근 방법 | HuggingFace `ag_news` **단일 소스** (TorchText·CSV와 혼용하지 않음) |
| 클래스 수 / 라벨 매핑 | 4개: 0=World, 1=Sports, 2=Business, 3=Sci/Tech (0-indexed, CrossEntropyLoss 호환) |
| Split | 공식 train을 90/10으로 train/val 분할 (`seed=42` 고정), 공식 test는 최종 평가 전용 |
| 샘플 수 | train 108,000 / val 12,000 / test 7,600 |
| 클래스 분포 | 4개 클래스 균등 (train 각 27,000, test 각 1,900) — **[TODO: 실제 카운트 그림/표 1개 첨부]** |
| 토크나이저 | `torchtext` `basic_english` |
| 어휘 구축 | **train split에서만** 구축, 상위 20,000개로 제한, `<pad>=0`, `<unk>=1` (누수 없음) |
| 패딩/절단 | 동적 배치 패딩, 최대 길이 128로 절단 |
| 최대 시퀀스 길이 | 128 |

두 모델은 **완전히 동일한 처리 결과(같은 split·어휘·max_len)**를 입력으로 사용하여 공정 비교를
보장한다. 어휘는 train에서만 만들어 validation/test 누수를 차단했다.

---

## 3. Models

두 모델은 §6 권장 시작 하이퍼파라미터를 기반으로 하며, 공정 비교를 위해 아래 조건을 **모든 실험에서
고정**한다: Adam(lr 1e-3), batch 64, 8 epochs, grad_clip 1.0, CrossEntropyLoss(logits),
dropout 0.3, 모델 선택은 **검증 macro-F1 최댓값** 기준(test set은 마지막에 한 번만 사용).

### 3.1 LSTM Classifier (baseline)
- embedding → 2-layer 단방향 LSTM → 마지막 hidden state pooling → linear → 4 logits
- embed_dim 128, hidden 256, 2 layers, dropout 0.3
- **trainable params: 3,482,628**

### 3.2 Transformer Encoder Classifier (comparison)
- embedding + sinusoidal positional encoding → 2-layer Encoder(4 heads, FF 256) →
  비-pad 토큰 mean pooling → linear → 4 logits
- embed_dim 128, 2 layers, 4 heads, FF 256, dropout 0.3
- **trainable params: 2,825,476**

> **공정성 비고.** 메인 비교(§5.1)는 위 두 baseline(3.48M vs 2.83M, 파라미터 규모가 비슷함)으로
> 수행한다. §5.3의 용량 ablation에서 도출한 *튜닝된* 구성(LSTM 16.3M, Transformer 1.38M)은
> 파라미터 규모가 크게 달라 메인 비교의 공정성 기준을 깨므로, 메인 표가 아니라 **효율성 논의**에서만
> 사용한다.

---

## 4. Experiments

### 4.1 Training setup & evaluation
- 평가 지표: accuracy, **macro-F1**(클래스 단위 성능 반영), train/val loss curve, confusion matrix.
- 재현성: `data_seed=42`로 split 고정, `model_seed ∈ {42..46}` 5개로 학습 과정 잡음(init/shuffle/
  dropout)을 정량화. 따라서 표의 σ는 **split 변동이 아니라 학습 과정 변동**이다.

### 4.2 Required ablation — Word Order Perturbation (핵심)
- **가설.** AG News는 주제 키워드 중심 과제이므로 단어 순서를 제거해도 성능 저하는 제한적일 것이다.
  추가로, 순차 처리에 의존하는 LSTM이 self-attention 기반 Transformer보다 순서 교란에 더 민감할 것이다.
- **변경 변수(한 가지).** 입력 토큰 순서 조건: `Original` / `Local Shuffle`(window 5) / `Full Shuffle`.
- **통제 변수.** 어휘(셔플은 단어 빈도를 바꾸지 않으므로 어휘 재구축 금지·공유), split, 토크나이저,
  max_len, 모델 하이퍼파라미터, optimizer, lr, batch, epoch, 지표 — 전부 고정. 셔플은
  deterministic(`seed + sample_idx`)으로 적용하고 `<pad>`는 셔플하지 않으며 순서는
  tokenize→truncate→shuffle→pad. train/val/test에 **동일 조건**을 적용한다.
- **정량화.** `Delta F1 = F1_perturbed − F1_original`, `Order Sensitivity = (F1_orig − F1_pert)/F1_orig`.
- **[TODO] 이 실험은 아직 미실행.** §5.2 표·해석은 실행 후 채운다.

### 4.3 추가 ablation — 용량/정규화 민감도 (extension)
필수는 아니지만, 두 아키텍처의 **용량 효율**을 보기 위해 각 모델에서 한 축씩 순차 sweep을 5-seed로
수행했다(각 모델 60 run). LSTM: embed_dim → dropout → hidden_size. Transformer: embed_dim →
dropout → num_layers. 각 단계는 직전 단계 승자를 고정하고 **검증 macro-F1 평균**으로 다음 승자를 선택
(test set은 선택에 불사용).

---

## 5. Results

### 5.1 Main comparison (baseline, 파라미터 규모 정합)

| Model | params | val macro-F1 | test acc | test macro-F1 |
|---|---:|---:|---:|---:|
| LSTM (embed128, hidden256, 2L) | 3,482,628 | 0.9174 ± 0.0012 | 0.9138 ± 0.0006 | 0.9137 ± 0.0006 |
| Transformer (embed128, 2L, 4h) | 2,825,476 | 0.9206 ± 0.0020 | 0.9199 ± 0.0022 | 0.9199 ± 0.0023 |

**해석.** 비슷한 파라미터 규모(≈3M)에서 Transformer가 test macro-F1 기준 약 **+0.6pp**
(0.9199 vs 0.9137) 더 높다. 두 모델 모두 안정적으로 수렴하며 0.91–0.92 macro-F1대에 도달한다.
AG News에서 절대 성능 차이는 크지 않다.

### 5.2 Word Order Perturbation (핵심 ablation)

**가설.** (i) AG News는 topic-keyword 중심 과제라 단어 순서를 제거해도 성능 저하는 제한적일 것이다.
(ii) 순차 처리에 의존하는 LSTM이 self-attention 기반 Transformer보다 순서 교란에 더 민감할 것이다.
**변경 변수:** 입력 토큰 순서(Original / Local Shuffle w=5 / Full Shuffle), 한 가지만. **통제:** 어휘·split·
토크나이저·max_len·하이퍼파라미터·optimizer·epoch 전부 고정. 셔플은 `data_seed`에 묶인 결정론적
변환이라 5개 seed가 동일한 교란 데이터를 본다(= augmentation 아님). train/val/test에 동일 적용.

test set 7,600개, 5 model-seed 평균:

| Order Condition | Model | val_f1 | test_acc | test_f1 | ΔF1 | Order Sens. |
|---|---|---|---|---|---|---|
| Original | LSTM | 0.9181 ± 0.0013 | 0.9154 ± 0.0031 | 0.9153 ± 0.0030 | — | — |
| Local Shuffle | LSTM | 0.9129 ± 0.0022 | 0.9079 ± 0.0022 | 0.9079 ± 0.0023 | −0.0075 | +0.82% |
| Full Shuffle | LSTM | 0.9077 ± 0.0010 | 0.9051 ± 0.0044 | 0.9050 ± 0.0043 | **−0.0103** | **+1.13%** |
| Original | Transformer | 0.9214 ± 0.0003 | 0.9225 ± 0.0014 | 0.9224 ± 0.0014 | — | — |
| Local Shuffle | Transformer | 0.9212 ± 0.0011 | 0.9216 ± 0.0013 | 0.9214 ± 0.0014 | −0.0009 | +0.10% |
| Full Shuffle | Transformer | 0.9204 ± 0.0010 | 0.9189 ± 0.0010 | 0.9188 ± 0.0010 | **−0.0036** | **+0.39%** |

**해석.**
1. **가설 (i) 지지 — AG News는 순서보다 주제 키워드로 풀린다.** 단어 순서를 *완전히* 파괴해도 LSTM은
   −1.0pp, Transformer는 −0.4pp만 떨어진다. 순서가 본질이었다면 붕괴했어야 하므로, 분류 신호의
   대부분은 `stock, NASA, team, oil` 같은 topic lexical cue가 담당한다.
2. **가설 (ii) 지지 — LSTM이 Transformer보다 순서에 ~3배 민감.** Full Shuffle ΔF1: LSTM −0.0103
   vs Transformer −0.0036. 순환 모델은 hidden state를 순차 갱신해 순서에 의존하고, self-attention은
   토큰을 순서 없는 집합처럼 집계해 더 강건하다. **용량 sweep(§5.3)은 두 모델이 똑같이 0.92로 수렴해
   아키텍처 차이를 못 드러냈지만, 이 ablation은 그 차이를 직접 보여준다.**
3. **단조 반응**: 두 모델 모두 Original > Local > Full로 깔끔히 감소(통제된 dose-response).
4. **유의성(n=5)**: LSTM의 하락은 둘 다 유의(σ≈0.003 대비 ~2.5–3σ). Transformer는 Full만 유의,
   **Local(−0.0009)은 노이즈 범위** — LSTM은 부분 교란에도 반응하나 Transformer는 완전 교란에만 미세
   반응. 또한 **셔플된 Transformer(0.9188)가 원문 LSTM(0.9153)보다도 높다.**

> *This ablation is a controlled diagnostic, not a claim that real-world news appears in shuffled order;
> it estimates how much each model relies on word order versus lexical topic cues.*

**Class-wise (test F1, 5-seed 평균) 및 Business↔Sci/Tech 혼동.**

| Model · Condition | World | Sports | Business | Sci/Tech | B→S% | S→B% |
|---|---|---|---|---|---|---|
| LSTM Original | 0.921 | 0.968 | 0.882 | 0.890 | 8.4 | 6.4 |
| LSTM Full Shuffle | 0.909 | 0.961 | 0.870 | 0.881 | 9.4 | 6.8 |
| Transformer Original | 0.927 | 0.971 | 0.891 | 0.900 | 8.2 | 5.3 |
| Transformer Full Shuffle | 0.923 | 0.970 | 0.884 | 0.898 | 8.9 | 5.2 |

Sports가 가장 쉽고(~0.97, 셔플에도 거의 불변), **Business·Sci/Tech가 가장 어렵다**(~0.87–0.90). 최대
오류원인 **Business→Sci/Tech 혼동이 셔플 시 증가**(LSTM 8.4→9.4%, TR 8.2→8.9%) → 단어 순서가 이
의미 중첩 쌍의 구분에 일부 기여했음을 보여준다. 클래스별 하락폭도 LSTM이 Transformer보다 크다.

> *환경 주의:* 본 ablation은 단일 환경(RTX 5090 / torch 2.7)에서 3조건을 self-contained로 돌렸으므로
> ΔF1·Order Sensitivity는 내부적으로 유효하다. Original의 절대값(예: LSTM 0.9153)이 §5.1 메인비교
> baseline(0.9137, 다른 GPU/torch)과 ~0.1–0.2pp 차이 나는 것은 하드웨어/수치 차이이며, 토크나이저·
> 어휘는 byte 단위로 동일함을 검증했다(§3.2 참조).

### 5.3 용량/정규화 ablation 요약 (extension)

LSTM (5-seed 평균):

| 축 | 최적값 | 비고 |
|---|---|---|
| embed_dim (32→256) | 256 | 평균 단조 증가, 32→256 차이 5/5 seed 유의(sign test p≈0.031), 인접 셀은 1σ 이내 |
| dropout (0.1–0.8) | 0.1 | 축이 평평함 — 어떤 쌍도 n=5에서 분리 안 됨 |
| hidden_size (256→922) | 922 | 256→922는 유의(5/5), **512에서 사실상 plateau** (512 vs 922는 3/5, ΔF1<1σ) |

→ LSTM 튜닝 ceiling ≈ **test F1 0.9177** (winner hidden=922, 16.3M params; cost-aware 대안
hidden=512는 8.8M로 동일 test F1 0.9177, 학습 2.1배 빠름).

Transformer (5-seed 평균):

| 축 | 최적값 | 비고 |
|---|---|---|
| embed_dim (32→256) | **64** | 64에서 정점 후 양방향 하락, embed=256은 −0.023로 8-epoch 예산 내 underfitting |
| dropout (0.1–0.8) | 0.1 | 0.1≈0.3(1σ 이내), 0.5·0.8 명확히 악화, **0.8은 학습 붕괴**(σ=0.10) |
| num_layers (1–5) | 2 | 1→2 유의(5/5), 2층 이후 plateau, 깊은 모델은 8-epoch 내 미수렴 추정 |

→ Transformer 튜닝 ceiling ≈ **test F1 0.9215** (winner embed=64/2L, **단 1.38M params**).

**효율성 핵심 발견.** Transformer는 **약 1.38M 파라미터로 LSTM의 16.3M보다 높은 test F1
(0.9215 vs 0.9177)**에 도달한다. 즉 self-attention 모델이 이 과제에서 **약 12배 적은 파라미터로 더
나은 표현 효율**을 보인다. 또한 dropout 0.8에서의 Transformer 학습 붕괴는 self-attention 모델이
강한 정규화에 더 취약할 수 있음을 보여주는 학습 안정성 사례다.

### 5.4 Loss curves & confusion matrices — **[TODO]**
- 두 baseline의 train/val loss 곡선 1쌍씩 (수렴 속도·과적합 여부).
- 두 baseline의 confusion matrix 1개씩 (`confusion_matrix.npy` + `class_names`로 그림).
- 수렴/안정성 서술: 어느 모델이 더 빨리 수렴했는지, 과적합 징후, dropout 0.8 붕괴 사례.

---

## 6. Failure Analysis

두 baseline(seed 42)을 **동일 test 7,600개**에 추론하고 예측을 같은 example index로 join했다
(스크립트: `failure_analysis.py`). 합치 결과:

| 범주 | 개수 | 비율 |
|---|---:|---:|
| 둘 다 정답 | 6,782 | 89.2% |
| 둘 다 오답 (Type E) | 380 | 5.0% |
| LSTM만 오답 (Type C) | 239 | 3.1% |
| Transformer만 오답 (Type D) | 199 | 2.6% |

두 모델이 불일치할 때 Transformer가 더 자주 정답(239 vs 199)이며, 이는 §5.1의 Transformer 우위와
일관된다. (검증: 위 카운트로 계산한 정확도 LSTM 0.9185 / Transformer 0.9238가 저장된 metrics와
소수점 4자리까지 일치 → 라벨 매핑·집계가 정확함.)

### 6.1 대표 오분류 (입력·정답·예측·confidence·원인·실패 양상)

| Text (발췌) | True | LSTM (conf) | Transf. (conf) | Possible reason | 실패 양상 |
|---|---|---|---|---|---|
| Nepal blockade 'blow to tourism' … rebel blockade | Business | World (1.00) | World (1.00) | 지정학 어휘(blockade, rebel)가 경제(관광 수입) 신호를 압도; World↔Business 중첩 | **동일** (Type E) |
| Mars water tops science honours … salty water on Mars | World | Sci/Tech (1.00) | Sci/Tech (1.00) | 명백한 과학 내용 → gold 라벨이 노이즈로 추정; 두 모델 모두 의미상 Sci/Tech | **동일** (Type E, 라벨 노이즈) |
| Med school move delayed … College of **Human Medicine** … cost | Business | Sci/Tech (1.00) | Business (0.53) | LSTM이 'Medicine' 한 단어에 고착→Sci/Tech; TR은 'move/relocated/cost' 통합→Business | **다름** (Type C, LSTM 키워드 과의존) |
| Great White Shark Loses Monitor Tag … **data-gathering device** | Sci/Tech | World (0.99) | Sci/Tech (0.52) | LSTM이 'shark/waters/Cape Cod'(지리)→World; TR은 'data-gathering device/tag'(과학 단서) 주목 | **다름** (Type C, LSTM 표면 키워드) |
| Meditation Practice Helping Arthritis Patients | Sci/Tech | Sci/Tech (0.99) | Sports (1.00) | Transformer의 드문 진짜 오류('practice'의 스포츠 연상 추정); LSTM 정답 | **다름** (Type D, TR 오류) |
| Maddux Wins No. 302 … Cubs … NL wild-card | World | World (0.76) | Sports (1.00) | 야구=Sports가 의미상 맞으나 gold=World; TR이 의미대로 분류 | **다름** (Type D, 라벨 노이즈) |

### 6.2 유형별 패턴
- **둘 다 오답(Type E)**: 두 모델이 *확신을 갖고 같이* 틀리는 경우로, **라벨 노이즈와 본질적 모호성**
  (World↔Business, Sci/Tech↔World 경계)에 집중. 오류가 아키텍처가 아니라 데이터에서 기인함을 시사.
- **LSTM만 오답(Type C)**: **단일 salient 키워드 과의존**이 전형(medicine→Sci/Tech, shark→World).
  Transformer의 전역 attention이 더 넓은 문맥을 통합해 교정하나, 종종 낮은 confidence(0.52–0.79)로.
- **Transformer만 오답(Type D)**: 상당수가 **라벨 노이즈**(스포츠 기사가 World로 라벨된 경우 TR이 의미상
  맞게 Sports라 했으나 gold와 불일치). 따라서 Transformer의 *실제* 오류율은 199보다 낮다.

### 6.3 Type A — 단어 순서가 정말 필요했던 사례 (Original 정답 → Full Shuffle 오답)
순서 정보가 없을 때 뒤집히는 예시 수: **LSTM 286건 vs Transformer 160건(≈1.8배)** — §5.2의 결론
(LSTM이 순서에 더 의존)을 개별 사례로 뒷받침한다. 예:

| Model | Text (발췌) | True | Shuffle 후 예측 (conf) | 원인 |
|---|---|---|---|---|
| LSTM | Sports in brief … his second **drive** … sixth **tee** | Sports | Sci/Tech (1.00) | 'drive/contact/tee'가 순서 안에서만 골프로 읽힘; 셔플되면 기술 용어처럼 보임 |
| LSTM | Flying Cars Reportedly Still Decades Away | Sci/Tech | World (0.99) | 구(phrase) 단위 의미('flying cars … decades away')가 셔플로 소실 |
| Transformer | Strikes at London airports … refuellers | Business | World (0.97) | 노동/경제 맥락이 순서 소실로 '공항+파업'의 지정학처럼 읽힘 |

**원인 카테고리 요약:** 클래스 중첩(특히 Business↔Sci/Tech, World↔Business), 단일 키워드 과의존(LSTM),
라벨 노이즈, phrase-level 문맥 손실(셔플 시). 절단(truncation)은 본 사례들에서 주요 원인이 아니었다
(대부분 trunc=N).

---

## 7. Conclusion

- 비슷한 파라미터 규모에서 Transformer Encoder가 LSTM보다 약간 높은 macro-F1을 보였고(§5.1),
  파라미터 효율은 훨씬 우수했다(§5.3, ~12배 적은 파라미터로 더 높은 ceiling).
- 두 모델 모두 0.92 macro-F1 부근에서 plateau하며, AG News에서 절대 성능 격차는 작다.
- **Word Order ablation(핵심)**: 단어 순서를 완전히 제거해도 성능 저하가 작아(LSTM −1.0pp, TR −0.4pp)
  **AG News는 주로 주제 키워드로 풀리는 과제**임을 진단했다. 동시에 **LSTM이 Transformer보다 순서에
  ~3배 민감**(ΔF1 −0.0103 vs −0.0036; Type A 286 vs 160건)해, 순환 모델이 self-attention보다 토큰
  순서에 더 의존한다는 가설을 지지했다. 이는 용량 sweep이 못 드러낸 두 아키텍처의 실질적 차이다.
- **Failure analysis**: 불일치 시 Transformer가 더 자주 정답(239 vs 199). LSTM 단독 오류는 단일 키워드
  과의존이, 공통 오류는 라벨 노이즈·클래스 중첩이 주원인이었다.
- Lessons learned: 단일 seed 선택의 위험(Transformer dropout에서 single-seed와 multi-seed 승자가
  뒤집힘), test set 격리, 공정 비교를 위한 파라미터 규모 정합.
- Future work: 더 긴 epoch 예산에서 깊은 Transformer 재평가, 데이터 효율(25/50/100%) 곡선.

---

## 8. AI Usage Appendix (≤1쪽) — **[TODO]**
1) 사용 도구 2) 사용 용도(코드 초안·디버깅·시각화·문장 다듬기) 3) AI 산출물 중 부정확/불완전했던 것
4) 팀이 수정한 부분 5) AI가 아닌 팀이 내린 결정(실험 설계·해석·승자 선택) 6) AI 사용이 프로젝트에 준 영향.
