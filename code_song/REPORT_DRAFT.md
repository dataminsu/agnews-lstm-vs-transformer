# Does Word Order Matter? Comparing LSTM and Transformer Encoder Models on AG News

> 보고서 초안 (v0). 교수님 공지 §19 "Final Report Structure"의 8개 절 구조를 그대로 따름.
> 굵게 표시한 `[TODO]`는 아직 산출물이 없어서 채워야 하는 부분입니다.
> 분량 목표: 본문 4–6쪽 (AI Usage Appendix 별도 1쪽).

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

### 5.2 Word Order Perturbation (핵심 ablation) — **[TODO: 실행 후]**

| Order Condition | Model | val acc | val F1 | test acc | test F1 | ΔF1 from Original |
|---|---|---|---|---|---|---|
| Original | LSTM | | | | | — |
| Original | Transformer | | | | | — |
| Local Shuffle | LSTM | | | | | |
| Local Shuffle | Transformer | | | | | |
| Full Shuffle | LSTM | | | | | |
| Full Shuffle | Transformer | | | | | |

해석(채울 항목): (1) Full Shuffle에서도 성능 저하가 작다면 → AG News는 주로 lexical topic cue로
풀린다는 진단. (2) 두 모델의 ΔF1 비교로 어느 아키텍처가 순서 정보에 더 의존하는지 결론. (3) class-wise
F1로 Business↔Sci/Tech 같은 의미 중첩 클래스가 셔플에 더 민감한지 확인. 명시 문장: *"This ablation
is a controlled diagnostic, not a claim that real news appears shuffled."*

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

## 6. Failure Analysis — **[TODO: 미실행, 필수 항목]**

두 baseline을 **동일 test 예시**에 적용하여 다음 5유형에서 각각 5건 이상(총 ≥5, 가능하면 유형별로)
오분류를 수집·분석한다. 각 예시는 입력 텍스트, 정답, LSTM 예측, Transformer 예측, confidence,
추정 원인, 두 모델의 실패가 유사한지/다른지를 기록한다.

- Type A: Original에서 맞고 Full Shuffle에서 틀린 사례 (순서 의존 진단)
- Type C: LSTM만 틀린 사례 / Type D: Transformer만 틀린 사례
- Type E: 두 모델 모두 같은 라벨로 틀린 사례

논의할 원인: 모호한 표현, 문맥 부족, 클래스 중첩(특히 Business↔Sci/Tech), 절단, 토큰화, 과적합/
미적합. (산출 경로: 각 run의 `failures.json`은 자기 모델 오분류 상위 20건만 있으므로, **두 모델
예측을 같은 test 인덱스로 join하는 셀을 notebook §7에 추가해야 함.**)

---

## 7. Conclusion

- 비슷한 파라미터 규모에서 Transformer Encoder가 LSTM보다 약간 높은 macro-F1을 보였고(§5.1),
  파라미터 효율은 훨씬 우수했다(§5.3, ~12배 적은 파라미터로 더 높은 ceiling).
- 두 모델 모두 0.92 macro-F1 부근에서 plateau하며, AG News에서 절대 성능 격차는 작다.
- **[핵심, TODO]** Word Order ablation 결과로 "AG News가 단어 순서보다 주제 키워드에 의존하는
  정도"와 "어느 아키텍처가 순서에 더 민감한지"를 결론.
- Lessons learned: 단일 seed 선택의 위험(Transformer dropout에서 single-seed와 multi-seed 승자가
  뒤집힘), test set 격리, 공정 비교를 위한 파라미터 규모 정합.
- Future work: 더 긴 epoch 예산에서 깊은 Transformer 재평가, 데이터 효율(25/50/100%) 곡선.

---

## 8. AI Usage Appendix (≤1쪽) — **[TODO]**
1) 사용 도구 2) 사용 용도(코드 초안·디버깅·시각화·문장 다듬기) 3) AI 산출물 중 부정확/불완전했던 것
4) 팀이 수정한 부분 5) AI가 아닌 팀이 내린 결정(실험 설계·해석·승자 선택) 6) AI 사용이 프로젝트에 준 영향.
