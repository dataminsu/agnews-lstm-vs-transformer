# Word Order Perturbation Ablation — 실행 안내 (팀원용)

데이터 파이프라인에 **단어 순서 교란** 옵션을 추가했습니다. 코드만 받아서 플래그만 바꿔
돌리면 됩니다. **전처리된 데이터 파일을 따로 받을 필요가 없습니다** — 셔플은 코드가
결정론적으로 생성하므로 모두가 byte-identical 입력으로 학습합니다.

## 핵심 보장
- 셔플 키 = `data_seed(42) + split별 salt + 샘플 index`. **`model_seed`와 무관**하므로
  한 조건의 5개 seed(42–46)는 **동일한 교란 데이터**를 보고 init/shuffle/dropout만 달라집니다
  (= 교란은 통제 변수, augmentation 아님).
- 교란은 truncate 이후 / padding 이전에 한 번만 적용. `<pad>`는 교란하지 않음.
- train/val/test에 **동일 조건** 적용. 어휘(vocab)는 재구축하지 않음(토큰 빈도 보존).

## 옵션
| 플래그 | 값 | 설명 |
|---|---|---|
| `--order-condition` | `original` / `local_shuffle` / `full_shuffle` | 순서 조건 |
| `--perturb-window` | 정수(기본 5) | `local_shuffle`의 윈도 크기 |

- `original`: 원문 순서 유지(기준)
- `local_shuffle`: window(=5) 안에서만 셔플 → 국소 어순만 손상
- `full_shuffle`: 문서 전체 토큰 무작위 순열 → 순서 정보 거의 제거

## 실행 매트릭스 (3조건 × 5 seed × 2모델 = 30 run)
고정 config: baseline(LSTM embed128/hidden256/2L, Transformer embed128/2L/4h), dropout 0.3,
8 epoch, batch 64, lr 1e-3.

### LSTM (이 폴더에서)
```powershell
conda activate agnews-dl
foreach ($s in 42,43,44,45,46) {
  python -u train_lstm.py --order-condition original      --model-seed $s --tag wo_orig_s$s
  python -u train_lstm.py --order-condition local_shuffle --perturb-window 5 --model-seed $s --tag wo_local_s$s
  python -u train_lstm.py --order-condition full_shuffle  --model-seed $s --tag wo_full_s$s
}
```

### Transformer (`cd transformer` 후)
```powershell
foreach ($s in 42,43,44,45,46) {
  python -u train_transformer.py --order-condition original      --seed $s --tag wo_orig_s$s --save-plots
  python -u train_transformer.py --order-condition local_shuffle --perturb-window 5 --seed $s --tag wo_local_s$s
  python -u train_transformer.py --order-condition full_shuffle  --seed $s --tag wo_full_s$s
}
```
> Transformer는 기존 방식대로 `--seed`가 모델 init/dropout 변동을 담당합니다(data_seed는 42 고정).

## 집계
```powershell
# LSTM: 조건별 mean±std
python summarize_multiseed.py --tag-prefix wo_ --sweep order_condition --save-md ablation_word_order.md
# Transformer
cd transformer
python summarize_multiseed_transformer.py --tag-prefix wo_ --sweep order_condition --save-md ablation_word_order_transformer.md
```

## 추가 지표(보고서용)
- ΔF1 = F1(perturbed) − F1(original)
- Order Sensitivity = (F1_orig − F1_pert) / F1_orig  (클수록 순서에 민감)
- class-wise F1: 각 run의 `confusion_matrix.npy`로 계산(Business↔Sci/Tech 민감도 확인)

## 빠른 점검(전체 돌리기 전에)
조건당 1 seed만 1 epoch로 돌려 `metrics.json`의 `config.order_condition`이 기록되고 정상
학습하는지 확인하세요. 기대: Full Shuffle test F1 ≤ Original(순서 정보 제거). 거의 안 떨어지면
"AG News는 주제 키워드 중심" 해석과 일치합니다.
