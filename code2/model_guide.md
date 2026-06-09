# 모델 구현 가이드 (LSTM / Transformer Encoder)

데이터 전처리와 파이프라인(`data_pipeline.py`)을 쓰는 사람을 위한 가이드입니다.

`models.py`의 `LSTMClassifier`와 `TransformerEncoderClassifier`는 이미 구현이
끝나 있습니다. 파이프라인 인터페이스는 고정돼 있으니, 아래 규칙만 지키면 두 모델을
그대로 학습 루프에 붙일 수 있습니다. 모델을 더 손보고 싶다면(특히 Transformer)
`transformer/transformer_guide.md`를 보세요.

---

## 0. 환경 준비 (conda, Python 3.11)

Windows 기준입니다.

```powershell
# 최초 1회: Python 3.11 conda 환경 생성과 의존성 설치
powershell -ExecutionPolicy Bypass -File setup_env.ps1
# 또는: conda env create -f environment.yml

# 이후 매번
conda activate agnews-dl
```

환경 자체(`.venv`나 conda 폴더)는 공유하지 않습니다. 각자 `setup_env.ps1`을 돌려
같은 환경을 만듭니다. 파이프라인이 잘 도는지 확인하려면 `python data_pipeline.py`를
실행하세요.

---

## 1. 빠른 확인 (파이프라인 smoke test)

모델 구현과 별개로, 완성된 baseline인 `AverageEmbeddingClassifier`로
batch에서 model을 거쳐 loss까지 연결되는지 먼저 확인합니다.

```python
import torch
import torch.nn as nn
from data_pipeline import DataConfig, build_pipeline, set_seed, to_device
from models import AverageEmbeddingClassifier

set_seed(42)

cfg = DataConfig()                 # max_len=128, batch_size=64, vocab 20k, seed=42
bundle = build_pipeline(cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AverageEmbeddingClassifier(
    vocab_size=bundle.vocab_size,
    num_classes=bundle.num_classes,
    pad_idx=bundle.pad_idx,
    embed_dim=128,
).to(device)

batch = to_device(next(iter(bundle.train_loader)), device)
logits = model(batch["input_ids"], batch["lengths"])      # (B, 4)
loss = nn.CrossEntropyLoss()(logits, batch["labels"])

print(logits.shape)   # (B, 4)
print(loss.item())
```

이건 파이프라인이 제대로 도는지 보는 용도입니다. 실제 비교 실험은
`LSTMClassifier`와 `TransformerEncoderClassifier`로 합니다.

---

## 2. 배치 계약 (모든 DataLoader가 내놓는 dict)

| 키 | 타입과 shape | 설명 |
|----|--------------|------|
| `input_ids` | `LongTensor (B, L)` | `pad_idx`로 패딩. `L`은 배치 안 최대 길이(`max_len` 이하) |
| `lengths` | `LongTensor (B,)` | 행별 실제 토큰 수 (패딩 제외, 항상 1 이상) |
| `labels` | `LongTensor (B,)` | 클래스 인덱스 `0..3` |
| `texts` | `List[str]` (길이 B) | 원문 텍스트 (실패 분석용) |
| `indices` | `List[int]` (길이 B) | 원본 dataset index |
| `orig_lengths` | `LongTensor (B,)` | truncation 전 토큰 수 |
| `truncated` | `BoolTensor (B,)` | 잘렸으면 True |

추가로 알아둘 규칙:

- `input_ids`와 `labels`는 `torch.long`, `lengths.min()`은 1 이상입니다(파이프라인이 assert로 보장).
- 패딩 마스크는 모델 안에서 `pad_mask = (input_ids == bundle.pad_idx)`로 만듭니다(True가 패딩).
- LSTM은 `pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)`로 패딩을 건너뜁니다.
- Transformer는 `nn.TransformerEncoder(..., src_key_padding_mask=pad_mask)`로 패딩을 무시합니다.
- mean pooling을 할 때는 패딩 토큰을 평균에 넣지 않습니다(`AverageEmbeddingClassifier` 구현 참고).
- forward는 logits `(B, 4)`를 반환합니다. softmax는 넣지 않습니다(`CrossEntropyLoss`가 logits를 받기 때문). softmax는 confidence나 추론 분석에서만 씁니다.
- `texts`와 `indices`는 list라 GPU로 옮기지 않습니다. `to_device(batch, device)`가 텐서만 옮겨 줍니다.

---

## 3. `PipelineBundle` 필드

| 필드 | 용도 |
|------|------|
| `train_loader` / `val_loader` / `test_loader` | DataLoader. `test_loader`는 최종 평가 전용 |
| `train_dataset` / `val_dataset` / `test_dataset` | Dataset 객체 (클래스 분포나 실패 분석 시 직접 접근) |
| `vocab` | torchtext Vocab. `vocab.get_itos()`로 index에서 token으로 디코딩 |
| `tokenizer` | `basic_english` 토크나이저 |
| `pad_idx` / `unk_idx` | 0 / 1 |
| `vocab_size` | 실제 만들어진 vocab 크기 (20,000 이하). `nn.Embedding`에 사용 |
| `num_classes` | 4 |
| `class_names` | `['World','Sports','Business','Sci/Tech']` |
| `label_to_name` / `name_to_label` | 라벨과 이름 매핑 (혼동 행렬 라벨, 보고서용) |
| `config` | 사용된 `DataConfig` |

`DataConfig`의 주요 필드: `max_len=128`, `vocab_size=20000`, `batch_size=64`,
`seed=42`, `train_fraction=1.0`, `pad_to_max_len=False`(기본은 배치별 동적 패딩,
True면 항상 128로 패딩), `return_text=True`, `return_metadata=True`.

---

## 4. 기준(base) 하이퍼파라미터

공유 학습 설정은 계획서를 따릅니다. Adam, lr 1e-3, batch 64, 8 epochs,
dropout 0.3, seed 고정, 사전학습 모델 금지.

| | LSTM | Transformer Encoder |
|---|------|---------------------|
| Embedding | dim 128 | dim 128 + positional encoding |
| Core | 2-layer, 단방향, hidden 256 | 2 layers, 4 heads, FF 256 |
| Pooling | 마지막 hidden state | non-pad 토큰 mean pooling |
| Head/Loss | linear, cross-entropy | linear, cross-entropy |

---

## 5. 재현성 (같은 batch order)

`set_seed(seed)`와 DataLoader generator로 재현성을 맞춥니다. 두 모델이 완전히
같은 shuffle order를 보게 하려면, 각 모델을 학습하기 직전에 `set_seed(42)`를 다시
부르고 `build_pipeline(cfg)`로 DataLoader를 새로 만듭니다. 한 모델 학습에 이미 쓴
DataLoader는 generator state가 진행돼 버리기 때문입니다.

```python
set_seed(42)
bundle_lstm = build_pipeline(cfg)
# ... LSTM 학습 ...

set_seed(42)
bundle_tr = build_pipeline(cfg)
# ... Transformer 학습 ...
```

과제에서 통제해야 하는 핵심 조건은 같은 split, tokenizer, vocabulary, max_len,
metric, training budget입니다.

---

## 6. Ablation 분담

| 바꾸는 것 | 어디서 | 누가 |
|-----------|--------|------|
| Embedding dim (64/128/256), 필수 ablation | 모델의 `embed_dim` 인자 (두 모델 같게) | 모델 담당 |
| 시퀀스 길이 (128 vs 256) | `DataConfig(max_len=...)` | 데이터 파이프라인 |
| 데이터 크기 (25/50/100%), 학습 곡선 | `DataConfig(train_fraction=...)` | 데이터 파이프라인 |
| Transformer depth(1 vs 3), LSTM hidden(128/256/512), 선택 확장 | 모델 인자 | 모델 담당 |

Transformer 쪽을 바꾸는 구체적인 방법은 `transformer/transformer_guide.md`에 단계별로 적어 뒀습니다.

---

## 7. 공고문 11번 Pitfalls 중 모델과 평가 쪽 체크리스트

데이터 쪽(라벨 0~3 assert, vocab 누수 방지, 패딩 정보 제공, truncation flag, seed,
device 헬퍼, test 분리)은 파이프라인에서 이미 처리했습니다. 모델과 학습, 평가에서는
아래를 꼭 확인하세요.

- [ ] #6 train/eval 모드: 학습은 `model.train()`, 평가와 검증은 `model.eval()` + `torch.no_grad()`.
- [ ] #7 dropout: train과 eval에서 동작이 다릅니다. eval 모드를 빠뜨리면 검증 점수가 왜곡됩니다.
- [ ] #8 device: `model`, `input_ids`, `labels`가 같은 device에 있어야 합니다. 배치는 `to_device(batch, device)`로 옮깁니다.
- [ ] #9 metric: 정확도와 macro F1을 같이 봅니다. `from sklearn.metrics import f1_score; f1_score(y_true, y_pred, average='macro')`. 혼동 행렬은 `class_names`로 라벨링합니다.
- [ ] #10 test set: 모델 선택, 튜닝, early stopping은 `val_loader`로만. `test_loader`는 맨 마지막에 한 번만.
- [ ] #3 padding: LSTM은 pack, Transformer는 key_padding_mask와 마스킹된 mean.
- [ ] #5 seed: 모델을 만들고 학습하기 전에 `set_seed(cfg.seed)` (5번 항목 참고).
- [ ] Transformer는 positional encoding을 적용하고, 두 모델 모두 trainable 파라미터 수를 `count_parameters`로 보고합니다.
- [ ] forward 출력은 logits입니다 (softmax 아님).

---

## 8. 다음 단계 (요약 워크플로)

1. 모델은 이미 구현돼 있으니, `python -c "import models"`로 import만 확인합니다.
2. 공유 학습 루프: Adam, lr 1e-3, 8 epoch. epoch마다 train과 val loss를 기록합니다(손실 곡선).
3. 검증셋으로 best epoch를 고른 뒤, 마지막에 `test_loader`로 한 번 평가합니다.
4. accuracy, macro F1, 손실 곡선, 혼동 행렬, 파라미터 수 표를 만듭니다.
5. embedding dim ablation(64/128/256)을 두 모델 모두에 대해 돌립니다.
6. 실패 분석: 배치의 `texts`, `indices`, `truncated`로 오분류 예시를 뽑습니다(공유, LSTM만, Transformer만 각각 5건 이상).
7. 최종 제출은 `train_eval.ipynb` 중심으로 정리합니다.

`train_lstm.py`와 `transformer/train_transformer.py`에 위 1~4번을 그대로 돌리는 스크립트가 이미
들어 있으니, 따라 쓰거나 참고하면 됩니다.

---
