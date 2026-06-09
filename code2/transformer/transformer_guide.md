# Transformer 이어서 작업하기 (혜주님 가이드)

이 문서는 Transformer Encoder 모델을 이어서 바꾸려는 혜주님을 위한 안내입니다.
"어디 파일의 어느 부분을, 어떻게 바꾸면 되는지"를 가능한 한 자세히 적었습니다.
딥러닝을 처음 다뤄도 따라올 수 있게 썼으니, 위에서부터 순서대로 읽으면 됩니다.

먼저 한 가지만 기억하세요. 대부분의 실험은 **코드를 한 줄도 고치지 않고**
명령어 옵션만 바꿔서 할 수 있습니다(3번 항목). 코드를 직접 손대야 하는 경우는
따로 5번에 모아 뒀습니다.

---

## 0. 큰 그림

우리 과제는 같은 데이터, 같은 학습 설정에서 LSTM과 Transformer를 공정하게
비교하는 것입니다. 데이터 처리는 `data_pipeline.py`가 다 해 주고, 두 모델은
거기서 나온 배치를 똑같이 받아 씁니다. 그래서 혜주님이 신경 쓸 곳은 모델과
학습 스크립트, 이렇게 두 군데뿐입니다.

---

## 1. 어떤 파일을 보면 되나요

Transformer 관련 파일은 이 폴더(`code/transformer/`)에 모여 있습니다. 단,
모델 구조와 데이터 처리는 LSTM과 함께 쓰는 공유 파일이라 상위 폴더 `code/`에 있습니다.

| 파일 | 위치 | 무슨 일을 하나 | 혜주님이 건드릴 일 |
|------|------|----------------|--------------------|
| `train_transformer.py` | `code/transformer/` (여기) | Transformer를 학습하고, 평가하고, 결과를 저장 | 새 옵션을 추가할 때 (5-C) |
| `summarize_ablation_transformer.py` | `code/transformer/` (여기) | 여러 실험 결과를 표 하나로 묶기 | 거의 안 건드림 |
| `models.py` | `code/` (상위 폴더) | 모델 구조 정의. 그중 `TransformerEncoderClassifier`가 우리 Transformer | 구조 자체를 바꿀 때만 (5번) |
| `data_pipeline.py` | `code/` (상위 폴더) | 데이터 로딩, 토큰화, 배치 만들기 | 거의 안 건드림. `max_len`만 예외 |

상위 폴더 `code/models.py` 안에서 Transformer와 관련된 부분은 두 클래스입니다.

- `SinusoidalPositionalEncoding` (대략 92번째 줄부터): 위치 정보를 더해 주는 부분.
- `TransformerEncoderClassifier` (대략 120번째 줄부터): 우리 Transformer 본체.

---

## 2. 지금 Transformer가 어떻게 생겼나요

`TransformerEncoderClassifier.forward`가 입력을 받아 예측까지 가는 흐름은 이렇습니다.
입력은 `input_ids`(토큰 번호들)와 `lengths`(문장별 실제 길이)입니다.

1. **패딩 마스크 만들기**: `pad_mask = (input_ids == self.pad_idx)`.
   짧은 문장을 길이 맞추려고 채운 `<pad>` 자리를 True로 표시합니다.
   이걸 attention에 넘겨서 빈 자리를 무시하게 합니다.
2. **embedding**: `self.embedding(input_ids)`로 토큰 번호를 벡터로 바꿉니다.
   벡터 크기가 `embed_dim`입니다.
3. **positional encoding**: `self.pos_encoding(emb)`로 "이 토큰이 문장에서 몇 번째인지"
   정보를 더합니다. Transformer는 순서를 자동으로 모르기 때문에 이 단계가 필요합니다.
4. **encoder**: `self.encoder(emb, src_key_padding_mask=pad_mask)`.
   여기가 self-attention이 일어나는 핵심부입니다. `num_layers`만큼 층을 쌓습니다.
5. **pooling**: 토큰마다 나온 벡터를 문장 하나당 벡터 하나로 줄입니다.
   기본값 `"mean"`은 `<pad>`를 뺀 실제 토큰들의 평균을 냅니다.
   `"cls"`로 바꾸면 맨 앞에 붙인 `[CLS]` 토큰 하나만 씁니다.
6. **분류기**: `self.fc(self.dropout(pooled))`로 클래스 4개에 대한 점수(logits)를 냅니다.
   여기서 softmax는 안 합니다. 손실 함수(`CrossEntropyLoss`)가 logits를 그대로 받기 때문입니다.

`__init__`의 인자가 이 구조를 결정합니다.

```python
def __init__(self, vocab_size, num_classes, pad_idx, embed_dim=128, nhead=4,
             num_layers=2, dim_feedforward=256, dropout=0.3, max_len=128,
             pooling="mean"):
```

- `embed_dim`: 토큰 벡터 크기. 클수록 표현력이 늘지만 파라미터도 늘어납니다.
- `nhead`: attention head 개수. `embed_dim`이 `nhead`로 나누어떨어져야 합니다.
- `num_layers`: encoder 층 수(깊이).
- `dim_feedforward`: 각 층 안 feedforward의 폭.
- `dropout`: 과적합을 줄이는 비율.
- `max_len`: 다룰 수 있는 최대 문장 길이. 데이터의 `max_len`과 같게 둡니다.
- `pooling`: `"mean"` 또는 `"cls"`.

---

## 3. 코드 수정 없이 옵션만 바꾸기 (제일 많이 쓰는 방법)

`train_transformer.py`는 위 인자들을 전부 명령줄 옵션으로 받습니다. 즉 터미널에서
숫자만 바꿔 주면 됩니다. 실행 위치는 이 폴더입니다(`cd code/transformer`).

| 옵션 | 바꾸는 것 | 기본값 | 주의 |
|------|-----------|--------|------|
| `--embed-dim` | 토큰 벡터 크기 | 128 | `--nhead`로 나누어떨어져야 함 |
| `--nhead` | attention head 수 | 4 | `embed_dim % nhead == 0` |
| `--num-layers` | encoder 깊이 | 2 | 깊을수록 느리고 과적합 위험 |
| `--dim-feedforward` | feedforward 폭 | 256 | |
| `--dropout` | dropout 비율 | 0.3 | 0~1 사이 |
| `--pooling` | pooling 방식 | mean | `mean` 또는 `cls`만 |
| `--lr` | 학습률 | 1e-3 | |
| `--epochs` | 학습 epoch 수 | 8 | |
| `--batch-size` | 배치 크기 | 64 | |
| `--max-len` | 최대 문장 길이 | 128 | 데이터 쪽도 같이 바뀜 |
| `--tag` | 결과 저장 폴더 이름 | baseline | 실험마다 다르게 주기 |

`--tag`가 중요합니다. 결과가 이 폴더 안 `outputs/<tag>/`에 저장되는데, tag를 안 바꾸면
이전 결과를 덮어씁니다. 실험마다 알아보기 쉬운 이름을 주세요.

### 필수 ablation: embedding dim 64 / 128 / 256

과제에서 요구하는 실험입니다. 세 번 돌리면 됩니다.

```powershell
python -u train_transformer.py --embed-dim 64  --tag embed64
python -u train_transformer.py --embed-dim 128 --tag baseline
python -u train_transformer.py --embed-dim 256 --tag embed256
```

### 선택 확장: 깊이 바꿔 보기

```powershell
python -u train_transformer.py --num-layers 1 --tag layers1
python -u train_transformer.py --num-layers 3 --tag layers3
```

다른 옵션도 같은 방식입니다. 한 번에 하나씩만 바꾸고 나머지는 기본값으로 두는 것이
좋습니다. 그래야 결과 차이가 그 옵션 때문인지 분명해집니다.

---

## 4. 결과 확인하기

한 번 학습이 끝나면 이 폴더 안 `outputs/<tag>/`에 아래가 저장됩니다.

| 파일 | 내용 |
|------|------|
| `metrics.json` | 최종 test 정확도, macro-F1, 파라미터 수, 설정값 |
| `history.json` | epoch마다의 train/val loss와 점수 |
| `confusion_matrix.npy` / `.png` | 혼동 행렬 (숫자와 그림) |
| `loss_curves.png` | train과 val 손실 곡선 |
| `failures.json` | 가장 자신 있게 틀린 오분류 예시 상위 20개 |
| `best.pt` | 검증 점수가 가장 좋았던 시점의 모델 가중치 |

여러 실험을 표 하나로 묶고 싶으면 같은 폴더의 `summarize_ablation_transformer.py`를
실행합니다. 기본값으로 이 폴더의 `outputs/`를 읽으니 옵션 없이도 됩니다.

```powershell
python summarize_ablation_transformer.py --save-md ablation_embed_dim_transformer.md
```

---

## 5. 코드를 직접 고쳐야 하는 경우

옵션으로 안 되는 변경은 상위 폴더의 `code/models.py`를 손봐야 합니다. 아래 세 가지가
대표적입니다. 고치기 전에 원본을 복사해 두거나 git으로 현재 상태를 커밋해 두면 안전합니다.

### (A) pooling 방식을 새로 추가하기

지금은 `"mean"`과 `"cls"`만 있습니다. 예를 들어 "마지막 토큰만 쓰기"를 추가한다고 합시다.

1. `TransformerEncoderClassifier.__init__` 맨 위의 검사 줄을 찾습니다.

   ```python
   assert pooling in ("mean", "cls"), f"pooling must be 'mean' or 'cls', got {pooling!r}"
   ```

   여기에 새 이름을 넣습니다.

   ```python
   assert pooling in ("mean", "cls", "last"), f"pooling must be 'mean', 'cls', or 'last', got {pooling!r}"
   ```

2. `forward`의 pooling 부분을 찾습니다.

   ```python
   if self.pooling == "mean":
       real_mask = (~pad_mask).unsqueeze(-1).float()
       pooled = (out * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)
   else:
       pooled = out[:, 0]
   ```

   여기에 `"last"` 분기를 추가합니다. `lengths`로 각 문장의 마지막 실제 토큰 위치를 찾습니다.

   ```python
   if self.pooling == "mean":
       real_mask = (~pad_mask).unsqueeze(-1).float()
       pooled = (out * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)
   elif self.pooling == "last":
       last_idx = (lengths - 1).clamp(min=0)
       pooled = out[torch.arange(out.size(0)), last_idx]
   else:  # "cls"
       pooled = out[:, 0]
   ```

3. 마지막으로 `train_transformer.py`의 `--pooling` 옵션에서 선택지를 늘립니다.

   ```python
   ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "last"])
   ```

이제 `python -u train_transformer.py --pooling last --tag pool_last`로 실험할 수 있습니다.

### (B) positional encoding을 learned 방식으로 바꾸기

지금은 고정된 sinusoidal 방식(`SinusoidalPositionalEncoding`)을 씁니다. 위치 벡터를
학습으로 배우게 바꾸려면, `__init__`에서 positional encoding을 만드는 줄을 바꿉니다.

원래:

```python
self.pos_encoding = SinusoidalPositionalEncoding(
    embed_dim, max_len + (1 if self.use_cls else 0), dropout
)
```

learned 버전으로 바꾸려면, 먼저 파일 위쪽에 작은 클래스를 하나 추가합니다.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, d_model)
        positions = torch.arange(x.size(1), device=x.device)
        return self.dropout(x + self.pos(positions).unsqueeze(0))
```

그리고 `__init__`의 위 줄을 이렇게 바꿉니다.

```python
self.pos_encoding = LearnedPositionalEncoding(
    embed_dim, max_len + (1 if self.use_cls else 0), dropout
)
```

`forward`는 그대로 둬도 됩니다. 입출력 모양이 같기 때문입니다. 두 방식을 비교하려면
원본을 한 번 돌려 두고(`--tag pos_sinusoidal`), 바꾼 뒤 다시 돌리세요(`--tag pos_learned`).

### (C) 새 하이퍼파라미터를 옵션으로 추가하기

모델에 이미 있는 인자(`nhead` 등)는 옵션이 다 연결돼 있습니다. 새 인자를 모델에
추가했다면, `train_transformer.py`에서 세 군데를 이어 줘야 옵션으로 쓸 수 있습니다.

1. `main()`의 `argparse` 부분에 옵션 추가:

   ```python
   ap.add_argument("--my-new-arg", type=int, default=10)
   ```

2. 모델을 만드는 `TransformerEncoderClassifier(...)` 호출에 인자 전달:

   ```python
   model = TransformerEncoderClassifier(
       ...,
       my_new_arg=args.my_new_arg,
   )
   ```

3. (선택) 어떤 값으로 돌렸는지 출력하는 `print(f"model : ...")` 줄에 같이 적어 두면
   로그만 봐도 설정을 알 수 있습니다.

`args`는 `metrics.json`의 `"config"`에 통째로 저장되므로, 옵션만 제대로 연결하면
설정 기록은 자동으로 남습니다.

---

## 6. 바꾸기 전에 꼭 지켜야 하는 규칙

- **`embed_dim`은 `nhead`로 나누어떨어져야 합니다.** 안 그러면 모델을 만들 때
  바로 에러가 납니다. 예: `embed_dim=64`면 `nhead`는 1, 2, 4, 8 중에서.
- **forward는 logits를 반환합니다. softmax를 넣지 마세요.** 손실이 이상해집니다.
  softmax는 추론할 때 confidence를 볼 때만 씁니다(`failures.json` 만드는 부분처럼).
- **`<pad>`는 항상 무시해야 합니다.** attention에는 `src_key_padding_mask=pad_mask`,
  mean pooling에는 마스크를 곱해서 평균을 냅니다. 이미 그렇게 돼 있으니, 새 코드를
  넣을 때도 이 점을 깨지 않게 조심하세요.
- **test set은 마지막에 한 번만 봅니다.** 모델을 고르거나 옵션을 정할 때는 검증셋(val)만
  봐야 합니다. 이건 스크립트가 이미 그렇게 해 두었으니, 직접 평가 코드를 새로 쓰지 않는 한
  신경 안 써도 됩니다.

---

## 7. 자주 하는 실수 체크리스트

- [ ] `--tag`를 안 바꿔서 이전 결과를 덮어쓰지 않았나요?
- [ ] `embed_dim`이 `nhead`로 나누어떨어지나요?
- [ ] 한 번에 옵션을 하나만 바꿨나요? (여러 개 바꾸면 원인 파악이 어려움)
- [ ] 코드를 고친 뒤 `python -u train_transformer.py`가 에러 없이 시작하는지 봤나요? (import 오류가 있으면 바로 멈춥니다)
- [ ] 같은 비교를 할 때 seed를 그대로 42로 뒀나요? (스크립트가 기본으로 고정)
- [ ] CPU에서는 한 번 학습에 몇 분 걸립니다. GPU가 있으면 자동으로 GPU를 씁니다.

---

## 8. 전체 ablation을 한 번에 돌리기

PowerShell에서 아래를 그대로 붙여 넣으면 embedding dim 세 개를 차례로 돌리고
표까지 만들어 줍니다.

```powershell
conda activate agnews-dl
cd code/transformer
python -u train_transformer.py --embed-dim 64  --tag embed64
python -u train_transformer.py --embed-dim 128 --tag baseline
python -u train_transformer.py --embed-dim 256 --tag embed256
python summarize_ablation_transformer.py --save-md ablation_embed_dim_transformer.md
```

다 돌고 나면 `ablation_embed_dim_transformer.md`에 세 실험이 한 표로 정리됩니다. 이 표를
보고서의 Transformer ablation 절에 그대로 넣으면 됩니다.

---

## 9. 막히면

- 모델 인터페이스(배치가 어떻게 생겼는지, 어떤 규칙을 지켜야 하는지)는 상위 폴더의 `code/model_guide.md`에 있습니다.
- 전체 실험 흐름과 결과는 저장소 루트의 `README.md`에 정리돼 있습니다.
- 에러 메시지를 그대로 읽어 보세요. `embed_dim must be divisible by nhead`처럼
  뭐가 문제인지 알려 주는 경우가 많습니다.
