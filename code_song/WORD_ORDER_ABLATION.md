# Word Order Perturbation Ablation — 실행 안내 (팀원용)

데이터 파이프라인에 **단어 순서 교란** 옵션을 추가했습니다. 코드만 받아서 플래그만 바꿔
돌리면 됩니다. **전처리된 데이터 파일을 따로 받을 필요가 없습니다** — 셔플은 코드가
결정론적으로 생성하므로 모두가 byte-identical 입력으로 학습합니다.

## 0. 환경 준비 (RTX 5090 / Linux) — torchtext 제거됨
RTX 5090(Blackwell, sm_120)은 torch≥2.7 / CUDA 12.8이 필요한데 torchtext 0.18은 torch 2.3
전용이라 설치 불가입니다. 그래서 **torchtext를 코드에서 들어내고**(`text_utils.py`로 대체,
`basic_english` 토크나이저·어휘가 torchtext와 **byte-identical**임을 `verify_text_utils.py`로
증명) torch 버전을 자유롭게 했습니다. 기존 4080 결과와 토큰화·어휘 인덱스가 동일합니다.
```bash
conda create -n agnews-dl python=3.11 -y && conda activate agnews-dl
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 5090 = cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
> `pip install torchtext`는 다시 하지 마세요(설치하면 또 ABI 에러). 코드는 torchtext 없이 돕니다.
> (참고: `verify_text_utils.py`는 torchtext가 깔린 머신에서만 도는 *증명용* 스크립트라 5090에선 안 돌아도 정상.)

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

## 실행 매트릭스 (3조건 × 5 seed × 2모델 = 30 run) — 모델별 15개씩 두 배치
고정 config는 **스크립트 기본값**이라 따로 줄 필요 없음(LSTM embed128/hidden256/2L,
Transformer embed128/2L/4h, dropout 0.3, 8 epoch, batch 64, lr 1e-3).

리눅스에서 (code 루트에서 실행):
```bash
conda activate agnews-dl
bash run_word_order_lstm.sh          # LSTM 15 run (3조건 × 5seed)
bash run_word_order_transformer.sh   # Transformer 15 run (스크립트가 transformer/로 cd)
```
- 두 스크립트 모두 **resumable**: `metrics.json`이 이미 있는 run은 건너뜀(중간에 끊겨도 다시 실행).
- 각 run 로그는 `outputs/<model>/_logs/<tag>.log`.
- GPU가 2장이면 동시에:
  ```bash
  CUDA_VISIBLE_DEVICES=0 bash run_word_order_lstm.sh &
  CUDA_VISIBLE_DEVICES=1 bash run_word_order_transformer.sh &
  wait
  ```
- env가 PATH에 없으면: `PYENV=/경로/envs/agnews-dl/bin/python bash run_word_order_lstm.sh`
> Transformer는 기존 방식대로 `--seed`가 모델 init/dropout 변동 담당(data_seed는 42 고정).

## 집계 — 두 모델 한 표로 (ΔF1 · Order Sensitivity 포함)
```bash
python summarize_word_order.py --save-md ablation_word_order.md
```
- `outputs/lstm/wo_*`와 `transformer/outputs/transformer/wo_*`를 모두 읽어 **조건별 mean±std +
  ΔF1 + Order Sensitivity**를 한 표로 출력하고 `ablation_word_order.md`로 저장.
- 부분 데이터에도 동작(한쪽만 끝나도 출력). 이 표를 보고서 §5.2에 그대로 붙이면 됩니다.
- ΔF1 = F1(perturbed) − F1(original), Order Sensitivity = (F1_orig − F1_pert)/F1_orig.
- (참고) LSTM 단독 표는 `run_word_order_lstm.sh`가 끝에 `ablation_word_order_lstm.md`로도 저장.
- class-wise F1(Business↔Sci/Tech 민감도)은 각 run의 `confusion_matrix.npy`로 별도 계산.

## 빠른 점검(전체 돌리기 전에)
조건당 1 seed만 1 epoch로 돌려 정상 학습·기록을 확인:
```bash
python -u train_lstm.py --order-condition full_shuffle --epochs 1 --train-fraction 0.05 \
  --model-seed 42 --data-seed 42 --tag wo_smoke
```
`outputs/lstm/wo_smoke/metrics.json`의 `config.order_condition`이 찍히면 OK(점검 후 폴더 삭제).
기대: Full Shuffle test F1 ≤ Original(순서 정보 제거). 거의 안 떨어지면 "AG News는 주제 키워드
중심" 해석과 일치합니다.
