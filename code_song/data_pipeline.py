"""AG News data preprocessing & pipeline.

Single source of truth for tokenization, vocabulary, and DataLoaders shared by
BOTH the LSTM and the Transformer Encoder models. Owning this one file guarantees
the "fair comparison" requirements: identical split, tokenizer, vocabulary, max
length, and processed inputs for both models.

Dataset source rule (instructor brief 3.2):
    The dataset is HuggingFace `ag_news` ONLY. TorchText is used *only* for its
    `basic_english` tokenizer and its vocabulary utility, not as a dataset
    loader. We never mix the HuggingFace, TorchText, and raw-CSV access routes.

Honors the team plan (assignment1_plan.docx) and the brief (2026_term_project.pdf):
  - data_seed=42 fixes the 90/10 train/val split (a CONTROLLED variable, held
    constant across replicate runs); model_seed varies across replicates and
    drives init/shuffle/dropout for the mean +/- std. The official test set is
    reserved for FINAL evaluation only.
  - vocabulary built from TRAIN data only, size <= 20,000, <pad>=0 / <unk>=1.
  - labels are integer indices 0..3 (asserted) for CrossEntropyLoss.

Truncation strategy:
    A tokenized sequence longer than `max_len` (default 128) is truncated by
    dropping trailing tokens. So the maximum sequence length is always controlled.

Padding strategy:
    Default is batch-wise dynamic padding, so each batch is padded to its own
    longest sequence, never beyond `max_len`. Set DataConfig.pad_to_max_len=True
    to instead pad every batch to a fixed length of `max_len`. Dynamic padding is
    the default to save compute. Both models consume the SAME processed inputs.

Ablation knobs that touch DATA live here (the model teammate does NOT change these):
  - DataConfig.max_len        -> sequence-length sweep (128 vs 256)
  - DataConfig.train_fraction -> learning-curve data sizes (0.25 / 0.5 / 1.0), stratified

Batch contract (what every DataLoader yields), see model_guide.md:
    {
      "input_ids":    LongTensor (batch, seq_len),   # padded with pad_idx
      "lengths":      LongTensor (batch,),            # true token count per row (>0, <= max_len)
      "labels":       LongTensor (batch,),            # class index in 0..3
      "texts":        List[str]   (len batch),        # raw text, for failure analysis
      "indices":      List[int]   (len batch),        # original dataset index
      "orig_lengths": LongTensor (batch,),            # token count BEFORE truncation
      "truncated":    BoolTensor (batch,),            # True if the row was truncated
    }
Build the padding mask in the model with `input_ids == bundle.pad_idx`.

Quick start (pipeline smoke test): see model_guide.md / train_eval.ipynb.
"""

from __future__ import annotations

import os
# cuBLAS determinism for reproducible GPU matmuls. Must be set before the first
# CUDA call (i.e. before torch initialises its cuBLAS handle), so we set it at
# import time. Harmless on CPU-only machines.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# torchtext-free: text_utils provides a byte-identical basic_english tokenizer +
# build_vocab_from_iterator (proven by verify_text_utils.py). This lets the code
# run on Blackwell GPUs (RTX 5090, torch>=2.7) where torchtext 0.18 cannot install.
from text_utils import get_tokenizer, build_vocab_from_iterator


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """All knobs for the data pipeline. Defaults match the team plan."""

    dataset_name: str = "ag_news"
    # Two INDEPENDENT seeds (see set_seed / build_pipeline / train_lstm.py):
    #   data_seed  -> the 90/10 train/val split AND the stratified train subsample.
    #                 Defines WHICH examples are train vs val. CONTROLLED variable:
    #                 keep FIXED across replicate runs so every seed trains and
    #                 validates on the EXACT SAME data.
    #   model_seed -> weight init, DataLoader shuffle order, and dropout masks.
    #                 The TRAINING-process randomness; vary it across the replicate
    #                 seeds (42..46) so the reported mean +/- std measures robustness
    #                 to init/shuffle/dropout on a fixed split.
    # The official test set (ds["test"]) is fixed regardless of either seed.
    data_seed: int = 42
    model_seed: int = 42
    val_ratio: float = 0.1          # 90/10 train/val split from official train
    vocab_size: int = 20_000        # total vocab incl. specials 
    min_freq: int = 1               # min token frequency to enter the vocab
    batch_size: int = 64
    num_workers: int = 0            # keep 0 on Windows for reproducible, hassle-free runs

    # Ablation knobs that change the DATA (owned by the pipeline, not the model):
    max_len: int = 128              # truncation length (ablation: 128 vs 256)
    train_fraction: float = 1.0     # learning-curve data size (stratified subset)

    # Word-order perturbation ablation (owned by the pipeline so every teammate
    # sees byte-identical perturbed inputs). The shuffle is keyed on data_seed
    # (NOT model_seed) so all replicate seeds in one condition see the SAME
    # perturbed data -- the perturbation is a CONTROLLED variable, not data
    # augmentation. Applied to train/val/test identically, after truncation and
    # before padding. Vocabulary is unchanged (shuffling preserves token freq).
    order_condition: str = "original"  # "original" | "local_shuffle" | "full_shuffle"
    perturb_window: int = 5            # window size for local_shuffle


    pad_token: str = "<pad>"        # forced to index 0
    unk_token: str = "<unk>"        # forced to index 1
    pad_idx: int = 0
    unk_idx: int = 1

    # False: batch-wise dynamic padding capped by max_len (default, cheaper)
    # True : always pad every batch to a fixed length of max_len
    pad_to_max_len: bool = False

    # Request torch deterministic algorithms (best-effort; see set_seed). cuDNN's
    # LSTM kernel has no fully deterministic GPU variant in every torch version,
    # so this is warn-only and never aborts a run.
    deterministic: bool = True

    # Keep raw text / per-sample metadata in batches (needed for failure analysis)
    return_text: bool = True
    return_metadata: bool = True


# --------------------------------------------------------------------------- #
# Reproducibility  
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed the RNGs that drive the TRAINING process: weight init, DataLoader
    shuffle, and dropout masks. Call again right before model construction so init
    does not depend on how many RNG draws the pipeline consumed (see train_lstm.py).

    The train/val split is NOT seeded here: it takes an explicit data_seed via
    train_test_split / _stratified_subset, so the split stays fixed while this
    (model) seed varies across replicate runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # cuDNN's LSTM / pack_padded_sequence backward has no fully deterministic
        # GPU kernel in every torch version; warn_only=True requests determinism
        # without aborting the run when a kernel lacks a deterministic variant.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """Worker seeding for DataLoader determinism when num_workers > 0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Vocabulary  
# --------------------------------------------------------------------------- #
def build_vocab(train_texts, tokenizer, cfg: DataConfig):
    """Build the Vocab from the TRAINING texts only.

    <pad> is forced to index 0 and <unk> to index 1 (special_first=True). The
    default index is <unk>, so any out-of-vocabulary token maps to <unk>.
    """

    def yield_tokens(texts):
        for text in texts:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(train_texts),
        min_freq=cfg.min_freq,
        specials=[cfg.pad_token, cfg.unk_token],
        special_first=True,
        max_tokens=cfg.vocab_size,
    )
    vocab.set_default_index(vocab[cfg.unk_token])
    return vocab


def encode(text, tokenizer, vocab, max_len: int, unk_idx: int):
    """Encode one raw string. Returns (ids, orig_len, truncated).

    Returned values:
      ids        list of integer token ids fed to the model
      orig_len   token count BEFORE truncation
      truncated  True if the sequence was cut because it exceeded max_len

    A non-empty sequence is guaranteed (empty input becomes [unk_idx]) so
    pack_padded_sequence never sees a length-0 row.
    """
    tokens = tokenizer(text)
    orig_len = len(tokens)
    ids = vocab(tokens[:max_len])
    if len(ids) == 0:
        ids = [unk_idx]
    truncated = orig_len > max_len
    return ids, orig_len, truncated


def _encode_texts(texts, tokenizer, vocab, max_len, unk_idx):
    encoded, orig_lens, truncated = [], [], []
    for text in texts:
        ids, ol, tr = encode(text, tokenizer, vocab, max_len, unk_idx)
        encoded.append(ids)
        orig_lens.append(ol)
        truncated.append(tr)
    return encoded, orig_lens, truncated


# --------------------------------------------------------------------------- #
# Word-order perturbation (ablation). Operates on the already-truncated, NOT-YET
# padded id list: no <pad> is present here (collate adds it) and there are no
# cls/bos/eos special tokens in this pipeline, so every position is a real token
# that may be reordered. Returns a NEW list (never mutates the input).
# --------------------------------------------------------------------------- #
def _perturb_ids(ids, mode, window, rng):
    """Reorder token ids for the word-order ablation.

    mode:
      "original"      -> unchanged
      "full_shuffle"  -> random permutation of all tokens (order info ~removed)
      "local_shuffle" -> shuffle only within consecutive windows of `window`
                         tokens (local phrase order broken, global order kept)

    `rng` is a deterministic random.Random seeded from (data_seed + split_salt +
    sample_idx) by the caller, so the perturbation is reproducible everywhere and
    independent of model_seed. token frequency (the vocab) is preserved.
    """
    if mode == "original" or len(ids) <= 1:
        return list(ids)
    if mode == "full_shuffle":
        out = list(ids)
        rng.shuffle(out)
        return out
    if mode == "local_shuffle":
        out = []
        for i in range(0, len(ids), window):
            w = ids[i:i + window]
            rng.shuffle(w)
            out.extend(w)
        return out
    raise ValueError(f"unknown order_condition: {mode!r}")


# --------------------------------------------------------------------------- #
# Dataset & batching  
# --------------------------------------------------------------------------- #
class AGNewsDataset(Dataset):
    """Holds encoded sequences, labels, and the metadata needed for failure analysis."""

    def __init__(self, encoded, labels, texts, indices, orig_lens, truncated):
        self.encoded = encoded
        self.labels = labels
        self.texts = texts
        self.indices = indices
        self.orig_lens = orig_lens
        self.truncated = truncated

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoded[idx],
            "label": int(self.labels[idx]),
            "text": self.texts[idx],
            "idx": int(self.indices[idx]),
            "orig_len": int(self.orig_lens[idx]),
            "truncated": bool(self.truncated[idx]),
        }


def collate_batch(batch, pad_idx, max_len, pad_to_max_len, return_text, return_metadata):
    """Collate samples into the batch contract dict.

    `lengths` lets the LSTM pack and the Transformer build a key-padding mask, so
    padding never dominates pooling/attention.
    """
    lengths = [len(s["input_ids"]) for s in batch]
    target = max_len if pad_to_max_len else max(lengths)

    input_ids = torch.full((len(batch), target), pad_idx, dtype=torch.long)
    for i, s in enumerate(batch):
        ids = s["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    assert lengths_t.min().item() > 0, "zero-length sequence in batch (encode() guard failed)"
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)

    out = {"input_ids": input_ids, "lengths": lengths_t, "labels": labels}
    if return_text:
        out["texts"] = [s["text"] for s in batch]
    if return_metadata:
        out["indices"] = [s["idx"] for s in batch]
        out["orig_lengths"] = torch.tensor([s["orig_len"] for s in batch], dtype=torch.long)
        out["truncated"] = torch.tensor([s["truncated"] for s in batch], dtype=torch.bool)
    return out


def to_device(batch, device):
    """Move tensor values in a batch dict to `device`. Non-tensor fields such as
    raw `texts` (List[str]) and `indices` (List[int]) are kept unchanged
    (pitfall #8: device mismatch)."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _stratified_subset(labels, fraction: float, seed: int):
    """Class-balanced subset of indices for the learning-curve sizes."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(labels):
        idx_c = np.where(labels == cls)[0]
        rng.shuffle(idx_c)
        k = int(round(len(idx_c) * fraction))
        keep.append(idx_c[:k])
    keep = np.concatenate(keep)
    keep.sort()
    return keep.tolist()


def load_splits(cfg: DataConfig):
    """Load ag_news, carve the val split with cfg.data_seed, optionally subsample train.

    The split uses cfg.data_seed (NOT the model seed), so the train/val partition
    is a controlled variable: it stays fixed while model_seed varies across the
    replicate runs. Returns three (texts, labels, original_indices) tuples plus the
    label names. original_indices are positions in the post-split arrays (stable
    keys for failure-analysis cross-referencing).
    """
    ds = load_dataset(cfg.dataset_name)
    label_names = list(ds["train"].features["label"].names)  # ['World','Sports','Business','Sci/Tech']

    split = ds["train"].train_test_split(test_size=cfg.val_ratio, seed=cfg.data_seed)
    train, val, test = split["train"], split["test"], ds["test"]

    train_texts, train_labels = list(train["text"]), list(train["label"])
    val_texts, val_labels = list(val["text"]), list(val["label"])
    test_texts, test_labels = list(test["text"]), list(test["label"])

    # pitfall #1: CrossEntropyLoss needs labels 0..num_classes-1. ag_news is
    # already 0-indexed; assert it so a source swap can't slip past us.
    valid = set(range(len(label_names)))
    assert set(train_labels) <= valid, f"unexpected labels: {set(train_labels) - valid}"

    train_idx = list(range(len(train_labels)))
    if cfg.train_fraction < 1.0:
        sel = _stratified_subset(train_labels, cfg.train_fraction, cfg.data_seed)
        train_texts = [train_texts[i] for i in sel]
        train_labels = [train_labels[i] for i in sel]
        train_idx = sel

    val_idx = list(range(len(val_labels)))
    test_idx = list(range(len(test_labels)))

    return (
        (train_texts, train_labels, train_idx),
        (val_texts, val_labels, val_idx),
        (test_texts, test_labels, test_idx),
        label_names,
    )


# --------------------------------------------------------------------------- #
# Pipeline bundle & entry point
# --------------------------------------------------------------------------- #
@dataclass
class PipelineBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    vocab: object
    tokenizer: Callable
    pad_idx: int
    unk_idx: int
    vocab_size: int          # actual built size (<= cfg.vocab_size)

    num_classes: int
    class_names: List[str]
    label_to_name: Dict[int, str]
    name_to_label: Dict[str, int]

    config: DataConfig


def _make_dataset(part, tokenizer, vocab, cfg, split_salt):
    texts, labels, indices = part
    encoded, orig_lens, truncated = _encode_texts(texts, tokenizer, vocab, cfg.max_len, cfg.unk_idx)
    if cfg.order_condition != "original":
        # Deterministic per-sample shuffle keyed on data_seed (so all model_seeds
        # in a condition see identical perturbed data) + a per-split salt (so the
        # same index in different splits gets a different permutation).
        encoded = [
            _perturb_ids(
                ids, cfg.order_condition, cfg.perturb_window,
                random.Random(cfg.data_seed + split_salt + int(idx)),
            )
            for ids, idx in zip(encoded, indices)
        ]
    return AGNewsDataset(encoded, labels, texts, indices, orig_lens, truncated)


def build_pipeline(cfg: DataConfig | None = None) -> PipelineBundle:
    """Build everything: tokenizer, vocab (train only), datasets, DataLoaders.

    This is the only function the model teammate needs to call.
    """
    cfg = cfg or DataConfig()
    # Seed the TRAINING-process RNG (the shuffle generator below + any global-RNG
    # consumers). The split is seeded separately via cfg.data_seed inside
    # load_splits, so it is unaffected by this call.
    set_seed(cfg.model_seed, deterministic=cfg.deterministic)

    train_part, val_part, test_part, label_names = load_splits(cfg)
    num_classes = len(label_names)

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_part[0], tokenizer, cfg)  # TRAIN ONLY -> no leakage
    pad_idx = vocab[cfg.pad_token]
    unk_idx = vocab[cfg.unk_token]

    # Distinct per-split salts so an identical sample index in different splits
    # does not receive the same permutation (harmless, but cleaner).
    train_ds = _make_dataset(train_part, tokenizer, vocab, cfg, split_salt=0)
    val_ds = _make_dataset(val_part, tokenizer, vocab, cfg, split_salt=10_000_000)
    test_ds = _make_dataset(test_part, tokenizer, vocab, cfg, split_salt=20_000_000)

    collate = partial(
        collate_batch,
        pad_idx=pad_idx,
        max_len=cfg.max_len,
        pad_to_max_len=cfg.pad_to_max_len,
        return_text=cfg.return_text,
        return_metadata=cfg.return_metadata,
    )
    generator = torch.Generator()
    generator.manual_seed(cfg.model_seed)  # DataLoader shuffle = training randomness

    common = dict(
        batch_size=cfg.batch_size,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    label_to_name = {i: name for i, name in enumerate(label_names)}
    name_to_label = {name: i for i, name in enumerate(label_names)}

    return PipelineBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        vocab=vocab,
        tokenizer=tokenizer,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
        vocab_size=len(vocab),
        num_classes=num_classes,
        class_names=label_names,
        label_to_name=label_to_name,
        name_to_label=name_to_label,
        config=cfg,
    )


# --------------------------------------------------------------------------- #
# Verification / dataset documentation.
# describe_pipeline() prints everything section 3.4 of the brief asks for; the
# __main__ block also runs an AverageEmbeddingClassifier smoke test.
# --------------------------------------------------------------------------- #
def _class_distribution(labels, num_classes):
    arr = np.asarray(labels)
    return {c: int((arr == c).sum()) for c in range(num_classes)}


def _length_stats(orig_lens, max_len):
    arr = np.asarray(orig_lens)
    return {
        "mean": round(float(arr.mean()), 2),
        "p95": int(np.percentile(arr, 95)),
        "max": int(arr.max()),
        "trunc_rate": round(float((arr > max_len).mean()), 4),  # pitfall #4
    }


def describe_pipeline(bundle: PipelineBundle) -> None:
    cfg = bundle.config
    line = "=" * 70
    pad_strategy = (
        f"fixed pad to max_len={cfg.max_len}" if cfg.pad_to_max_len
        else f"batch-wise dynamic padding (capped at max_len={cfg.max_len})"
    )

    print(line)
    print("AG News pipeline summary")
    print(line)
    print(f"dataset source    : HuggingFace '{cfg.dataset_name}' (single source)")
    print(f"tokenizer         : basic_english (pure-python, torchtext-identical)")
    print(f"data_seed (split) : {cfg.data_seed} (FIXED)  |  model_seed (init/shuffle/dropout): {cfg.model_seed}")
    print(f"determinism       : {cfg.deterministic}  |  val_ratio: {cfg.val_ratio}  |  train_fraction: {cfg.train_fraction}")
    print(f"max_len           : {cfg.max_len}  |  batch_size: {cfg.batch_size}")
    order_extra = f" (window={cfg.perturb_window})" if cfg.order_condition == "local_shuffle" else ""
    print(f"order_condition   : {cfg.order_condition}{order_extra}  (word-order ablation; keyed on data_seed)")
    print(f"padding strategy  : {pad_strategy}")
    print(f"vocab size (built): {bundle.vocab_size} (cap {cfg.vocab_size})  |  pad={bundle.pad_idx} unk={bundle.unk_idx}")
    print(f"num classes       : {bundle.num_classes}")
    print(f"label mapping     : {bundle.label_to_name}")

    splits = {"train": bundle.train_dataset, "val": bundle.val_dataset, "test": bundle.test_dataset}

    print("\n-- split sizes --")
    for name, dsx in splits.items():
        print(f"  {name:<5}: {len(dsx):,}")

    print("\n-- class distribution (count per label index) --")
    for name, dsx in splits.items():
        print(f"  {name:<5}: {_class_distribution(dsx.labels, bundle.num_classes)}")

    print("\n-- sequence length (pre-truncation tokens) & truncation rate --")
    for name, dsx in splits.items():
        st = _length_stats(dsx.orig_lens, cfg.max_len)
        print(f"  {name:<5}: mean={st['mean']:<6} p95={st['p95']:<4} max={st['max']:<5} "
              f"trunc_rate@{cfg.max_len}={st['trunc_rate']:.2%}")

    itos = bundle.vocab.get_itos()
    sample = bundle.train_dataset[0]
    decoded = " ".join(itos[i] for i in sample["input_ids"])
    print("\n-- sample (train[0]) --")
    print(f"  label    : {sample['label']} ({bundle.label_to_name[sample['label']]})")
    print(f"  orig_len : {sample['orig_len']}  truncated: {sample['truncated']}")
    print(f"  tokens   : {decoded[:160]}{'...' if len(decoded) > 160 else ''}")

    batch = next(iter(bundle.train_loader))
    pad_ratio = (batch["input_ids"] == bundle.pad_idx).float().mean().item()
    n_trunc = int(batch["truncated"].sum().item()) if "truncated" in batch else 0
    print("\n-- one train batch --")
    print(f"  input_ids: shape={tuple(batch['input_ids'].shape)} dtype={batch['input_ids'].dtype}")
    print(f"  lengths  : shape={tuple(batch['lengths'].shape)} dtype={batch['lengths'].dtype} "
          f"(min={batch['lengths'].min().item()}, max={batch['lengths'].max().item()})")
    print(f"  labels   : shape={tuple(batch['labels'].shape)} dtype={batch['labels'].dtype}")
    print(f"  padding ratio in batch    : {pad_ratio:.2%}")
    print(f"  truncated samples in batch: {n_trunc}")
    print(line)


def _smoke_test(bundle: PipelineBundle) -> None:
    """Baseline forward + loss to prove the batch->model->loss wiring (pitfall checks)."""
    import torch.nn as nn
    from models import AverageEmbeddingClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AverageEmbeddingClassifier(
        vocab_size=bundle.vocab_size,
        num_classes=bundle.num_classes,
        pad_idx=bundle.pad_idx,
        embed_dim=128,
    ).to(device)

    batch = to_device(next(iter(bundle.train_loader)), device)
    logits = model(batch["input_ids"], batch["lengths"])
    loss = nn.CrossEntropyLoss()(logits, batch["labels"])

    print("-- AverageEmbeddingClassifier smoke test --")
    print(f"  device      : {device}")
    print(f"  logits shape: {tuple(logits.shape)} (expected ({bundle.config.batch_size}, {bundle.num_classes}))")
    print(f"  loss        : {loss.item():.4f}")
    print("Pipeline OK. Import build_pipeline / DataConfig from this module.")
    print("=" * 70)


if __name__ == "__main__":
    _bundle = build_pipeline(DataConfig())
    describe_pipeline(_bundle)
    _smoke_test(_bundle)
