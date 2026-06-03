"""LSTM baseline training & evaluation for AG News.

Trains LSTMClassifier with the plan's base config (embed=128, hidden=256, 2 layers,
unidirectional, dropout=0.3), Adam lr=1e-3, batch=64, 8 epochs, seed=42.

Selects the best epoch by validation macro-F1, then evaluates ONCE on the official
test set. Writes history, metrics, and the confusion matrix to
outputs/lstm/<tag>/ (defaults to outputs/lstm/baseline/).

Usage:
    conda activate agnews-dl
    python -u train_lstm.py
    # Override knobs from CLI for ablations:
    python -u train_lstm.py --embed-dim 64 --tag embed64
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_pipeline import DataConfig, build_pipeline, set_seed, to_device
from models import LSTMClassifier, count_parameters


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss ({loss.item()}); aborting before it corrupts the run")
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = batch["labels"].size(0)
        total += loss.item() * bs
        n += bs
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    y_true, y_pred = [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch["input_ids"], batch["lengths"])
        bs = batch["labels"].size(0)
        total += criterion(logits, batch["labels"]).item() * bs
        n += bs
        y_true.extend(batch["labels"].cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    avg = total / n
    return (
        avg,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        y_true,
        y_pred,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tag", type=str, default="baseline",
                    help="Output subfolder name under outputs/lstm/")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Max grad norm for clip_grad_norm_; 0 disables")
    ap.add_argument("--seed", type=int, default=42,
                    help="Controls 90/10 split, DataLoader shuffle, and model init")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "outputs" / "lstm" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    cfg = DataConfig(
        max_len=args.max_len,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"config: {cfg}")
    bundle = build_pipeline(cfg)
    print(f"splits: train={len(bundle.train_dataset):,}  "
          f"val={len(bundle.val_dataset):,}  test={len(bundle.test_dataset):,}")

    set_seed(args.seed)  # so model init is independent of pipeline RNG drift
    model = LSTMClassifier(
        vocab_size=bundle.vocab_size,
        num_classes=bundle.num_classes,
        pad_idx=bundle.pad_idx,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"model : LSTM embed={args.embed_dim} hidden={args.hidden_size} "
          f"layers={args.num_layers} bidir={args.bidirectional} dropout={args.dropout}")
    print(f"params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "epoch_time": []}
    # Seed best_state with epoch-0 weights so a NaN-only run still produces a
    # loadable checkpoint instead of crashing on load_state_dict(None).
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_f1, best_epoch = -1.0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        et0 = time.time()
        tr = train_one_epoch(model, bundle.train_loader, optimizer, criterion, device,
                             grad_clip=args.grad_clip)
        vl, acc, f1, _, _ = evaluate(model, bundle.val_loader, criterion, device)
        dt = time.time() - et0
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        history["epoch_time"].append(dt)
        marker = ""
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"epoch {ep:>2}/{args.epochs}: train_loss={tr:.4f}  "
              f"val_loss={vl:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}  "
              f"({dt:.1f}s){marker}")
    train_time = time.time() - t0
    print(f"training done in {train_time:.1f}s, best epoch={best_epoch} val_f1={best_f1:.4f}")

    # Final test evaluation: ONCE, with the best-val checkpoint.
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, bundle.test_loader, criterion, device
    )
    print(f"TEST: loss={test_loss:.4f}  acc={test_acc:.4f}  macro_f1={test_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(bundle.num_classes)))
    print("confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Persist artifacts.
    metrics = {
        "config": vars(args),
        "data": {
            "vocab_size": bundle.vocab_size,
            "pad_idx": bundle.pad_idx,
            "num_classes": bundle.num_classes,
            "class_names": bundle.class_names,
            "train_size": len(bundle.train_dataset),
            "val_size": len(bundle.val_dataset),
            "test_size": len(bundle.test_dataset),
        },
        "model": {
            "trainable_params": n_params,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
        },
        "test": {"loss": test_loss, "accuracy": test_acc, "macro_f1": test_f1},
        "train_time_sec": train_time,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    np.save(out_dir / "confusion_matrix.npy", cm)
    torch.save(best_state, out_dir / "best.pt")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
