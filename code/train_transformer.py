"""Transformer Encoder training & evaluation for AG News.

Trains TransformerEncoderClassifier with the plan's base config
(embed=128, nhead=4, 2 layers, FF 256, dropout=0.3, mean pooling),
Adam lr=1e-3, batch=64, 8 epochs, seed=42.

Selects the best epoch by validation macro-F1, then evaluates ONCE on the
official test set. Writes history, metrics, and the confusion matrix to
outputs/transformer/<tag>/ (defaults to outputs/transformer/baseline/).

Usage:
    conda activate agnews-dl
    python -u train_transformer.py
    # Ablation examples:
    python -u train_transformer.py --embed-dim 64  --tag embed64
    python -u train_transformer.py --embed-dim 256 --tag embed256
    python -u train_transformer.py --num-layers 1  --tag layers1
    python -u train_transformer.py --num-layers 3  --tag layers3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_pipeline import DataConfig, build_pipeline, set_seed, to_device
from models import TransformerEncoderClassifier, count_parameters


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss ({loss.item()}) — aborting")
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
    ap.add_argument("--embed-dim",       type=int,   default=128)
    ap.add_argument("--nhead",           type=int,   default=4)
    ap.add_argument("--num-layers",      type=int,   default=2)
    ap.add_argument("--dim-feedforward", type=int,   default=256)
    ap.add_argument("--dropout",         type=float, default=0.3)
    ap.add_argument("--pooling",         type=str,   default="mean", choices=["mean", "cls"])
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--epochs",          type=int,   default=8)
    ap.add_argument("--max-len",         type=int,   default=128)
    ap.add_argument("--train-fraction",  type=float, default=1.0)
    ap.add_argument("--batch-size",      type=int,   default=64)
    ap.add_argument("--tag",             type=str,   default="baseline",
                    help="Output subfolder name under outputs/transformer/")
    ap.add_argument("--grad-clip",       type=float, default=1.0,
                    help="Max grad norm for clip_grad_norm_; 0 disables")
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "outputs" / "transformer" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    cfg = DataConfig(
        max_len=args.max_len,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
    )
    print(f"config: {cfg}")
    bundle = build_pipeline(cfg)
    print(f"splits: train={len(bundle.train_dataset):,}  "
          f"val={len(bundle.val_dataset):,}  test={len(bundle.test_dataset):,}")

    set_seed(42)
    model = TransformerEncoderClassifier(
        vocab_size=bundle.vocab_size,
        num_classes=bundle.num_classes,
        pad_idx=bundle.pad_idx,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        pooling=args.pooling,
    ).to(device)
    n_params = count_parameters(model)
    print(f"model : Transformer embed={args.embed_dim} nhead={args.nhead} "
          f"layers={args.num_layers} ff={args.dim_feedforward} "
          f"pooling={args.pooling} dropout={args.dropout}")
    print(f"params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_f1, best_epoch = -1.0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, bundle.train_loader, optimizer, criterion, device,
                             grad_clip=args.grad_clip)
        vl, acc, f1, _, _ = evaluate(model, bundle.val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)
        marker = ""
        if f1 > best_f1:
            best_f1, best_epoch = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"epoch {ep:>2}/{args.epochs}: train_loss={tr:.4f}  "
              f"val_loss={vl:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}{marker}")
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

    # Loss curves PNG
    epochs = range(1, args.epochs + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax.plot(epochs, history["val_loss"],   marker="s", label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"Transformer Encoder — Loss Curves ({args.tag})")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # Confusion matrix PNG
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(bundle.num_classes)); ax.set_yticks(range(bundle.num_classes))
    ax.set_xticklabels(bundle.class_names, rotation=30, ha="right")
    ax.set_yticklabels(bundle.class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Transformer Encoder — Confusion Matrix ({args.tag})")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Failure analysis JSON (test set, misclassified examples with confidence)
    label_to_name = bundle.label_to_name
    failures = []
    with torch.no_grad():
        for batch in bundle.test_loader:
            texts = batch.get("texts", [""] * batch["labels"].size(0))
            truncated = batch.get("truncated", torch.zeros(batch["labels"].size(0), dtype=torch.bool))
            batch_gpu = to_device(batch, device)
            probs = F.softmax(model(batch_gpu["input_ids"], batch_gpu["lengths"]), dim=-1).cpu()
            for i in range(batch["labels"].size(0)):
                true_idx = int(batch["labels"][i])
                pred_idx = int(probs[i].argmax())
                if true_idx != pred_idx:
                    failures.append({
                        "text": texts[i],
                        "true_label": label_to_name[true_idx],
                        "pred_label": label_to_name[pred_idx],
                        "confidence": round(float(probs[i].max()), 4),
                        "probs": {label_to_name[j]: round(float(probs[i][j]), 4)
                                  for j in range(probs.size(1))},
                        "truncated": bool(truncated[i]),
                    })
    failures.sort(key=lambda r: r["confidence"], reverse=True)
    (out_dir / "failures.json").write_text(
        json.dumps(failures[:20], indent=2, ensure_ascii=False)
    )
    print(f"failures: {len(failures)} misclassified, saved top 20")

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
