"""Class-wise precision / recall / F1 from the saved confusion matrices, for the
word-order ablation. Also reports the Business<->Sci/Tech confusion (the class
pair the PDF flags as most lexically overlapping) per condition.

Reads outputs/<model>/wo_<cond>_s<seed>/confusion_matrix.npy (rows=true, cols=pred),
computes per-class F1 per seed, then mean +/- std across the 5 seeds.

Usage (from the code/ root, or pass explicit roots):
    python summarize_classwise.py
    python summarize_classwise.py --lstm-root outputs/outputs/lstm \
        --transformer-root outputs/outputs/transformer --save-md ablation_word_order_classwise.md
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

try:  # keep unicode (±, ->) printable on Windows cp949 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
COND_ORDER = ["original", "local_shuffle", "full_shuffle"]
COND_TAG = {"original": "wo_orig", "local_shuffle": "wo_local", "full_shuffle": "wo_full"}


def per_class_f1(cm: np.ndarray):
    """Return (precision, recall, f1) arrays for a confusion matrix (rows=true)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def load_cms(root: Path, cond: str):
    cms = []
    for d in sorted(root.glob(f"{COND_TAG[cond]}_s*")):
        f = d / "confusion_matrix.npy"
        if f.exists():
            cms.append(np.load(f))
    return cms


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def summarize_model(name: str, root: Path, lines: list):
    if not root.is_dir():
        lines.append(f"(no runs under {root})")
        return
    lines.append(f"\n### {name} — per-class macro-F1 (mean +/- std over 5 seeds)\n")
    header = ["order_condition"] + CLASS_NAMES + ["B->S%", "S->B%"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for cond in COND_ORDER:
        cms = load_cms(root, cond)
        if not cms:
            continue
        # per-seed per-class F1
        f1s = np.array([per_class_f1(cm)[2] for cm in cms])  # (n_seed, 4)
        cell = []
        for ci in range(4):
            m, s = mean_std(list(f1s[:, ci]))
            cell.append(f"{m:.3f}±{s:.3f}")
        # Business(2)<->Sci/Tech(3) confusion, row-normalized, averaged over seeds
        b2s, s2b = [], []
        for cm in cms:
            cm = cm.astype(float)
            b2s.append(cm[2, 3] / cm[2].sum() if cm[2].sum() else 0.0)  # true Business -> pred Sci/Tech
            s2b.append(cm[3, 2] / cm[3].sum() if cm[3].sum() else 0.0)  # true Sci/Tech -> pred Business
        cell.append(f"{100*statistics.mean(b2s):.1f}")
        cell.append(f"{100*statistics.mean(s2b):.1f}")
        lines.append("| " + " | ".join([cond] + cell) + " |")


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm-root", type=Path, default=here / "outputs" / "lstm")
    ap.add_argument("--transformer-root", type=Path,
                    default=here / "transformer" / "outputs" / "transformer")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    lines = ["# Word-order ablation — class-wise F1 + Business<->Sci/Tech confusion",
             "",
             "Per-class macro-F1 mean±std over 5 seeds. `B->S%` = share of true Business "
             "predicted as Sci/Tech; `S->B%` = true Sci/Tech predicted as Business "
             "(row-normalized, the most lexically overlapping pair)."]
    summarize_model("LSTM", args.lstm_root, lines)
    summarize_model("Transformer", args.transformer_root, lines)

    out = "\n".join(lines)
    print(out)
    if args.save_md is not None:
        args.save_md.write_text(out + "\n", encoding="utf-8")
        print(f"\nsaved -> {args.save_md}")


if __name__ == "__main__":
    main()
