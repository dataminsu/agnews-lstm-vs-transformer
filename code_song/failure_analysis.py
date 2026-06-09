"""Head-to-head failure analysis for the LSTM vs Transformer comparison, plus a
word-order "Type A" view (correct on original order, wrong once order is removed).

Loads the saved best.pt checkpoints, runs BOTH models on the SAME test examples,
joins predictions by example index, and reports:

  Part 1 (required, brief sec. 9): on ORIGINAL-order inputs, examples where
    - both models are wrong,
    - only the LSTM is wrong (Transformer correct),
    - only the Transformer is wrong (LSTM correct),
    each with text, true label, both predictions, and softmax confidence.

  Part 2 (word-order tie-in, ablation PDF sec. 10 Type A): per model, examples it
    classifies correctly on original order but misclassifies under full shuffle —
    i.e. examples that genuinely needed word order.

Run from the code/ root, pointing at the seed-42 checkpoints:
    python failure_analysis.py \
        --lstm-orig  outputs/lstm/wo_orig_s42  --tr-orig  transformer/outputs/transformer/wo_orig_s42 \
        --lstm-full  outputs/lstm/wo_full_s42  --tr-full  transformer/outputs/transformer/wo_full_s42 \
        --save-md failure_analysis.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data_pipeline import DataConfig, build_pipeline
from models import LSTMClassifier, TransformerEncoderClassifier

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_model(kind: str, run_dir: Path, bundle, device):
    cfg = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["config"]
    if kind == "lstm":
        model = LSTMClassifier(
            vocab_size=bundle.vocab_size, num_classes=bundle.num_classes, pad_idx=bundle.pad_idx,
            embed_dim=cfg["embed_dim"], hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"], bidirectional=cfg.get("bidirectional", False),
            dropout=cfg["dropout"],
        )
    else:
        model = TransformerEncoderClassifier(
            vocab_size=bundle.vocab_size, num_classes=bundle.num_classes, pad_idx=bundle.pad_idx,
            embed_dim=cfg["embed_dim"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"],
            max_len=cfg["max_len"], pooling=cfg.get("pooling", "mean"),
        )
    state = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def infer(model, loader, device):
    """Return {idx: (pred, confidence)} and {idx: (text, true, truncated)}."""
    pred_by_idx, meta_by_idx = {}, {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids, lengths)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for j, idx in enumerate(batch["indices"]):
            pred_by_idx[idx] = (int(pred[j]), float(conf[j]))
            meta_by_idx[idx] = (batch["texts"][j], int(batch["labels"][j]),
                                bool(batch["truncated"][j]))
    return pred_by_idx, meta_by_idx


def clip(text, n=140):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm-orig", type=Path, required=True)
    ap.add_argument("--tr-orig", type=Path, required=True)
    ap.add_argument("--lstm-full", type=Path, default=None)
    ap.add_argument("--tr-full", type=Path, default=None)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=6, help="examples to show per category")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = None

    # ---- Part 1: both models on ORIGINAL-order test ----
    bundle = build_pipeline(DataConfig(order_condition="original", data_seed=args.data_seed,
                                       model_seed=args.data_seed))
    cls = bundle.class_names
    lstm = load_model("lstm", args.lstm_orig, bundle, device)
    tr = load_model("transformer", args.tr_orig, bundle, device)
    lp, meta = infer(lstm, bundle.test_loader, device)
    tp, _ = infer(tr, bundle.test_loader, device)

    cats = {"both_wrong": [], "lstm_only_wrong": [], "tr_only_wrong": []}
    n_both_ok = 0
    for idx, (text, true, trunc) in meta.items():
        lpred, lconf = lp[idx]
        tpred, tconf = tp[idx]
        lwrong, twrong = lpred != true, tpred != true
        rec = dict(idx=idx, text=text, true=true, trunc=trunc,
                   lpred=lpred, lconf=lconf, tpred=tpred, tconf=tconf)
        if not lwrong and not twrong:
            n_both_ok += 1
        elif lwrong and twrong:
            cats["both_wrong"].append(rec)
        elif lwrong:
            cats["lstm_only_wrong"].append(rec)
        else:
            cats["tr_only_wrong"].append(rec)

    L = ["# Failure analysis — LSTM vs Transformer (seed 42, original-order test)", ""]
    L.append(f"Test examples: {len(meta):,}.  Both correct: {n_both_ok:,}.  "
             f"Both wrong: {len(cats['both_wrong'])}.  "
             f"LSTM-only wrong: {len(cats['lstm_only_wrong'])}.  "
             f"Transformer-only wrong: {len(cats['tr_only_wrong'])}.")
    L.append("")
    titles = {
        "both_wrong": "Both models wrong (hard / ambiguous examples)",
        "lstm_only_wrong": "Only the LSTM is wrong (Transformer correct)",
        "tr_only_wrong": "Only the Transformer is wrong (LSTM correct)",
    }
    # Sort each category by the *wrong* model's confidence (confidently-wrong = most telling).
    def sort_key(c):
        if c == "lstm_only_wrong":
            return lambda r: -r["lconf"]
        if c == "tr_only_wrong":
            return lambda r: -r["tconf"]
        return lambda r: -(r["lconf"] + r["tconf"]) / 2
    for c in ["both_wrong", "lstm_only_wrong", "tr_only_wrong"]:
        L.append(f"\n## {titles[c]}  ({len(cats[c])} total)\n")
        L.append("| # | text | true | LSTM pred (conf) | Transf. pred (conf) | trunc |")
        L.append("|---|------|------|------------------|---------------------|-------|")
        for i, r in enumerate(sorted(cats[c], key=sort_key(c))[: args.n], 1):
            L.append(f"| {i} | {clip(r['text'])} | {cls[r['true']]} | "
                     f"{cls[r['lpred']]} ({r['lconf']:.2f}) | {cls[r['tpred']]} ({r['tconf']:.2f}) | "
                     f"{'Y' if r['trunc'] else ''} |")

    # ---- Part 2: Type A — correct on original, wrong under full shuffle ----
    if args.lstm_full and args.tr_full:
        full = build_pipeline(DataConfig(order_condition="full_shuffle", data_seed=args.data_seed,
                                         model_seed=args.data_seed))
        lstm_f = load_model("lstm", args.lstm_full, full, device)
        tr_f = load_model("transformer", args.tr_full, full, device)
        lpf, _ = infer(lstm_f, full.test_loader, device)
        tpf, _ = infer(tr_f, full.test_loader, device)

        def type_a(orig_preds, full_preds, conf_src):
            out = []
            for idx, (text, true, trunc) in meta.items():
                if orig_preds[idx][0] == true and full_preds[idx][0] != true:
                    out.append((idx, text, true, full_preds[idx][0], full_preds[idx][1]))
            return out

        la = type_a(lp, lpf, lpf)
        ta = type_a(tp, tpf, tpf)
        L.append("\n## Part 2 — Type A: correct on original order, wrong after FULL shuffle\n")
        L.append(f"(Examples that genuinely needed word order.)  "
                 f"LSTM: {len(la):,} such examples.  Transformer: {len(ta):,}.\n")
        for name, rows in [("LSTM", la), ("Transformer", ta)]:
            L.append(f"\n### {name} — original-correct → full-shuffle-wrong\n")
            L.append("| # | text | true | wrong pred under shuffle (conf) |")
            L.append("|---|------|------|---------------------------------|")
            for i, (idx, text, true, wp, wc) in enumerate(sorted(rows, key=lambda x: -x[4])[: args.n], 1):
                L.append(f"| {i} | {clip(text)} | {cls[true]} | {cls[wp]} ({wc:.2f}) |")

    out = "\n".join(L)
    print(out)
    if args.save_md is not None:
        args.save_md.write_text(out + "\n", encoding="utf-8")
        print(f"\nsaved -> {args.save_md}")


if __name__ == "__main__":
    main()
