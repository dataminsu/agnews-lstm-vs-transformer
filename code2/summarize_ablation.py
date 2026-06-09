"""Collect the metrics.json files from a set of runs into one ablation table.

Reads outputs/<model>/<tag>/metrics.json for each subfolder and prints (and
optionally saves) a table sorted by the swept variable. Works for both models:
point --out-root at the LSTM or the Transformer output folder.

Usage:
    python summarize_ablation.py                                            # LSTM, embed_dim sweep
    python summarize_ablation.py --out-root outputs/transformer             # Transformer
    python summarize_ablation.py --sweep num_layers --out-root outputs/transformer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Config columns we show when they are present in metrics.json["config"].
# LSTM runs carry hidden_size/bidirectional; Transformer runs carry nhead/
# dim_feedforward. Anything missing for a model is simply skipped.
OPTIONAL_COLUMNS = [
    ("hidden_size", "hidden"),
    ("num_layers", "layers"),
    ("bidirectional", "bidir"),
    ("nhead", "heads"),
    ("dim_feedforward", "ff"),
    ("dropout", "drop"),
]


def load_runs(out_root: Path):
    runs = []
    for d in sorted(out_root.iterdir()):
        f = d / "metrics.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent / "outputs" / "lstm")
    ap.add_argument("--sweep", type=str, default="embed_dim")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    runs = load_runs(args.out_root)
    if not runs:
        print(f"no runs found under {args.out_root}")
        return

    # The folder name (e.g. "lstm" or "transformer") labels the table.
    model_name = args.out_root.name

    # Keep only the optional columns that at least one run actually has.
    present = []
    for key, label in OPTIONAL_COLUMNS:
        if any(m["config"].get(key) is not None for _, m in runs):
            present.append((key, label))

    rows = []
    for tag, m in runs:
        cfg = m["config"]
        row = {
            "tag": tag,
            args.sweep: cfg.get(args.sweep),
            "params": m["model"]["trainable_params"],
            "best_epoch": m["model"]["best_epoch"],
            "best_val_f1": m["model"]["best_val_f1"],
            "test_acc": m["test"]["accuracy"],
            "test_f1": m["test"]["macro_f1"],
            "train_sec": m["train_time_sec"],
        }
        for key, _ in present:
            row[key] = cfg.get(key)
        rows.append(row)
    rows.sort(key=lambda r: (r[args.sweep] if r[args.sweep] is not None else 0))

    header = ["tag", args.sweep] + [label for _, label in present] + [
        "params", "best_ep", "val_f1", "test_acc", "test_f1", "t(s)"]

    def cells(r):
        out = [str(r["tag"]), str(r[args.sweep])]
        out += [str(r[key]) for key, _ in present]
        out += [
            f"{r['params']:,}",
            str(r["best_epoch"]),
            f"{r['best_val_f1']:.4f}",
            f"{r['test_acc']:.4f}",
            f"{r['test_f1']:.4f}",
            f"{r['train_sec']:.1f}",
        ]
        return out

    widths = [max(len(header[i]), max(len(cells(r)[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"{model_name} ablation: {args.sweep} sweep")
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(*cells(r)))

    if args.save_md is not None:
        md = [f"# {model_name} ablation: {args.sweep} sweep", "",
              "| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            md.append("| " + " | ".join(cells(r)) + " |")
        args.save_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nsaved markdown -> {args.save_md}")


if __name__ == "__main__":
    main()
