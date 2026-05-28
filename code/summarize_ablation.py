"""Aggregate metrics.json files across LSTM runs into one ablation table.

Reads code/outputs/lstm/<tag>/metrics.json for each subfolder and prints/saves a
table sorted by the swept variable. Default sweep variable is `embed_dim`.

Usage:
    python summarize_ablation.py                       # embed_dim sweep
    python summarize_ablation.py --sweep hidden_size   # hidden-size sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    rows = []
    for tag, m in runs:
        cfg = m["config"]
        rows.append({
            "tag": tag,
            args.sweep: cfg.get(args.sweep),
            "hidden_size": cfg.get("hidden_size"),
            "num_layers": cfg.get("num_layers"),
            "bidirectional": cfg.get("bidirectional"),
            "dropout": cfg.get("dropout"),
            "params": m["model"]["trainable_params"],
            "best_epoch": m["model"]["best_epoch"],
            "best_val_f1": m["model"]["best_val_f1"],
            "test_acc": m["test"]["accuracy"],
            "test_f1": m["test"]["macro_f1"],
            "train_sec": m["train_time_sec"],
        })
    rows.sort(key=lambda r: (r[args.sweep] if r[args.sweep] is not None else 0))

    header = ("tag", args.sweep, "hidden", "L", "bidir", "drop", "params",
              "best_ep", "val_f1", "test_acc", "test_f1", "t(s)")
    widths = [12, 10, 7, 3, 6, 5, 10, 8, 8, 9, 8, 7]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(
            r["tag"],
            r[args.sweep],
            r["hidden_size"],
            r["num_layers"],
            str(r["bidirectional"]),
            r["dropout"],
            f"{r['params']:,}",
            r["best_epoch"],
            f"{r['best_val_f1']:.4f}",
            f"{r['test_acc']:.4f}",
            f"{r['test_f1']:.4f}",
            f"{r['train_sec']:.1f}",
        ))

    if args.save_md is not None:
        md = ["# LSTM ablation: " + args.sweep + " sweep",
              "",
              "| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            md.append("| " + " | ".join([
                str(r["tag"]),
                str(r[args.sweep]),
                str(r["hidden_size"]),
                str(r["num_layers"]),
                str(r["bidirectional"]),
                str(r["dropout"]),
                f"{r['params']:,}",
                str(r["best_epoch"]),
                f"{r['best_val_f1']:.4f}",
                f"{r['test_acc']:.4f}",
                f"{r['test_f1']:.4f}",
                f"{r['train_sec']:.1f}",
            ]) + " |")
        args.save_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nsaved markdown -> {args.save_md}")


if __name__ == "__main__":
    main()
