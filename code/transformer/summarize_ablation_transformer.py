"""Aggregate metrics.json files across Transformer runs into one ablation table.

Reads outputs/<tag>/metrics.json next to this script and prints (and optionally
saves) a table sorted by the swept variable. Default sweep variable is `embed_dim`.

Usage (run from inside code/transformer/):
    python summarize_ablation_transformer.py                          # embed_dim sweep
    python summarize_ablation_transformer.py --sweep num_layers       # layer-depth sweep
    python summarize_ablation_transformer.py --sweep train_fraction   # learning-curve sweep
    python summarize_ablation_transformer.py --save-md ablation_embed_dim_transformer.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PRESET_TAGS = {
    "embed_dim":      {"embed64", "baseline", "embed256"},
    "num_layers":     {"layers1", "baseline", "layers3"},
    "train_fraction": {"lc25", "lc50", "baseline"},
}


def load_runs(out_root: Path, wanted: set | None = None):
    runs = []
    for d in sorted(out_root.iterdir()):
        if wanted is not None and d.name not in wanted:
            continue
        f = d / "metrics.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent / "outputs")
    ap.add_argument("--sweep", type=str, default="embed_dim")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    if not args.out_root.exists():
        print(f"out_root not found: {args.out_root}")
        return

    wanted = PRESET_TAGS.get(args.sweep)
    runs = load_runs(args.out_root, wanted)
    if not runs:
        print(f"no runs found under {args.out_root}")
        return

    rows = []
    for tag, m in runs:
        cfg = m["config"]
        rows.append({
            "tag":            tag,
            args.sweep:       cfg.get(args.sweep),
            "num_layers":     cfg.get("num_layers"),
            "nhead":          cfg.get("nhead"),
            "dim_feedforward":cfg.get("dim_feedforward"),
            "pooling":        cfg.get("pooling"),
            "dropout":        cfg.get("dropout"),
            "params":         m["model"]["trainable_params"],
            "best_epoch":     m["model"]["best_epoch"],
            "best_val_f1":    m["model"]["best_val_f1"],
            "test_acc":       m["test"]["accuracy"],
            "test_f1":        m["test"]["macro_f1"],
            "train_sec":      m["train_time_sec"],
        })
    rows.sort(key=lambda r: (r[args.sweep] if r[args.sweep] is not None else 0))

    header = ("tag", args.sweep, "layers", "heads", "ff", "pool", "drop",
              "params", "best_ep", "val_f1", "test_acc", "test_f1", "t(s)")
    widths  = [12, 10, 7, 6, 5, 5, 5, 10, 8, 8, 9, 8, 7]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(
            r["tag"],
            r[args.sweep],
            r["num_layers"],
            r["nhead"],
            r["dim_feedforward"],
            r["pooling"],
            r["dropout"],
            f"{r['params']:,}",
            r["best_epoch"],
            f"{r['best_val_f1']:.4f}",
            f"{r['test_acc']:.4f}",
            f"{r['test_f1']:.4f}",
            f"{r['train_sec']:.1f}",
        ))

    if args.save_md is not None:
        md = ["# Transformer ablation: " + args.sweep + " sweep",
              "",
              "| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            md.append("| " + " | ".join([
                str(r["tag"]),
                str(r[args.sweep]),
                str(r["num_layers"]),
                str(r["nhead"]),
                str(r["dim_feedforward"]),
                str(r["pooling"]),
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
