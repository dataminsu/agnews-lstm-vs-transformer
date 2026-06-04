"""Aggregate multi-seed Transformer runs into mean ± std tables.

Tags follow the naming convention: <base>_s<seed>  e.g. embed64_s42 ... embed64_s46
Single-seed tags (no suffix) are also accepted (treated as n=1).

Winner is selected by mean val_f1 (test set is never used for config selection).

Usage:
    python summarize_multiseed_transformer.py                          # embed_dim sweep
    python summarize_multiseed_transformer.py --sweep dropout
    python summarize_multiseed_transformer.py --sweep num_layers
    python summarize_multiseed_transformer.py --sweep train_fraction
    python summarize_multiseed_transformer.py --save-md ablation_embed_dim.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


PRESET_BASES = {
    "embed_dim":       {"embed32", "embed64", "baseline", "embed256"},
    "dropout":         {"dropout01", "embed64", "dropout05", "dropout08"},
    "num_layers":      {"layers1", "embed64", "layers3", "layers4", "layers5"},        # dropout=0.3 (original, single-seed winner)
    "num_layers_d01":  {"layers1_d01", "dropout01", "layers3_d01", "layers4_d01", "layers5_d01"},  # dropout=0.1 (multi-seed winner, corrected)
    "train_fraction":  {"lc25", "lc50", "embed64"},
}

# alias → actual config field name
SWEEP_KEY_ALIAS = {
    "num_layers_d01": "num_layers",
}

SEED_RE = re.compile(r"^(.+)_s(\d+)$")


def base_name(tag: str) -> str:
    m = SEED_RE.match(tag)
    return m.group(1) if m else tag


def load_runs(out_root: Path, wanted_bases: set | None = None):
    runs = []
    for d in sorted(out_root.iterdir()):
        bn = base_name(d.name)
        if wanted_bases is not None and bn not in wanted_bases:
            continue
        f = d / "metrics.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text())))
    return runs


def aggregate(runs, sweep_key):
    groups: dict[str, list] = {}
    for tag, m in runs:
        bn = base_name(tag)
        groups.setdefault(bn, []).append(m)

    rows = []
    for bn, mlist in groups.items():
        cfg = mlist[0]["config"]
        val_f1s = [m["model"]["best_val_f1"] for m in mlist]
        accs    = [m["test"]["accuracy"]      for m in mlist]
        f1s     = [m["test"]["macro_f1"]      for m in mlist]
        times   = [m["train_time_sec"]        for m in mlist]
        n = len(mlist)
        rows.append({
            "tag":            bn,
            "n":              n,
            sweep_key:        cfg.get(sweep_key),
            "params":         mlist[0]["model"]["trainable_params"],
            "val_f1_mean":    statistics.mean(val_f1s),
            "val_f1_std":     statistics.stdev(val_f1s) if n > 1 else 0.0,
            "test_acc_mean":  statistics.mean(accs),
            "test_acc_std":   statistics.stdev(accs)    if n > 1 else 0.0,
            "test_f1_mean":   statistics.mean(f1s),
            "test_f1_std":    statistics.stdev(f1s)     if n > 1 else 0.0,
            "time_mean":      statistics.mean(times),
        })
    rows.sort(key=lambda r: (r[sweep_key] if r[sweep_key] is not None else 0))

    best_val = max(r["val_f1_mean"] for r in rows)
    for r in rows:
        r["winner"] = "<-- best val_f1" if r["val_f1_mean"] == best_val else ""
    return rows


def fmt_stat(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent / "outputs" / "transformer")
    ap.add_argument("--sweep",   type=str, default="embed_dim")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    if not args.out_root.exists():
        print(f"out_root not found: {args.out_root}")
        return

    wanted_bases = PRESET_BASES.get(args.sweep)
    runs = load_runs(args.out_root, wanted_bases)
    if not runs:
        print(f"no runs found under {args.out_root}")
        return

    cfg_key = SWEEP_KEY_ALIAS.get(args.sweep, args.sweep)
    rows = aggregate(runs, cfg_key)

    # Console output
    header = (args.sweep, "n", "params", "val_f1 (mean+/-std)",
              "test_acc (mean+/-std)", "test_f1 (mean+/-std)", "t(s) mean", "winner")
    widths  = [10, 3, 12, 22, 22, 22, 10, 16]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(
            r[cfg_key], r["n"], f"{r['params']:,}",
            fmt_stat(r["val_f1_mean"], r["val_f1_std"]),
            fmt_stat(r["test_acc_mean"], r["test_acc_std"]),
            fmt_stat(r["test_f1_mean"], r["test_f1_std"]),
            f"{r['time_mean']:.1f}",
            r["winner"],
        ))

    winner = next(r for r in rows if r["winner"])
    print(f"\n**Winner:** {cfg_key}={winner[cfg_key]} "
          f"(val_f1 {winner['val_f1_mean']:.4f} +/- {winner['val_f1_std']:.4f}, n={winner['n']} seeds).")

    # Markdown output
    if args.save_md is not None:
        md_header = (cfg_key, "n", "params",
                     "val_f1 (mean +/- std)", "test_acc (mean +/- std)",
                     "test_f1 (mean +/- std)", "t(s) mean", "winner")
        lines = [
            f"# transformer multi-seed ablation: {args.sweep} sweep",
            "",
            "Winner selected by **mean val_f1** (per PDF section 3.3: test set is held out and never used for config selection).",
            "",
            "| " + " | ".join(md_header) + " |",
            "|" + "|".join(["---"] * len(md_header)) + "|",
        ]
        for r in rows:
            lines.append("| " + " | ".join([
                str(r[cfg_key]), str(r["n"]), f"{r['params']:,}",
                fmt_stat(r["val_f1_mean"], r["val_f1_std"]),
                fmt_stat(r["test_acc_mean"], r["test_acc_std"]),
                fmt_stat(r["test_f1_mean"], r["test_f1_std"]),
                f"{r['time_mean']:.1f}",
                r["winner"],
            ]) + " |")
        lines += [
            "",
            f"**Winner:** {cfg_key}={winner[cfg_key]} "
            f"(val_f1 {winner['val_f1_mean']:.4f} +/- {winner['val_f1_std']:.4f}, n={winner['n']} seeds).",
        ]
        args.save_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nsaved markdown -> {args.save_md}")


if __name__ == "__main__":
    main()
