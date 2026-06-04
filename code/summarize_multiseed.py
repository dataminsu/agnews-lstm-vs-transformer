"""Aggregate multi-seed ablation runs into mean +/- std tables.

Reads outputs/<model>/<tag>/metrics.json for tags matching --tag-prefix,
groups them by the swept config key, and reports mean and standard
deviation across seeds for val_f1, test_acc, test_f1, and train_time.
The winner row (max mean val_f1) is flagged for selection — per PDF
section 3.3, the test set is held out and never used to choose configs.

Usage:
    python summarize_multiseed.py --tag-prefix stage1_emb --sweep embed_dim
    python summarize_multiseed.py --tag-prefix stage2_drop --sweep dropout \
        --save-md ablation_dropout.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_runs(out_root: Path, tag_prefix: str):
    runs = []
    for d in sorted(out_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(tag_prefix):
            continue
        f = d / "metrics.json"
        if f.exists():
            runs.append((d.name, json.loads(f.read_text(encoding="utf-8"))))
    return runs


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=Path(__file__).parent / "outputs" / "lstm")
    ap.add_argument("--tag-prefix", type=str, required=True,
                    help="Only aggregate runs whose folder name starts with this")
    ap.add_argument("--sweep", type=str, required=True,
                    help="Config key whose values define the groups (e.g. embed_dim)")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    runs = load_runs(args.out_root, args.tag_prefix)
    if not runs:
        print(f"no runs matching prefix '{args.tag_prefix}' under {args.out_root}")
        return

    groups = {}
    for tag, m in runs:
        cfg = m["config"]
        key = cfg.get(args.sweep)
        groups.setdefault(key, []).append((tag, cfg, m))

    rows = []
    for sweep_val, items in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        seeds = [it[1].get("model_seed", it[1].get("seed")) for it in items]
        val_f1s = [it[2]["model"]["best_val_f1"] for it in items]
        test_accs = [it[2]["test"]["accuracy"] for it in items]
        test_f1s = [it[2]["test"]["macro_f1"] for it in items]
        train_secs = [it[2]["train_time_sec"] for it in items]
        params = items[0][2]["model"]["trainable_params"]
        v_m, v_s = mean_std(val_f1s)
        a_m, a_s = mean_std(test_accs)
        f_m, f_s = mean_std(test_f1s)
        t_m, _ = mean_std(train_secs)
        rows.append({
            "sweep_val": sweep_val,
            "n": len(items),
            "seeds": seeds,
            "params": params,
            "val_f1_mean": v_m, "val_f1_std": v_s,
            "test_acc_mean": a_m, "test_acc_std": a_s,
            "test_f1_mean": f_m, "test_f1_std": f_s,
            "train_sec_mean": t_m,
            "tags": [it[0] for it in items],
        })

    # Winner is the group with the highest MEAN val_f1 (per PDF section 3.3,
    # the test set is never used for config selection).
    winner_idx = max(range(len(rows)), key=lambda i: rows[i]["val_f1_mean"])

    header = [args.sweep, "n", "params",
              "val_f1 (mean +/- std)", "test_acc (mean +/- std)",
              "test_f1 (mean +/- std)", "t(s) mean", "winner"]

    def fmt_pm(m, s):
        return f"{m:.4f} +/- {s:.4f}"

    def cells(i, r):
        return [
            str(r["sweep_val"]),
            str(r["n"]),
            f"{r['params']:,}",
            fmt_pm(r["val_f1_mean"], r["val_f1_std"]),
            fmt_pm(r["test_acc_mean"], r["test_acc_std"]),
            fmt_pm(r["test_f1_mean"], r["test_f1_std"]),
            f"{r['train_sec_mean']:.1f}",
            "<-- best val_f1" if i == winner_idx else "",
        ]

    widths = [max(len(header[i]), max(len(cells(j, r)[i]) for j, r in enumerate(rows)))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(f"{args.out_root.name} multi-seed ablation: {args.sweep} sweep "
          f"(tag-prefix='{args.tag_prefix}')")
    print(fmt.format(*header))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for i, r in enumerate(rows):
        print(fmt.format(*cells(i, r)))

    print()
    print(f"Winner by mean val_f1: {args.sweep}={rows[winner_idx]['sweep_val']}  "
          f"(val_f1 {rows[winner_idx]['val_f1_mean']:.4f} +/- "
          f"{rows[winner_idx]['val_f1_std']:.4f})")

    # Per-seed detail.
    print("\nPer-seed detail:")
    for r in rows:
        print(f"  {args.sweep}={r['sweep_val']}: " + ", ".join(
            f"s{seed}={tag.split('_')[-1]}:val_f1={vf:.4f}/test_f1={tf:.4f}"
            for tag, seed, vf, tf in zip(
                r["tags"], r["seeds"],
                [items[2]["model"]["best_val_f1"] for items in groups[r["sweep_val"]]],
                [items[2]["test"]["macro_f1"] for items in groups[r["sweep_val"]]],
            )
        ))

    if args.save_md is not None:
        md = [
            f"# {args.out_root.name} multi-seed ablation: {args.sweep} sweep",
            "",
            f"Tag prefix: `{args.tag_prefix}`. Winner selected by **mean val_f1** "
            f"(per PDF section 3.3: test set is held out and never used for "
            f"config selection).",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        for i, r in enumerate(rows):
            md.append("| " + " | ".join(cells(i, r)) + " |")
        md += [
            "",
            f"**Winner:** {args.sweep}={rows[winner_idx]['sweep_val']} "
            f"(val_f1 {rows[winner_idx]['val_f1_mean']:.4f} +/- "
            f"{rows[winner_idx]['val_f1_std']:.4f}, n={rows[winner_idx]['n']} seeds).",
        ]
        args.save_md.write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nsaved markdown -> {args.save_md}")


if __name__ == "__main__":
    main()
