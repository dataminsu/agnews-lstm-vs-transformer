"""Combine the word-order ablation runs (BOTH models) into one report-ready table.

Unlike the per-model summarizers, this one is built for the word-order ablation:
it groups by `order_condition` (read from each run's saved config, so it is robust
to tag naming), reports mean +/- std across seeds for val/test metrics, and adds
the two quantities the report needs:

    Delta F1        = mean test_f1(condition) - mean test_f1(original)
    Order Sensitivity = (test_f1(original) - test_f1(condition)) / test_f1(original)

Reads:
    outputs/lstm/wo_*/metrics.json                      (run_word_order_lstm.sh)
    transformer/outputs/transformer/wo_*/metrics.json   (run_word_order_transformer.sh)
Tolerant of partial data: prints whatever exists so far.

Usage (from the code/ root):
    python summarize_word_order.py
    python summarize_word_order.py --save-md ablation_word_order.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Canonical display order; anything else is appended after these.
COND_ORDER = ["original", "local_shuffle", "full_shuffle"]


def load_condition_runs(root: Path):
    """Return {order_condition: [metrics, ...]} for wo_* runs under `root`."""
    groups: dict[str, list] = {}
    if not root.is_dir():
        return groups
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("wo_"):
            continue
        f = d / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        cond = m.get("config", {}).get("order_condition", "unknown")
        groups.setdefault(cond, []).append(m)
    return groups


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def summarize_model(name: str, root: Path):
    groups = load_condition_runs(root)
    if not groups:
        return None
    conds = [c for c in COND_ORDER if c in groups] + \
            [c for c in groups if c not in COND_ORDER]

    # baseline = original mean test_f1 (for Delta / sensitivity)
    orig_f1 = None
    if "original" in groups:
        orig_f1 = mean_std([m["test"]["macro_f1"] for m in groups["original"]])[0]

    rows = []
    for c in conds:
        ml = groups[c]
        vf = mean_std([m["model"]["best_val_f1"] for m in ml])
        ta = mean_std([m["test"]["accuracy"] for m in ml])
        tf = mean_std([m["test"]["macro_f1"] for m in ml])
        if orig_f1 is not None:
            delta = tf[0] - orig_f1
            sens = (orig_f1 - tf[0]) / orig_f1 if orig_f1 else float("nan")
            delta_s = f"{delta:+.4f}" if c != "original" else "-"
            sens_s = f"{sens:+.4f}" if c != "original" else "-"
        else:
            delta_s = sens_s = "n/a"
        rows.append({
            "cond": c, "n": len(ml),
            "val_f1": vf, "test_acc": ta, "test_f1": tf,
            "delta": delta_s, "sens": sens_s,
        })
    return {"model": name, "rows": rows}


def render(tables, fmt_md: bool):
    header = ["model", "order_condition", "n",
              "val_f1 (mean+/-std)", "test_acc (mean+/-std)",
              "test_f1 (mean+/-std)", "Delta F1", "Order Sens."]

    def pm(ms):
        return f"{ms[0]:.4f} +/- {ms[1]:.4f}"

    body = []
    for t in tables:
        for r in t["rows"]:
            body.append([
                t["model"], r["cond"], str(r["n"]),
                pm(r["val_f1"]), pm(r["test_acc"]), pm(r["test_f1"]),
                r["delta"], r["sens"],
            ])

    if fmt_md:
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(lines)

    widths = [max(len(header[i]), max((len(row[i]) for row in body), default=0))
              for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*header), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    out += [fmt.format(*row) for row in body]
    return "\n".join(out)


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm-root", type=Path, default=here / "outputs" / "lstm")
    ap.add_argument("--transformer-root", type=Path,
                    default=here / "transformer" / "outputs" / "transformer")
    ap.add_argument("--save-md", type=Path, default=None)
    args = ap.parse_args()

    tables = []
    for name, root in [("LSTM", args.lstm_root), ("Transformer", args.transformer_root)]:
        t = summarize_model(name, root)
        if t is None:
            print(f"(no wo_* runs found under {root})")
        else:
            tables.append(t)

    if not tables:
        print("nothing to summarize yet.")
        return

    print("\nWord Order Perturbation ablation -- combined (test set = 7,600 examples)\n")
    print(render(tables, fmt_md=False))
    print("\nDelta F1 / Order Sensitivity are relative to each model's own "
          "original-order mean test_f1.")

    if args.save_md is not None:
        md = [
            "# Word Order Perturbation ablation (combined)",
            "",
            "Mean +/- std across 5 model-seeds per condition. Delta F1 and Order "
            "Sensitivity = (F1_orig - F1_cond)/F1_orig are relative to each model's "
            "own original-order mean test_f1. The token-order perturbation is keyed "
            "on data_seed, so all seeds in a condition see identical perturbed data.",
            "",
            render(tables, fmt_md=True),
            "",
        ]
        args.save_md.write_text("\n".join(md), encoding="utf-8")
        print(f"\nsaved markdown -> {args.save_md}")


if __name__ == "__main__":
    main()
