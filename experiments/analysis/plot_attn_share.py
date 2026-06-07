"""把 run_attn_share.py 产出的 csv 画成逐层 attention 三段占比堆叠图。

用法:
  python -m experiments.analysis.plot_attn_share \
    --csv experiments/analysis/out/attn_share_object.csv \
    --out experiments/analysis/out/fig_B2_attn_share.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="experiments/analysis/out/fig_B2_attn_share.png")
    ap.add_argument("--title", default="LIBERO-Object-Pro")
    args = ap.parse_args()

    layers, tok, caq, cr = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            tok.append(float(row["share_tokens"]))
            caq.append(float(row["share_CAQ"]))
            cr.append(float(row["share_CR"]))

    plt.figure(figsize=(10, 4.5))
    plt.bar(layers, tok, label="self tokens", color="lightgray")
    plt.bar(layers, caq, bottom=tok, label="C^AQ (ActionQuery)", color="seagreen")
    bottom2 = [t + a for t, a in zip(tok, caq)]
    plt.bar(layers, cr, bottom=bottom2, label="C^R (Raw, gated)", color="crimson")
    plt.xlabel("Policy layer index (0–23)")
    plt.ylabel("mean attention share")
    plt.title(f"Per-layer attention share over conditions ({args.title})")
    plt.xticks(layers)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")

    avg_cr = sum(cr) / len(cr)
    print(f"avg C^R share across layers = {avg_cr:.4f}")


if __name__ == "__main__":
    main()
