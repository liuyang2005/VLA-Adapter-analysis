"""把 run_attn_share.py 产出的 csv 画图。

输出两个子图:
  上: 逐层 attention 三段【总占比】堆叠图 (受段长度影响, C^R 段长易显大)
  下: 逐层【每-token平均】注意力 (消除段长度差异, 看模型是否真重视每个 token)

兼容旧 csv (无 per-token 列时只画上图)。

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
    pt_tok, pt_caq, pt_cr = [], [], []
    has_pertoken = False
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        has_pertoken = "pertoken_CR" in reader.fieldnames
        for row in reader:
            layers.append(int(row["layer"]))
            tok.append(float(row["share_tokens"]))
            caq.append(float(row["share_CAQ"]))
            cr.append(float(row["share_CR"]))
            if has_pertoken:
                pt_tok.append(float(row["pertoken_tokens"]))
                pt_caq.append(float(row["pertoken_CAQ"]))
                pt_cr.append(float(row["pertoken_CR"]))

    nrows = 2 if has_pertoken else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4.5 * nrows))
    if nrows == 1:
        axes = [axes]

    # 上: 总占比堆叠
    ax = axes[0]
    ax.bar(layers, tok, label="self tokens", color="lightgray")
    ax.bar(layers, caq, bottom=tok, label="C^AQ (ActionQuery)", color="seagreen")
    bottom2 = [t + a for t, a in zip(tok, caq)]
    ax.bar(layers, cr, bottom=bottom2, label="C^R (Raw, gated)", color="crimson")
    ax.set_ylabel("total attention share")
    ax.set_title(f"Per-layer TOTAL attention share ({args.title})\n"
                 f"(C^R segment has many more tokens, so total share looks large)")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    # 下: 每-token平均
    if has_pertoken:
        ax = axes[1]
        width = 0.27
        x = layers
        ax.bar([i - width for i in x], pt_tok, width, label="self tokens", color="gray")
        ax.bar(x, pt_caq, width, label="C^AQ per token", color="seagreen")
        ax.bar([i + width for i in x], pt_cr, width, label="C^R per token", color="crimson")
        ax.set_xlabel("Policy layer index (0–23)")
        ax.set_ylabel("mean attention PER TOKEN")
        ax.set_title("Per-layer PER-TOKEN attention "
                     "(removes segment-length effect: does the model really weight each token?)")
        ax.set_xticks(layers)
        ax.legend(loc="upper right")
    else:
        axes[0].set_xlabel("Policy layer index (0–23)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    print(f"avg C^R total-share = {sum(cr)/len(cr):.4f}")
    if has_pertoken:
        print(f"avg C^R per-token = {sum(pt_cr)/len(pt_cr):.6f} | "
              f"avg C^AQ per-token = {sum(pt_caq)/len(pt_caq):.6f}")


if __name__ == "__main__":
    main()
