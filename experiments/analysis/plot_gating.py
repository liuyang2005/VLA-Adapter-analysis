"""把 read_gating.py 产出的 csv 画成逐层 tanh(g) 图。

值通常都极小（接近 0），所以同时输出:
  - 主图: 逐层 tanh(g) 柱状图（真实尺度，凸显"全部接近 0"）
  - 副信息: 在标题标注最大绝对值与所在层

用法:
  python -m experiments.analysis.plot_gating \
    --csv experiments/analysis/out/gating_object_pro.csv \
    --out experiments/analysis/out/fig_B1_gating.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="experiments/analysis/out/fig_B1_gating.png")
    ap.add_argument("--title", default="LIBERO-Object-Pro")
    args = ap.parse_args()

    layers, tanh_g = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            tanh_g.append(float(row["tanh_g"]))

    # 最大绝对值与所在层，用于标注
    abs_vals = [abs(v) for v in tanh_g]
    max_i = abs_vals.index(max(abs_vals))
    max_v = tanh_g[max_i]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(layers, tanh_g, color="steelblue")
    bars[max_i].set_color("crimson")  # 高亮最大值层
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Policy layer index (0–23)")
    plt.ylabel("tanh(g)  —  C^R injection strength")
    plt.title(
        f"Per-layer learned gating ({args.title})\n"
        f"all |tanh(g)| ≤ {max(abs_vals):.4f}  (max at layer {max_i}: {max_v:+.4f})"
    )
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    print(f"max |tanh(g)| = {max(abs_vals):.6f} at layer {max_i} (value {max_v:+.6f})")


if __name__ == "__main__":
    main()
