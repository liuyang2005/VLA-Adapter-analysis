"""Part D: 对比标量门控 vs per-head 向量门控学到的值。

两个子图:
  上: 逐层"门控分化程度"对比 —— scalar 是单标量(分化恒为0), perhead 用 8 个 head
      的 spread(max-min)。直观展示 perhead 在深层学出分化, scalar 无从分化。
  下: perhead 的 24层×8head 门控值热图, 高亮深层(19-23)的分化。

用法:
  python -m experiments.analysis.plot_partd_gating \
    --scalar_csv experiments/analysis/out/gating_scalar.csv \
    --perhead_csv experiments/analysis/out/gating_perhead.csv \
    --out experiments/analysis/out/fig_D_gating_compare.png
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_scalar(path):
    layers, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            vals.append(abs(float(row["tanh_g"])))  # |tanh(g)|, scalar 无 spread, 用绝对值示意大小
    return layers, vals


def _read_perhead(path):
    layers, spreads, mat = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        head_cols = [c for c in reader.fieldnames if c.startswith("tanh_g_h")]
        for row in reader:
            layers.append(int(row["layer"]))
            spreads.append(float(row["spread"]))
            mat.append([float(row[c]) for c in head_cols])
    return layers, spreads, np.array(mat), len(head_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar_csv", required=True)
    ap.add_argument("--perhead_csv", required=True)
    ap.add_argument("--out", default="experiments/analysis/out/fig_D_gating_compare.png")
    args = ap.parse_args()

    s_layers, s_absg = _read_scalar(args.scalar_csv)
    p_layers, p_spread, p_mat, n_head = _read_perhead(args.perhead_csv)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # 上: 分化程度对比
    ax = axes[0]
    width = 0.4
    ax.bar([i - width / 2 for i in s_layers], s_absg, width,
           label="scalar gate |tanh(g)| (no per-head spread possible)", color="lightsteelblue")
    ax.bar([i + width / 2 for i in p_layers], p_spread, width,
           label="per-head gate spread (max−min over 8 heads)", color="crimson")
    ax.set_xlabel("Policy layer index (0–23)")
    ax.set_ylabel("gate magnitude / spread")
    ax.set_title("Scalar vs per-head gating: per-head learns DIVERGENCE in deep layers (19–23)")
    ax.set_xticks(s_layers)
    ax.legend(loc="upper left")

    # 下: perhead 8-head 热图
    ax = axes[1]
    im = ax.imshow(p_mat.T, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(p_mat).max(), vmax=np.abs(p_mat).max())
    ax.set_xlabel("Policy layer index (0–23)")
    ax.set_ylabel("attention head (0–7)")
    ax.set_title("Per-head gate tanh(g) heatmap — deep layers diverge across heads")
    ax.set_xticks(range(len(p_layers)))
    ax.set_xticklabels(p_layers)
    ax.set_yticks(range(n_head))
    fig.colorbar(im, ax=ax, label="tanh(g)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")
    # 打印关键数字
    deep = [(l, sp) for l, sp in zip(p_layers, p_spread) if l >= 19]
    print("deep-layer per-head spreads:", [f"L{l}:{sp:.4f}" for l, sp in deep])
    print(f"max scalar |tanh(g)| = {max(s_absg):.4f}  |  max per-head spread = {max(p_spread):.4f}")


if __name__ == "__main__":
    main()
