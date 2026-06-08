"""读取 VLA-Adapter action head 各层门控 gating_factor 并保存。

支持标量门控(每层1个)与 per-head 向量门控(每层 num_heads 个)。
只加载 action head（不需要 VLM / 仿真），是最轻量的诊断步骤。

用法:
  # 标量版
  python -m experiments.analysis.read_gating \
    --pretrained_checkpoint outputs/...PARTD-scalar--5000_chkpt \
    --out experiments/analysis/out/gating_scalar.csv
  # 向量版 (必须加 --gating_per_head)
  python -m experiments.analysis.read_gating \
    --pretrained_checkpoint outputs/...PARTD-perhead--5000_chkpt \
    --gating_per_head --out experiments/analysis/out/gating_perhead.csv
"""
import argparse
import csv
import os
import math

import torch

from experiments.robot.openvla_utils import get_action_head


class _Cfg:
    """get_action_head 需要的最小 cfg。"""
    def __init__(self, ckpt, gating_per_head):
        self.pretrained_checkpoint = ckpt
        self.use_l1_regression = True
        self.use_pro_version = True
        self.save_version = ckpt
        self.gating_per_head = gating_per_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", required=True)
    ap.add_argument("--gating_per_head", action="store_true",
                    help="ckpt 用 per-head 向量门控时必须加")
    ap.add_argument("--llm_dim", type=int, default=896)
    ap.add_argument("--out", default="experiments/analysis/out/gating.csv")
    args = ap.parse_args()

    cfg = _Cfg(args.pretrained_checkpoint, args.gating_per_head)
    action_head = get_action_head(cfg, llm_dim=args.llm_dim)

    blocks = action_head.model.mlp_resnet_blocks
    rows = []
    for i, block in enumerate(blocks):
        g = block.gating_factor.detach().float().flatten().tolist()  # 标量->[x], 向量->[x0..x7]
        tanh_g = [math.tanh(v) for v in g]
        rows.append((i, g, tanh_g))
        if len(g) == 1:
            print(f"layer {i:2d}: tanh(g)={tanh_g[0]:+.4f}")
        else:
            spread = max(tanh_g) - min(tanh_g)
            print(f"layer {i:2d}: tanh(g) per-head ="
                  f" [{', '.join(f'{v:+.3f}' for v in tanh_g)}]  spread={spread:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_head = len(rows[0][2])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        if n_head == 1:
            w.writerow(["layer", "tanh_g"])
            for i, g, tg in rows:
                w.writerow([i, tg[0]])
        else:
            w.writerow(["layer"] + [f"tanh_g_h{h}" for h in range(n_head)] + ["spread"])
            for i, g, tg in rows:
                w.writerow([i] + tg + [max(tg) - min(tg)])
    print(f"\nSaved {len(rows)} layers ({n_head} gate value(s)/layer) to {args.out}")


if __name__ == "__main__":
    main()
