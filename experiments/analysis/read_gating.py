"""读取 VLA-Adapter action head 各层门控 gating_factor 并保存。

只加载 action head（不需要 VLM / 仿真），是最轻量的诊断第一步。
用法:
  python -m experiments.analysis.read_gating \
    --pretrained_checkpoint outputs/LIBERO-Object-Pro \
    --out experiments/analysis/out/gating_object_pro.csv
"""
import argparse
import csv
import os
import math

import torch

from experiments.robot.openvla_utils import get_action_head


class _Cfg:
    """get_action_head 需要的最小 cfg。"""
    def __init__(self, ckpt):
        self.pretrained_checkpoint = ckpt
        self.use_l1_regression = True
        self.use_pro_version = True
        self.save_version = ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_checkpoint", required=True)
    ap.add_argument("--llm_dim", type=int, default=896,
                    help="Qwen2.5-0.5B 的隐藏维度，与训练一致")
    ap.add_argument("--out", default="experiments/analysis/out/gating.csv")
    args = ap.parse_args()

    cfg = _Cfg(args.pretrained_checkpoint)
    action_head = get_action_head(cfg, llm_dim=args.llm_dim)

    rows = []
    blocks = action_head.model.mlp_resnet_blocks
    for i, block in enumerate(blocks):
        g = block.gating_factor.detach().float().item()
        rows.append((i, g, math.tanh(g)))
        print(f"layer {i:2d}: g={g:+.4f}  tanh(g)={math.tanh(g):+.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "g", "tanh_g"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} layers to {args.out}")


if __name__ == "__main__":
    main()
