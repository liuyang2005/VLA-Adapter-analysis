"""推理期对 action_head 各层门控做反事实干预（不重训）。

门控 tanh(g) 乘在 C^R(Raw 图像特征) 的 attention 分数上 (action_heads.py:391)。
通过改写 gating_factor 的值，使 tanh(g) 达到目标，从而在推理期屏蔽/增强 C^R。

干预模式:
  - "none":        不干预（baseline，必须先验证能复现原成功率）
  - "zero_all_CR": 把所有层 tanh(g) 设 0（屏蔽全部 C^R 注入）
  - "zero_layer":  只把第 --layer 层 tanh(g) 设 0
  - "full_layer":  只把第 --layer 层 tanh(g) 设 1（强制拉满该层 C^R 注入）

数值: tanh(g)=0 -> g=0; tanh(g)=±1 用 g=±5 近似 (tanh(5)=0.9999)。
"""
import math

import torch


def _set_tanh(block, target):
    """把 block.gating_factor 设到使 tanh(g)≈target 的值。target∈[-1,1]。"""
    target = max(-0.9999, min(0.9999, target))
    g = math.atanh(target)
    with torch.no_grad():
        block.gating_factor.fill_(g)


def apply_intervention(action_head, mode, layer=None):
    """对 action_head 做门控干预。返回原始 gating_factor 列表以便恢复。"""
    blocks = action_head.model.mlp_resnet_blocks
    original = [b.gating_factor.detach().clone() for b in blocks]

    if mode == "none":
        pass
    elif mode == "zero_all_CR":
        for b in blocks:
            _set_tanh(b, 0.0)
    elif mode == "zero_layer":
        assert layer is not None, "zero_layer 需要 --gating_layer"
        _set_tanh(blocks[layer], 0.0)
    elif mode == "full_layer":
        assert layer is not None, "full_layer 需要 --gating_layer"
        _set_tanh(blocks[layer], 1.0)
    else:
        raise ValueError(f"unknown gating mode: {mode}")

    # 打印改动后各层 tanh(g)，确认干预生效
    after = [math.tanh(b.gating_factor.detach().float().item()) for b in blocks]
    print(f"[GATING] mode={mode} layer={layer}")
    print("[GATING] tanh(g) after =", [f"{v:+.4f}" for v in after])
    return original


def restore(action_head, original):
    blocks = action_head.model.mlp_resnet_blocks
    with torch.no_grad():
        for b, g in zip(blocks, original):
            b.gating_factor.copy_(g)
