"""门控干预评测（独立脚本，不修改任何源码）。

原理: monkey-patch run_libero_eval.initialize_model —— 在它返回前对 action_head
的 gating_factor 做干预，然后调用原版 eval_libero 跑评测。源码零改动。

门控 tanh(g) 乘在 C^R(Raw 图像特征) 的 attention 分数上 (action_heads.py:391)。

用法 (在 AutoDL 上, 与 run_libero_eval.py 相同的评测参数, 额外加 --gating_mode):
  # baseline (不干预, 应复现原成功率)
  python -m experiments.analysis.run_gating_eval \
    --use_proprio True --num_images_in_input 2 --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Object-Pro \
    --task_suite_name libero_object --use_pro_version True \
    --num_trials_per_task 10 --gating_mode none

  # 屏蔽全部 C^R
  ... --gating_mode zero_all_CR
  # 只关第 5 层
  ... --gating_mode zero_layer --gating_layer 5
  # 只把第 5 层拉满
  ... --gating_mode full_layer --gating_layer 5

注意: --gating_mode / --gating_layer 通过环境变量传入(不进 draccus 的 cfg),
所以放在命令任意位置都行, 但推荐用环境变量形式更稳:
  GATING_MODE=zero_all_CR python -m experiments.analysis.run_gating_eval ...
本脚本同时支持从命令行剥离 --gating_mode/--gating_layer 再交给 draccus。
"""
import math
import os
import sys

import torch

import experiments.robot.libero.run_libero_eval as rle


def _set_tanh(block, target):
    target = max(-0.9999, min(0.9999, target))
    g = math.atanh(target)
    with torch.no_grad():
        block.gating_factor.fill_(g)


def apply_intervention(action_head, mode, layer=None):
    blocks = action_head.model.mlp_resnet_blocks
    if mode == "none":
        pass
    elif mode == "zero_all_CR":
        for b in blocks:
            _set_tanh(b, 0.0)
    elif mode == "zero_layer":
        assert layer is not None, "zero_layer 需要 gating_layer"
        _set_tanh(blocks[layer], 0.0)
    elif mode == "full_layer":
        assert layer is not None, "full_layer 需要 gating_layer"
        _set_tanh(blocks[layer], 1.0)
    else:
        raise ValueError(f"unknown gating mode: {mode}")
    after = [math.tanh(b.gating_factor.detach().float().item()) for b in blocks]
    print(f"[GATING] mode={mode} layer={layer}", flush=True)
    print("[GATING] tanh(g) after =", [f"{v:+.4f}" for v in after], flush=True)


def _parse_and_strip_gating_args():
    """从 sys.argv 剥离 --gating_mode / --gating_layer, 返回 (mode, layer)。

    剥离后剩下的 argv 全部交给 draccus 解析(它只认识原评测参数)。
    也支持环境变量 GATING_MODE / GATING_LAYER 作为后备。
    """
    mode = os.environ.get("GATING_MODE", "none")
    layer = os.environ.get("GATING_LAYER")
    layer = int(layer) if layer is not None else None

    argv = sys.argv[1:]
    cleaned = []
    i = 0
    while i < len(argv):
        if argv[i] == "--gating_mode":
            mode = argv[i + 1]; i += 2
        elif argv[i] == "--gating_layer":
            layer = int(argv[i + 1]); i += 2
        else:
            cleaned.append(argv[i]); i += 1
    sys.argv = [sys.argv[0]] + cleaned
    return mode, layer


def main():
    mode, layer = _parse_and_strip_gating_args()

    # monkey-patch: 包装原 initialize_model, 在返回前注入干预
    _orig_init = rle.initialize_model

    def _patched_init(cfg):
        model, action_head, proprio_projector, noisy_action_projector, processor = _orig_init(cfg)
        if mode != "none" and action_head is not None:
            apply_intervention(action_head, mode, layer)
        else:
            print(f"[GATING] mode=none (baseline, no intervention)", flush=True)
        return model, action_head, proprio_projector, noisy_action_projector, processor

    rle.initialize_model = _patched_init

    # 调用原版评测入口(它是 @draccus.wrap(), 会解析剩余 sys.argv)
    rle.eval_libero()


if __name__ == "__main__":
    main()
