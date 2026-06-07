"""门控/条件干预评测（独立脚本，不修改任何源码）。

两类干预，语义严格区分:

【gate 类】只改 gating_factor 参数值 (tanh(g))。注意: gate 乘在 C^R 的 attention
  logits 上, 不是 hard mask。gate=0 时 C^R logits=0, softmax 后 C^R 仍获得注意力
  (实测约 85%)。所以 gate_zero_all ≠ 屏蔽 C^R, 它只是"把门控设回 0"(≈官方本来状态)。
    - none:           不干预 (baseline)
    - gate_zero_all:  所有层 tanh(g)=0   (旧名 zero_all_CR, 保留为别名)
    - gate_zero_layer:第 --layer 层 tanh(g)=0   (旧名 zero_layer)
    - gate_full_layer:第 --layer 层 tanh(g)=1   (旧名 full_layer)

【mask 类】真正屏蔽 C^R: 把 C^R 的 attention logits 设为 -1e9, softmax 后权重→0,
  C^R 对输出无贡献。通过 monkey-patch MLPResNetBlock_Pro.forward 实现 (复刻原逻辑,
  仅在 C^R logits 处注入 -1e9)。这才能回答"C^R 到底有没有用"。
    - mask_all_CR:    所有层屏蔽 C^R
    - mask_layer:     只屏蔽第 --layer 层的 C^R

用法 (AutoDL, 与 run_libero_eval.py 相同参数, 额外 --gating_mode [--gating_layer N]):
  python -m experiments.analysis.run_gating_eval \
    --use_proprio True --num_images_in_input 2 --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Object-Pro \
    --task_suite_name libero_object --use_pro_version True \
    --num_trials_per_task 10 --gating_mode mask_all_CR \
    --run_id_note MASK-allCR
"""
import math
import os
import sys

import torch

import experiments.robot.libero.run_libero_eval as rle
from prismatic.models.action_heads import apply_rope


# 旧名 -> 新名 别名
_ALIAS = {
    "zero_all_CR": "gate_zero_all",
    "zero_layer": "gate_zero_layer",
    "full_layer": "gate_full_layer",
}

GATE_MODES = {"gate_zero_all", "gate_zero_layer", "gate_full_layer"}
MASK_MODES = {"mask_all_CR", "mask_layer"}


def _set_tanh(block, target):
    target = max(-0.9999, min(0.9999, target))
    g = math.atanh(target)
    with torch.no_grad():
        block.gating_factor.fill_(g)


def apply_gate_intervention(action_head, mode, layer=None):
    """gate 类: 只改 gating_factor。"""
    blocks = action_head.model.mlp_resnet_blocks
    if mode == "gate_zero_all":
        for b in blocks:
            _set_tanh(b, 0.0)
    elif mode == "gate_zero_layer":
        assert layer is not None, "gate_zero_layer 需要 --gating_layer"
        _set_tanh(blocks[layer], 0.0)
    elif mode == "gate_full_layer":
        assert layer is not None, "gate_full_layer 需要 --gating_layer"
        _set_tanh(blocks[layer], 1.0)
    after = [math.tanh(b.gating_factor.detach().float().item()) for b in blocks]
    print(f"[GATE] mode={mode} layer={layer}", flush=True)
    print("[GATE] tanh(g) after =", [f"{v:+.4f}" for v in after], flush=True)


def _make_masked_forward(orig_block):
    """返回 patched forward: 复刻原逻辑, 但把 C^R(task) 的 attention logits 设 -1e9,
    使 softmax 后 C^R 权重→0 (真正屏蔽 C^R)。"""

    def masked_forward(x, h_a=None, h_t=None, p=None):
        self = orig_block
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        h_adapter = torch.cat((h_a, p), dim=1)
        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        q_1 = self.q_proj(x)
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        s_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1))
        s_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1))
        s_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g
        # ---- 真正屏蔽 C^R: logits 设 -1e9, softmax 后该段权重→0 ----
        s_task = torch.full_like(s_task, -1e9)
        attn_scores = torch.cat([s_tokens, s_adapter, s_task], dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        x = self.ffn(output + x)
        return x

    return masked_forward


def apply_mask_intervention(action_head, mode, layer=None):
    """mask 类: patch forward 把 C^R logits 设 -1e9。"""
    blocks = action_head.model.mlp_resnet_blocks
    if mode == "mask_all_CR":
        targets = list(range(len(blocks)))
    elif mode == "mask_layer":
        assert layer is not None, "mask_layer 需要 --gating_layer"
        targets = [layer]
    else:
        raise ValueError(mode)
    for li in targets:
        blocks[li].forward = _make_masked_forward(blocks[li])
    print(f"[MASK] mode={mode} layer={layer} -> masked C^R in layers {targets}", flush=True)


def _parse_and_strip_gating_args():
    mode = os.environ.get("GATING_MODE", "none")
    layer = os.environ.get("GATING_LAYER")
    layer = int(layer) if layer is not None else None
    argv = sys.argv[1:]
    cleaned, i = [], 0
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
    mode = _ALIAS.get(mode, mode)  # 旧名映射到新名

    _orig_init = rle.initialize_model

    def _patched_init(cfg):
        out = _orig_init(cfg)
        action_head = out[1]
        if mode == "none" or action_head is None:
            print("[INTERV] mode=none (baseline, no intervention)", flush=True)
        elif mode in GATE_MODES:
            apply_gate_intervention(action_head, mode, layer)
        elif mode in MASK_MODES:
            apply_mask_intervention(action_head, mode, layer)
        else:
            raise ValueError(f"unknown gating_mode: {mode}")
        return out

    rle.initialize_model = _patched_init
    rle.eval_libero()


if __name__ == "__main__":
    main()
