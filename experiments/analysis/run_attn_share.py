"""B2: 统计每层 attention 在 [自注意力tokens | C^AQ | C^R] 三段的平均占比。

原理: monkey-patch MLPResNetBlock_Pro.forward —— 完整复刻原 forward 逻辑(保证
模型行为不变), 仅在算出 attn_weights 后, 把三段(tokens / C^AQ=adapter / C^R=task)
的权重和按 query 平均, 累加到全局收集器。然后复用 run_libero_eval 跑少量 episode。
源码零改动。

attn_weights 最后一维顺序 = [T tokens | K_a adapter(=C^AQ) | K_t task(=C^R)]
(见 action_heads.py:388-393)。

用法 (AutoDL):
  python -m experiments.analysis.run_attn_share \
    --use_proprio True --num_images_in_input 2 --use_film False \
    --pretrained_checkpoint outputs/LIBERO-Object-Pro \
    --task_suite_name libero_object --use_pro_version True \
    --num_trials_per_task 3 \
    --share_out experiments/analysis/out/attn_share_object.csv
"""
import csv
import math
import os
import sys
from collections import defaultdict

import torch

import experiments.robot.libero.run_libero_eval as rle
from prismatic.models.action_heads import MLPResNetBlock_Pro, apply_rope


# 全局收集器: layer_id -> [sum_tokens, sum_caq, sum_cr, count]
_COLLECTOR = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
# 各层段长度: layer_id -> (T, K_a, K_t)
_SEGLEN = {}


def _make_patched_forward(orig_block, layer_id):
    """返回一个绑定到具体 block 的 patched forward, 复刻原逻辑并记录三段占比。"""

    def patched_forward(x, h_a=None, h_t=None, p=None):
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

        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # ---- 记录三段占比 (不影响计算) ----
        with torch.no_grad():
            # attn_weights: (B, H, T, T+K_a+K_t)。后两维顺序 = tokens|adapter|task
            w = attn_weights.float()
            tok = w[..., :T].sum(dim=-1).mean().item()
            caq = w[..., T:T + K_a].sum(dim=-1).mean().item()
            cr = w[..., T + K_a:].sum(dim=-1).mean().item()
            acc = _COLLECTOR[layer_id]
            acc[0] += tok; acc[1] += caq; acc[2] += cr; acc[3] += 1
            # 记录各段 token 数(用于每-token归一化), 只记一次
            if acc[3] == 1:
                _SEGLEN[layer_id] = (T, K_a, K_t)

        v_list = [v_tokens, v_adapter, v_task]
        v_combined = torch.cat(v_list, dim=2)
        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)
        x = self.ffn(output + x)
        return x

    return patched_forward


def main():
    # 剥离 --share_out
    share_out = "experiments/analysis/out/attn_share.csv"
    argv = sys.argv[1:]
    cleaned, i = [], 0
    while i < len(argv):
        if argv[i] == "--share_out":
            share_out = argv[i + 1]; i += 2
        else:
            cleaned.append(argv[i]); i += 1
    sys.argv = [sys.argv[0]] + cleaned

    _orig_init = rle.initialize_model

    def _patched_init(cfg):
        out = _orig_init(cfg)
        action_head = out[1]
        if action_head is not None:
            blocks = action_head.model.mlp_resnet_blocks
            for li, b in enumerate(blocks):
                b.forward = _make_patched_forward(b, li)
            print(f"[ATTN] patched {len(blocks)} blocks' forward", flush=True)
        return out

    rle.initialize_model = _patched_init
    rle.eval_libero()

    # 生效自检: 收集器为空说明 patch 没生效(forward 没被走到)
    if len(_COLLECTOR) == 0 or all(v[3] == 0 for v in _COLLECTOR.values()):
        raise RuntimeError(
            "[ATTN] collector is EMPTY — patched forward was never called. "
            "Check use_pro_version=True (only MLPResNetBlock_Pro is patched) and that "
            "instance-level forward override took effect."
        )

    # 评测结束后写出平均占比
    os.makedirs(os.path.dirname(share_out), exist_ok=True)
    with open(share_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "share_tokens", "share_CAQ", "share_CR", "n_calls",
            "len_tokens", "len_CAQ", "len_CR",
            "pertoken_tokens", "pertoken_CAQ", "pertoken_CR",
        ])
        for li in sorted(_COLLECTOR):
            s_tok, s_caq, s_cr, n = _COLLECTOR[li]
            if n == 0:
                continue
            T, K_a, K_t = _SEGLEN.get(li, (0, 0, 0))
            avg_tok, avg_caq, avg_cr = s_tok / n, s_caq / n, s_cr / n
            # 每-token平均 = 该段总占比 / 该段token数
            pt_tok = avg_tok / T if T else 0.0
            pt_caq = avg_caq / K_a if K_a else 0.0
            pt_cr = avg_cr / K_t if K_t else 0.0
            w.writerow([li, avg_tok, avg_caq, avg_cr, n,
                        T, K_a, K_t, pt_tok, pt_caq, pt_cr])
    print(f"[ATTN] saved attention shares to {share_out}", flush=True)


if __name__ == "__main__":
    main()
