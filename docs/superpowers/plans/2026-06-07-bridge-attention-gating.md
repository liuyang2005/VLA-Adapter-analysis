# Bridge Attention 门控诊断与改进 —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 诊断 VLA-Adapter 的逐层标量门控（可视化 + 推理期干预），据结果决定是否做向量门控改进。

**Architecture:** 分析代码独立放在 `experiments/analysis/`，复用仓库现有的 `get_action_head`/`get_vla` 加载逻辑与 `run_libero_eval.py` 评测循环。先做零训练的 Part B（B1 读门控→B2 抓attention→B3 推理干预），再据 B 的出口判断决定 Part D（向量门控短训练）。

**Tech Stack:** PyTorch 2.2.0, VLA-Adapter(Prismatic), LIBERO, matplotlib。

**开发模式:** 代码在本地写（路径 `/home/leserein/code/VLA-Adapter`），git push → AutoDL `~/autodl-tmp/VLA-Adapter` 拉取后运行。每个 Task 明确标注 [本地] 写代码 / [AutoDL] 运行。

**关键代码事实（已核对）:**
- 门控访问路径：`action_head.model.mlp_resnet_blocks[i].gating_factor`（标量 nn.Parameter），共 24 层。
- Pro 版门控 `tanh(g)` 调制 C^R（Raw 图像特征，代码变量 `h_t`/`k_task`），见 `action_heads.py:391`。
- 加载：`get_action_head(cfg, llm_dim)`（`openvla_utils.py:487`），从 HF 或本地 `outputs/LIBERO-Object-Pro` 加载 `action_head--checkpoint.pt`。
- 用户已有基线：4090 上官方 Object-Pro = 95.2%（论文 H100 99.6%，硬件差异）。

---

## Part B：门控诊断（零训练）

### Task 1: 项目目录与门控读取脚本（B1）

**Files:**
- Create: `experiments/analysis/__init__.py`（空文件）
- Create: `experiments/analysis/read_gating.py`

- [ ] **Step 1: [本地] 创建目录与空 __init__.py**

```bash
mkdir -p experiments/analysis
touch experiments/analysis/__init__.py
```

- [ ] **Step 2: [本地] 写门控读取脚本**

创建 `experiments/analysis/read_gating.py`。该脚本只加载 action_head（不跑仿真、不加载 VLM），读出 24 层 `gating_factor`，算 `tanh(g)`，存成 csv 并打印。

```python
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
```

- [ ] **Step 3: [本地] 提交**

```bash
git add experiments/analysis/__init__.py experiments/analysis/read_gating.py
git commit -m "feat(analysis): add gating_factor reader script (B1)"
git push
```

- [ ] **Step 4: [AutoDL] 拉取并运行，验证能读出 24 个门控值**

```bash
cd ~/autodl-tmp/VLA-Adapter && git pull
python -m experiments.analysis.read_gating \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --out experiments/analysis/out/gating_object_pro.csv
```

Expected: 打印 24 行 `layer 0..23` 的 g 与 tanh(g)，并保存 csv。
⚠️ 若 `--llm_dim 896` 报维度不匹配，按报错里的实际维度调整（Qwen2.5-0.5B 通常是 896）。把输出贴回。

---

### Task 2: 门控可视化画图（B1 出图）

**Files:**
- Create: `experiments/analysis/plot_gating.py`

- [ ] **Step 1: [本地] 写画图脚本**

创建 `experiments/analysis/plot_gating.py`，读 csv，画逐层 tanh(g) 柱状/折线图。

```python
"""把 read_gating.py 产出的 csv 画成逐层 tanh(g) 图。

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
    args = ap.parse_args()

    layers, tanh_g = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            layers.append(int(row["layer"]))
            tanh_g.append(float(row["tanh_g"]))

    plt.figure(figsize=(10, 4))
    plt.bar(layers, tanh_g, color="steelblue")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Policy layer index")
    plt.ylabel("tanh(g)  —  C^R injection strength")
    plt.title("Per-layer learned gating (LIBERO-Object-Pro)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: [本地] 提交**

```bash
git add experiments/analysis/plot_gating.py
git commit -m "feat(analysis): add per-layer gating plot (B1)"
git push
```

- [ ] **Step 3: [AutoDL] 拉取并出图**

```bash
cd ~/autodl-tmp/VLA-Adapter && git pull
python -m experiments.analysis.plot_gating \
  --csv experiments/analysis/out/gating_object_pro.csv \
  --out experiments/analysis/out/fig_B1_gating.png
```

Expected: 生成 `fig_B1_gating.png`。把图下载/截图回来看：哪些层 tanh(g) 大（C^R 注入强）、哪些接近 0（几乎关闭）。这是第一张诊断图与第一个发现。

---

### Task 3: 推理期门控干预（B3 核心，论文未做）

**Files:**
- Create: `experiments/analysis/gating_intervention.py`（提供在评测前修改门控的工具函数）
- Modify: `experiments/robot/libero/run_libero_eval.py`（加几个干预相关 CLI 参数与调用钩子）

> 说明：B3 复用现成的 LIBERO 评测循环，只在"加载模型后、跑评测前"插入一次门控干预。不重写评测逻辑。

- [ ] **Step 1: [本地] 写干预工具函数**

创建 `experiments/analysis/gating_intervention.py`：

```python
"""推理期对 action_head 各层门控做反事实干预（不重训）。

干预模式:
  - "none": 不干预（baseline，必须先验证能复现原成功率）
  - "zero_layer": 把第 --layer 层的 gating_factor 置为 -inf 等价（tanh→ 用大负数使 tanh(g)≈-1? ）
                  这里采用直接覆盖 forward 中的 ratio 更安全：把该层 tanh(g) 强制设 0。
  - "full_layer": 把第 --layer 层 tanh(g) 强制设 1。

实现方式：直接改写 gating_factor 的值，使 tanh(g) 达到目标。
  tanh(g)=0  -> g=0
  tanh(g)=1  -> 数值上用 g=5（tanh(5)=0.9999）足够近似
  tanh(g)=-1 -> g=-5
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
    """对 action_head 做门控干预。返回原始 g 值列表以便恢复。"""
    blocks = action_head.model.mlp_resnet_blocks
    original = [b.gating_factor.detach().clone() for b in blocks]

    if mode == "none":
        pass
    elif mode == "zero_all_CR":
        for b in blocks:
            _set_tanh(b, 0.0)          # 关闭所有层的 C^R 注入
    elif mode == "zero_layer":
        assert layer is not None
        _set_tanh(blocks[layer], 0.0)  # 只关第 layer 层
    elif mode == "full_layer":
        assert layer is not None
        _set_tanh(blocks[layer], 1.0)  # 只把第 layer 层拉满
    else:
        raise ValueError(f"unknown mode {mode}")

    return original


def restore(action_head, original):
    blocks = action_head.model.mlp_resnet_blocks
    with torch.no_grad():
        for b, g in zip(blocks, original):
            b.gating_factor.copy_(g)
```

- [ ] **Step 2: [本地] 在 run_libero_eval.py 接入干预参数**

在 `run_libero_eval.py` 的 `GenerateConfig` 里新增字段，并在 `initialize_model` 之后、评测循环之前调用 `apply_intervention`。

先查看 run_libero_eval.py 中 `GenerateConfig` 定义和 `initialize_model` 调用位置（约 line 489），在其后插入：

```python
# 在文件顶部 import 区加：
from experiments.analysis.gating_intervention import apply_intervention

# 在 GenerateConfig 中加字段（与其它字段并列）：
    gating_mode: str = "none"            # none | zero_all_CR | zero_layer | full_layer
    gating_layer: int = -1               # 当 mode=zero_layer/full_layer 时指定层

# 在 model, action_head, ... = initialize_model(cfg) 之后插入：
    if cfg.gating_mode != "none":
        layer = None if cfg.gating_layer < 0 else cfg.gating_layer
        apply_intervention(action_head, cfg.gating_mode, layer)
        print(f"[GATING INTERVENTION] mode={cfg.gating_mode} layer={layer}")
```

- [ ] **Step 3: [本地] 提交**

```bash
git add experiments/analysis/gating_intervention.py experiments/robot/libero/run_libero_eval.py
git commit -m "feat(analysis): inference-time gating intervention hooks (B3)"
git push
```

- [ ] **Step 4: [AutoDL] 先验证 baseline（mode=none）复现 95.2%，再小规模干预**

为省时间，干预实验用减量 episode（每任务 10 次）。先确认无干预 sanity：

```bash
cd ~/autodl-tmp/VLA-Adapter && git pull
# baseline sanity（减量），确认接入参数没破坏原流程
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object --use_pro_version True \
  --num_trials_per_task 10 --gating_mode none
```

Expected: 成功率应接近 95%（减量会有波动）。若明显异常，说明接入有 bug，先修。

- [ ] **Step 5: [AutoDL] 跑干预实验**

```bash
# 关闭所有层 C^R 注入，看掉多少
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object --use_pro_version True \
  --num_trials_per_task 10 --gating_mode zero_all_CR

# 逐层置零扫描（示例：第 5 层），换 --gating_layer 跑多个关键层
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object --use_pro_version True \
  --num_trials_per_task 10 --gating_mode zero_layer --gating_layer 5
```

Expected: 记录每种干预的成功率，和 baseline 对比。把所有数字贴回，我们整理成表 B1 + 图 B3。重点看：关掉哪些层掉得最多（=最重要的层），是否和 B1 门控值大的层吻合。

---

### Task 4: B2 — attention 条件占比（可选，按 B1/B3 进展决定是否做）

> 若 B1+B3 已足够支撑诊断结论，B2 可省。B2 需要 forward hook 抓 `attn_weights`，实现稍复杂，留作加深分析。具体实现待 B1/B3 出结果后再细化（届时更新本任务的代码）。

- [ ] 占位：B1/B3 完成后评估是否需要 B2，需要则补充 hook 代码。

---

## Part B 出口判断

完成 B1（门控可视化）+ B3（逐层干预敏感度）后，回答：
- 是否有多层共享一个标量、却表现出"该差异化却被迫统一"的迹象？
- 是否有层门控该开（C^R 重要）却学成接近 0，或反之？

**若是 → 进入 Part D（向量门控）。若否 → 诚实报告"标量门控已足够"，项目以诊断收尾。**

---

## Part D：向量/分头门控改进（条件性，2-4 次 Object 短训练）

> 仅在 Part B 诊断出标量门控局限后执行。具体代码在 B 出结果后细化（改 `MLPResNetBlock_Pro.gating_factor` 从标量 `torch.zeros(1)` 改为向量 `torch.zeros(num_heads)` 或 `torch.zeros(dim)`，并相应改 line 391 的广播）。届时更新本任务。

- [ ] 占位：依据 Part B 诊断结论，设计向量门控的确切维度与训练对比方案，更新本计划后执行。

---

## 报告

- [ ] 汇总 B1（图）、B3（表+图）、（D 若做）的结果，写中文项目报告：动机（含 95.2% vs 99.6% 硬件现象铺垫）→ 方法 → 诊断发现 → （改进）→ 结论。
