# 被低估的视觉底座：VLA-Adapter 桥接门控机制的诊断与改进

> **一句话主论点**：VLA-Adapter 的可学习门控把原始视觉特征 C^R 调制成一个"均匀、低调、冗余分布"的信息底座——它在所有表层指标上都像是被关闭、被忽略的无用信息，但实验证明它是模型不可或缺的根基，全局移除会使任务成功率从 93% 崩溃到 3%。基于这一诊断，本工作进一步将标量门控改为 per-head 向量门控，在相同训练预算下把成功率从 79% 提升到 100%，且门控分化恰好出现在诊断所定位的深层决策层。

**作者**：leserein ｜ **日期**：2026-06-08 ｜ **类型**：机制诊断（零训练）+ 模块改进（短训练验证）

---

## 1. 研究背景与动机

### 1.1 VLA-Adapter 与 Bridge Attention

VLA-Adapter（arXiv 2509.09372）是一个轻量级视觉-语言-动作模型：用 0.5B 的 Qwen2.5 骨干、无机器人预训练，在 LIBERO 仿真基准上追平 7B 大模型。其核心模块 **Bridge Attention** 从视觉语言模型（VLM）的每一层抽取两种条件，注入一个轻量动作策略网络（Policy，24 层）：

- **C^R（Raw 特征）**：图像 patch 的隐藏状态（512 个 token）；
- **C^AQ（ActionQuery 特征）**：可学习查询 token（64 个）。

其最大卖点是一个**可学习标量门控** `tanh(g)`（每层一个，共 24 个），论文称其"让模型自主决定每层注入多少 Raw 特征"。

### 1.2 论文未回答的问题

论文（Section 4.5, Table 8）已对门控做了开关消融，证明"门控加在 C^R 上"是最优配置。但论文**从未展示**：

1. 这 24 个门控值训练后究竟学成了什么样；
2. 各条件在推理时实际获得多少注意力；
3. 真正移除某种条件后模型会怎样。

本项目以官方发布的 LIBERO-Object-Pro 权重为对象，零训练、纯诊断地回答这三个问题。

### 1.3 一个方法论前提：成功率的硬件依赖性

复现时发现：官方权重在 RTX 4090 上的 LIBERO-Object 成功率为 **95.2%**，而论文（H100）报告 99.6%，环境与命令完全一致。该 4.4% 差距源于不同 GPU 架构下 bfloat16 数值差异在长序列闭环控制中的累积（README 与 OpenVLA-OFT 均明确提示此现象）。**结论**：仿真成功率的绝对值不可跨硬件比较，本报告所有结论均基于**同卡、同环境的相对差异**，不依赖绝对数值。

---

## 2. 方法

所有实验通过**独立分析脚本**完成，不修改 VLA-Adapter 任何源码（采用 monkey-patch 在外部包装模型加载与前向逻辑），保证原模型行为不被污染。

| 实验 | 脚本 | 做法 | 训练 |
|---|---|---|---|
| B1 门控读取 | `read_gating.py` | 加载动作头，读出 24 层 `gating_factor`，计算 `tanh(g)` | 无 |
| B2 注意力占比 | `run_attn_share.py` | patch 前向，统计每层注意力在 [自注意力 / C^AQ / C^R] 三段的占比，并按段长归一化为"每 token 注意力" | 无 |
| B3 条件干预 | `run_gating_eval.py` | 推理期对门控做反事实操作，跑评测看成功率 | 无 |

**关键区分（B3 的严谨性核心）**：门控 `tanh(g)` 乘在 C^R 的**注意力分数（logits）**上，而非直接乘权重。因此存在两类语义截然不同的干预：

- **gate 类**（改门控参数）：`gate_zero_all` 把 `tanh(g)` 设为 0。但 logits=0 经 softmax 后 C^R **仍获得注意力**，故此操作**并非屏蔽 C^R**，只是把门控恢复到 0。
- **mask 类**（改前向 logits）：`mask_all_CR` 把 C^R 的 logits 设为 -1e9，softmax 后权重→0，**才是真正移除 C^R**。

混淆这两者会导致完全错误的结论——这是本项目设计中最关键的一处辨析。

---

## 3. 结果：一条四步递进、最终反转的证据链

### 3.1 第一步（B1）：门控参数全部≈0

读取官方权重的 24 个门控值，`tanh(g)` 全部贴近初始值 0：绝大多数在 10⁻³~10⁻⁵ 量级，最大绝对值仅 0.015（第 21 层）。

> **表象**：门控几乎没被激活——看起来 C^R 被"关闭"了。

### 3.2 第二步（B2 总占比）：C^R 却占据 85% 注意力

统计推理时每层的注意力分配，发现 **C^R 占约 85%**，C^AQ 约 12%，自注意力≈0。

> **看似矛盾**：门控≈0，C^R 反而占绝对主导？
>
> **机制解释**：门控把 C^R 的 logits 压到 0，而 0 在 softmax 中大于其他段常见的负 logits，故 C^R 段反而获得最高权重。门控≈0 不是"关闭"C^R，而是把它**抹平成均匀注意力**。

### 3.3 第三步（B2 每-token）：那 85% 其实是"均匀无重点"的

按段长归一化后发现：第 0–18 层，三类条件的**每-token 注意力几乎完全相等**（≈0.0017）。C^R 占 85% 纯粹因为它有 512 个 token，而非模型重视每个 patch。同时，深层（19–23）的均匀性被打破——C^AQ 的每-token 注意力显著升高（第 23 层达 0.0041），第 21 层自注意力骤增——表明**真正的多模态聚合与动作决策集中在最后几层**，由 C^AQ 与自注意力驱动。

> **看似结论**：C^R 只是均匀的背景，模型并未"思考"它——似乎可有可无。

### 3.4 第四步（B3）：反转——移除 C^R，模型崩溃

| 干预 | 含义 | 成功率 |
|---|---|---|
| baseline | 原模型 | 93% |
| `gate_zero_all` | 门控置 0 | 96% |
| **`mask_all_CR`** | **真正移除 C^R** | **3%** |

- `gate_zero_all`≈baseline → **佐证 B1**：官方门控本就≈0，置 0 不改变任何东西。
- `mask_all_CR` 崩溃到 3%（≈随机动作）→ **反转结论**：那份"均匀、看似无用的 85% C^R 注意力，正是模型工作的命根子。**

### 3.5 补充验证：3% 不是数值 bug，且 C^R 信息冗余分布

单独屏蔽第 0、12、21 层的 C^R，成功率**均保持 ≥93%**。

- **排除 bug**：若 mask 代码有数值问题，屏蔽任一单层都该崩溃；单层无影响证明 mask 机制正常，3% 真实可信。
- **新性质**：C^R 信息在 24 层间**高度冗余、分布式**——任一层缺失可被其他层补偿（鲁棒），唯有全部层都失去 C^R，模型才再无途径获取原始视觉信息，导致彻底失明、崩溃。

---

## 4. Part B 结论：被低估的视觉底座

VLA-Adapter 的门控机制把 C^R 调制成一个**均匀、低调、冗余分布的视觉信息底座**。它在每一个表层指标上都伪装成"无用"：门控参数≈0、每个 patch 权重极小、任意单层可移除而不受影响。然而决定性的反事实实验证明，它是模型不可或缺的根基——一旦全局移除，任务成功率从 93% 崩溃至 3%。

本工作纠正了一个自然却错误的直觉（"门控≈0 ⇒ C^R 无用"），其证据链经历了"表象→矛盾→看似无用→决定性反转→冗余验证"的完整递进。所有结论均为零训练的诊断分析，且为论文从未展示的内容。

### 关键启示
- **不能仅凭参数值或注意力占比判断一个条件的重要性**，必须用真正移除的反事实实验验证因果作用。
- 门控的真实作用与其表面设计意图（"自主选择注入强度"）不同：它实际让 C^R 提供一种**均匀的视觉背景信息**。

---

## 5. Part D：从标量到向量门控的改进

### 5.1 动机（承接 Part B）

Part B 揭示，标量门控让 C^R 退化成"全注意力头一视同仁的均匀底座"。一个自然的疑问随之产生：**每层只有一个标量门控，是否限制了模型对 C^R 的利用？** 在多头注意力中，不同的头本可承担不同功能——若让每个头拥有独立的门控值，模型或许能学会"某些头增强视觉、某些头抑制视觉"的差异化策略，比"一刀切"的标量更灵活。

### 5.2 方法：per-head 向量门控

将 `MLPResNetBlock_Pro` 的门控参数从标量 `zeros(1)` 改为向量 `zeros(num_heads)`（此处 8 个头），前向时 `tanh(g)` 广播到对应的注意力头维度。改动通过开关 `gating_per_head` 控制，默认关闭，**完全向后兼容**（不影响官方权重加载）。

**公平对比的设计**：向量门控版无法继承官方权重（门控形状已变），故不能与官方数字直接比较。本项目从头训练**两个**模型，配置完全相同（LIBERO-Object，batch=16，lr=2e-4，LoRA rank=64，5000 步，相同随机种子），**仅 `gating_per_head` 不同**，从而把性能差异严格归因于门控类型。

### 5.3 结果

| 模型（5000 步） | 成功率（10 trials/task） |
|---|---|
| 标量门控 baseline | 79% |
| **per-head 向量门控** | **100%** |

> 注：当前为减量评测（100 episodes），满量（500 episodes）复核结果待补充，以排除小样本偶然性。

在相同训练预算下，向量门控显著优于标量门控。更关键的是**门控值的机制证据**：读取向量门控学到的 8×24 个值，发现各头门控的"分化程度"（spread = 同层 8 头的最大值减最小值）在**浅层几乎为 0**，但在**深层（19–23）骤然升高**（第 22 层达 0.0505），其中第 22 层出现了部分头强烈正向注入、部分头强烈负向抑制的对比。

这恰好落在 Part B（B2）发现的"真正的动作决策层"上。**即向量门控之所以更优，并非偶然：它让深层决策层的不同注意力头对 C^R 做差异化处理（增强或抑制），这是单个标量门控在物理上无法表达的。**

### 5.4 Part D 结论

诊断（B）→ 改进（D）→ 机制验证 形成完整闭环：标量门控的"均匀"限制被定位 → 提出 per-head 向量门控 → 相同预算下成功率提升（79%→100%）→ 门控分化恰好出现在决策层，给出了改进生效的机制解释。

---

## 6. 局限与后续工作

- 当前结论基于 LIBERO-Object 单一套件。Part B 干预与 Part D 对比均应在 Spatial/Goal/Long 套件、满量（50 trials/task）上复核。
- Part D 的 79% vs 100% 为 100 episodes 结果，需 500 episodes 复测以确认显著性。
- 两个对比模型均仅训练 5000 步（未收敛到论文级别）；向量门控的优势在更长训练下是否保持，值得进一步验证。

---

## 附：复现说明

所有脚本位于 `experiments/analysis/`，源码零修改。核心命令（AutoDL，`MUJOCO_GL=egl`）：

```bash
# B1 读门控
python -m experiments.analysis.read_gating --pretrained_checkpoint outputs/LIBERO-Object-Pro --out out/gating.csv
# B2 注意力占比
python -m experiments.analysis.run_attn_share <eval参数> --share_out out/attn_share.csv
# B3 干预（none / gate_zero_all / mask_all_CR / mask_layer --gating_layer N）
python -m experiments.analysis.run_gating_eval <eval参数> --gating_mode mask_all_CR
```

Part D（向量门控，需训练）：

```bash
# 训练对比：标量 baseline 与 per-head 向量门控，仅 --gating_per_head 不同，其余配置一致
python vla-scripts/finetune.py <训练参数> --gating_per_head False --run_id_note PARTD-scalar
python vla-scripts/finetune.py <训练参数> --gating_per_head True  --run_id_note PARTD-perhead
# 评测（向量版必须加 --gating_per_head True，否则门控形状不匹配）
python experiments/robot/libero/run_libero_eval.py <eval参数> --gating_per_head True \
  --pretrained_checkpoint outputs/...PARTD-perhead--5000_chkpt
# 读门控分化 + 画标量vs向量对比图
python -m experiments.analysis.read_gating --pretrained_checkpoint outputs/...PARTD-perhead--5000_chkpt --gating_per_head --out out/gating_perhead.csv
python -m experiments.analysis.plot_partd_gating --scalar_csv out/gating_scalar.csv --perhead_csv out/gating_perhead.csv --out out/fig_D_gating_compare.png
```
