# 从标量到向量：VLA-Adapter 桥接门控机制的诊断与细粒度改进

**日期**: 2026-06-07（修订版，取代原 bridge-attention-analysis-design）
**作者**: leserein
**定位**: 保研用研究型工程项目（仿真 + 单卡 4090，先零训练诊断，按结果决定是否短训练）

---

## 1. 背景与动机

VLA-Adapter（arXiv 2509.09372）核心模块 Bridge Attention 从 VLM 每层取两种条件注入轻量 Policy，并用**一个可学习标量门控** `tanh(g)`（每层一个标量，共 24 个）控制 Raw 特征 C^R 的注入强度。论文 Table 8 已验证"门控加在 Raw 上"是最优配置（95.0），但：

- **论文从未可视化过这 24 个门控值学成了什么样**；
- **从未做过推理期逐层 counterfactual 干预**（Table 8 是训练期换配置，不是推理期屏蔽）；
- **从未探索门控的"粒度"**——始终是每层一个标量，从没试过向量/分头门控。

本项目补这三个空白：**先打开黑盒看门控（诊断），再据诊断结果决定是否做细粒度改进（开方）**。

## 2. 关键代码事实（已逐行核对，务必准确）

- 门控：`self.gating_factor = nn.Parameter(torch.zeros(1))`，每个 block 一个标量，24 个。
- **变量名有误导性**：Pro 版 `MLPResNetBlock_Pro.forward` 中，`h_t`（名为 task）实际承载 **Raw 图像特征 C^R**；`h_a`（名为 adapter）+proprio 承载 **C^AQ**。
- 数据流：finetune.py 构造 `cat(task_latten_states[=图像patch=C^R], actions_hidden_states[=ActionQuery=C^AQ])`；`predict_action` 切分前段→h_t、后段→h_a。
- `ratio_g = tanh(g)` 乘在 `k_task`（=h_t=**C^R**）上（`action_heads.py:391`）。
- **结论：门控调制的是 C^R，与论文 Table 8 最优配置一致。代码与论文无矛盾**（早前怀疑的矛盾是变量名误读，已排除，不作为发现）。
- 加载链：`run_libero_eval.py` → `get_action_head`（`openvla_utils.py:487`）→ `get_action` → `predict_action`。钩子打在加载后的 `action_head.model.mlp_resnet_blocks[i]`。

## 3. 已有的真实证据（项目可直接引用）

用户已实测：官方 LIBERO-Object-Pro ckpt 在 **RTX 4090 上 95.2%**（476/500），论文 H100 上 99.6%。环境与官方完全一致（torch 2.2.0 / flash-attn 2.5.5 / 官方原版命令）。4.4% 差距是已知的**硬件依赖现象**（README line 559 明确警告）。→ 支撑项目方法论原则：**只比同卡同环境的相对差异，不比跨硬件绝对值**。

## 4. 目标与非目标

**目标**
- 可视化 24 层学到的 `tanh(g)`，刻画"模型学到的逐层注入模式"。
- 推理期逐层干预（置零/拉满门控、屏蔽 C^R/C^AQ 分支），量化每层条件的真实贡献，验证论文 Key Findings。
- 据诊断结果判断标量门控是否有可改进的局限；若有，提出并短训练验证向量/分头门控。

**非目标**
- 不刷绝对成功率（饱和 + 硬件依赖，无意义）。
- 不重复论文 Section 4.5 已做的消融（AQ 数量、condition type、标量门控 on/off）。
- 不做真机 / 多卡 / 全套件训练。

## 5. 方案：Part B 先行（零训练），Part D 条件性后续

### Part B — 门控诊断（零训练，先做，建立动机）

**B1 门控可视化**
- 加载官方 Object Pro ckpt，读取 24 个 `gating_factor`，算 `tanh(g)`，画逐层曲线/热图。
- 产出图 B1 + 解读：哪些层注入强、哪些层几乎关闭（g≈0）。

**B2 attention 条件占比**
- Object 推理（减量 episode，如每任务 10-20）时用 forward hook 捕获各 block 的 `attn_weights`，统计每层分给 {自注意力 tokens, C^R(被门控), C^AQ} 的平均权重占比。
- 产出图 B2：逐层条件占比堆叠图。

**B3 推理期逐层干预（核心，论文未做）**
- 包装/patch `action_head`，在推理时对单层做反事实：(i) 该层门控强制置 0；(ii) 强制拉满。扫描全部 24 层，看 Object 成功率敏感度。
- 整体屏蔽 C^R 分支 vs 屏蔽 C^AQ 分支，对比掉幅。
- 产出图 B3（逐层敏感度）+ 表 B1（屏蔽 C^R vs C^AQ）+ 与论文 Key Findings 对照。

**Part B 出口判断**：若 B 显示标量门控存在明显局限（如多层被迫共享一个标量、某些层门控该开却没开），→ 触发 Part D；否则诚实报告"标量门控已足够"，项目以诊断结论收尾（仍是完整交付）。

### Part D — 向量/分头门控改进（条件性，2-4 次 Object 短训练）

- 把每层标量 `gating_factor` 改成向量（per-channel）或 per-head 门控，让每层每头独立控制 C^R 注入强度。
- Object 短训练对比：baseline 官方配置 vs 改造版，**同卡同种子同配置，只改门控**，`max_steps` 受控（如 20k-50k）。
- 产出表 D1（成功率 + 参数量/速度）+ 分析。负结果（无提升）照样交付，分析原因。

## 6. 交付物

1. 分析代码（读门控 / attention hook / 推理期干预 / 画图），建议放新目录 `experiments/analysis/`，不污染原仓库。
2. 图表 B1/B2/B3 + 表 B1（+ 若做 D：表 D1）。
3. 中文项目报告：动机 → 方法 → 诊断发现 → （改进）→ 结论，含 95.2% 硬件现象作为方法论铺垫。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 推理期干预改错地方导致全崩 | 先验证"无干预时复现 95.2%"，再逐步加干预 |
| 评测耗时（500 ep 数小时） | B/D 用减量 episode；仅最终汇报跑满 |
| B 诊断不出明显局限 | 那 D 不做，诚实报告"标量门控足够"——这是有效负结论 |
| D 改进无提升 | 负结果交付，分析为何无效，面试可讲 |
| 硬件波动 ±数% | 所有对比同卡同配置，只看相对差异 |

## 8. 执行顺序

B1 → B2 → B3（全零训练，依次出图）→ 据 B 出口判断决定是否做 D → （D）→ 写报告。先做 B，用证据驱动是否做 D 的决策。
