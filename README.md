# VLA-Adapter 桥接门控机制的诊断与改进

> VLA-Adapter 的可学习门控把原始视觉特征 C^R 调制成一个"均匀、低调、冗余分布"的信息底座——它在所有表层指标上都像是被关闭、被忽略的无用信息，但实验证明它是模型不可或缺的根基，全局移除会使任务成功率从 93% 崩溃到 3%。基于这一诊断，本工作进一步将标量门控改为 per-head 向量门控，在相同训练预算下把成功率从 79% 提升到 100%，且门控分化恰好出现在诊断所定位的深层决策层。

---

## 1. 研究背景与动机

### 1.1 VLA-Adapter 与 Bridge Attention

VLA-Adapter（arXiv 2509.09372）是一个轻量级视觉-语言-动作模型：用 0.5B 的 Qwen2.5 骨干、无机器人预训练，在 LIBERO 仿真基准上追平 7B 大模型。其核心模块 **Bridge Attention** 从视觉语言模型（VLM）的每一层抽取两种条件，注入一个轻量动作策略网络（Policy，24 层）：

- **C^R（Raw 特征）**：图像 patch 的隐藏状态（512 个 token）；
- **C^AQ（ActionQuery 特征）**：可学习查询 token（64 个）。

其核心设计是一个**可学习标量门控** `tanh(g)`（每层一个，共 24 个），论文称其"让模型自主决定每层注入多少 Raw 特征"。

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

## 3. 实验一（诊断）：一条四步递进、最终反转的证据链

### 3.1 第一步（B1）：门控参数全部≈0

读取官方权重的 24 个门控值，`tanh(g)` 全部贴近初始值 0：绝大多数在 10⁻³~10⁻⁵ 量级，最大绝对值仅 0.015（第 21 层）。

![图 B1：官方 LIBERO-Object-Pro 的 24 层门控值，全部接近 0](../experiments/analysis/out/fig_B1_gating.png)

> **表象**：门控几乎没被激活——看起来 C^R 被"关闭"了。

### 3.2 第二步（B2 总占比）：C^R 却占据 85% 注意力

统计推理时每层的注意力分配，发现 **C^R 占约 85%**，C^AQ 约 12%，自注意力≈0。

> **看似矛盾**：门控≈0，C^R 反而占绝对主导？
>
> **机制解释**：门控把 C^R 的 logits 压到 0，而 0 在 softmax 中大于其他段常见的负 logits，故 C^R 段反而获得最高权重。门控≈0 不是"关闭"C^R，而是把它**抹平成均匀注意力**。

### 3.3 第三步（B2 每-token）：那 85% 其实是"均匀无重点"的

按段长归一化后发现：第 0–18 层，三类条件的**每-token 注意力几乎完全相等**（≈0.0017）。C^R 占 85% 纯粹因为它有 512 个 token，而非模型重视每个 patch。同时，深层（19–23）的均匀性被打破——C^AQ 的每-token 注意力显著升高（第 23 层达 0.0041），第 21 层自注意力骤增——表明**真正的多模态聚合与动作决策集中在最后几层**，由 C^AQ 与自注意力驱动。

![图 B2：上为各条件总注意力占比（受段长影响，C^R 因 token 多而占 85%），下为每-token 归一化注意力（浅层均匀，深层 19–23 由 C^AQ 与自注意力主导）](../experiments/analysis/out/fig_B2_attn_share.png)

> **看似结论**：C^R 只是均匀的背景，模型并未"思考"它——似乎可有可无。

### 3.4 第四步（B3）：反转——移除 C^R，模型崩溃

| 干预 | 含义 | 成功率 |
|---|---|---|
| baseline | 原模型 | 93% |
| `gate_zero_all` | 门控置 0 | 96% |
| **`mask_all_CR`** | **真正移除 C^R** | **3%** |

- `gate_zero_all`≈baseline → **佐证 B1**：官方门控本就≈0，置 0 不改变任何东西。
- `mask_all_CR` 崩溃到 3%（≈随机动作）→ **反转结论**：那份"均匀、看似无用"的 85% C^R 注意力，正是模型工作的命根子。

### 3.5 补充验证：3% 不是数值 bug，且 C^R 信息冗余分布

单独屏蔽第 0、12、21 层的 C^R，成功率**均保持 ≥93%**。

- **排除 bug**：若 mask 代码有数值问题，屏蔽任一单层都该崩溃；单层无影响证明 mask 机制正常，3% 真实可信。
- **新性质**：C^R 信息在 24 层间**高度冗余、分布式**——任一层缺失都可被其他层补偿（鲁棒）；唯有当全部层都失去 C^R 时，模型才再无任何途径获取原始视觉信息，导致动作生成彻底失效。

---

## 4. 实验一 结论：被低估的视觉底座

VLA-Adapter 的门控机制把 C^R 调制成一个**均匀、低调、冗余分布的视觉信息底座**。它在每一个表层指标上都伪装成"无用"：门控参数≈0、每个 patch 权重极小、任意单层可移除而不受影响。然而决定性的反事实实验证明，它是模型不可或缺的根基——一旦全局移除，任务成功率从 93% 崩溃至 3%。

本工作纠正了一个自然却错误的直觉（"门控≈0 ⇒ C^R 无用"），其证据链经历了"表象→矛盾→看似无用→决定性反转→冗余验证"的完整递进。所有结论均为零训练的诊断分析，且为论文从未展示的内容。

### 关键启示
- **不能仅凭参数值或注意力占比判断一个条件的重要性**，必须用真正移除的反事实实验验证因果作用。
- 门控的真实作用与其表面设计意图（"自主选择注入强度"）不同：它实际让 C^R 提供一种**均匀的视觉背景信息**。

---

## 5. 实验二：从标量到向量门控的改进

### 5.1 动机（承接实验一）

实验一揭示，标量门控让 C^R 退化成"全注意力头一视同仁的均匀底座"。一个自然的疑问随之产生：**每层只有一个标量门控，是否限制了模型对 C^R 的利用？** 在多头注意力中，不同的头本可承担不同功能——若让每个头拥有独立的门控值，模型或许能学会"某些头增强视觉、某些头抑制视觉"的差异化策略，比"一刀切"的标量更灵活。

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

![图 D：标量 vs per-head 向量门控。上为逐层分化程度对比（向量门控仅在深层 19–23 学出显著 spread），下为向量门控 24 层×8 头的门控值热图（深层各头红蓝分化，浅层一致接近 0）](../experiments/analysis/out/fig_D_gating_compare.png)

这恰好落在实验一（步骤3）发现的"真正的动作决策层"上。**即向量门控之所以更优，并非偶然：它让深层决策层的不同注意力头对 C^R 做差异化处理（增强或抑制），这是单个标量门控在物理上无法表达的。**

### 5.4 实验二 结论

诊断（实验一）→ 改进（实验二）→ 机制验证 形成完整闭环：标量门控的"均匀"限制被定位 → 提出 per-head 向量门控 → 相同预算下成功率提升（79%→100%）→ 门控分化恰好出现在决策层，给出了改进生效的机制解释。

---

## 6. 局限与后续工作

- 当前结论基于 LIBERO-Object 单一套件。实验一干预与实验二对比均应在 Spatial/Goal/Long 套件、满量（50 trials/task）上复核。
- 实验二的 79% vs 100% 为 100 episodes 结果，需 500 episodes 复测以确认显著性。
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

实验二（向量门控，需训练）：

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


<div align="center">
  <img src="figure/LOGO2.png" width="70%" style="vertical-align:-7px;" />


[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2509.09372) [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/VLA-Adapter) [![Twitter](https://img.shields.io/badge/AK-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/_akhaliq/status/1966610780838621241) [![WeChat](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/OpenHelix-Team/VLA-Adapter/issues/1)

</div>

### The official implementation of **VLA-Adapter**.
<br/>

<div id="top" align="center">
<p align="center">
<img src=figure/Framework.png width=90% />
</p>
</div>

> **📝 Paper: https://arxiv.org/abs/2509.09372**<br/>
> **🌍 Project page: https://vla-adapter.github.io/**<br/>
> **🤗 HuggingFace: https://huggingface.co/VLA-Adapter**<br/>
> **Github: https://github.com/OpenHelix-Team/VLA-Adapter**

<br/>

## :loudspeaker: News!
- **[2026/03/16]** We added **real-world ALOHA deployment** support, verified on [Cobot Magic](https://global.agilex.ai/products/cobot-magic). See [`experiments/robot/aloha/`](experiments/robot/aloha/) for details.
- **[2025/09/22]** We released our codes! An enhanced **Pro** version is also released (this version conforms to the pipeline in the original paper, but is optimized in implementation). Everyone is welcome to use it!🎉
- **[2025/09/13]** Our paper won the 🥇**first place** in the [daily list](https://huggingface.co/papers/date/2025-09-12), the 🥈**second place** in the [weekly list](https://huggingface.co/papers/week/2025-W37), and 🥉**third place** in the [Monthly list](https://huggingface.co/papers/month/2025-09) in HF! ⭐
- **[2025/09/13]** Our paper listed in the [Trending Paper](https://huggingface.co/papers/trending) in HF! ⭐
- **[2025/09/12]** We released the original version of the VLA-Adapter for four LIBERO models on [HuggingFace](https://huggingface.co/VLA-Adapter).
- **[2025/09/11]** We released our paper on [ArXiv](https://arxiv.org/abs/2509.09372).

<br/>

## :black_nib: TODO List<a name="todo"></a>

- [x]  Release **checkpoints** for reproduction.
- [x]  Release [VLA-Adapter v2 paper](https://arxiv.org/abs/2509.09372).
- [ ]  A more **powerful version**, **VLA-Adapter++**, and a detailed **technical report** 📝 will be released soon.<br/>
- [x]  **ALOHA real-world deployment** on [Cobot Magic](https://global.agilex.ai/products/cobot-magic) — training, server-client inference, and evaluation ([details](experiments/robot/aloha/)).<br/>
- [ ]  Continue to update the code to adapt to various **real-world systems** deployments, including the configuration of our paper, Franka, UR-5, and AGILE Piper.<br/>
- [ ]  It will soon be compatible with **various foundation models**, including but not limited to [VPP](https://arxiv.org/abs/2412.14803), [π0.5](https://arxiv.org/abs/2504.16054).<br/>
- [ ]  We will update the **diffusion transformers** and **flow matching** policy networks in the future, and the results will be updated in the subsequent VLA-Adapter++ technical report.
- [ ]  We will also update and give more experiments on **Frozen backbone**.
- [ ]  We will expand its **generalization** further in the future. Work is in progress! So please stay tuned!
- [ ]  **RL post-training** is also in progress. Interested researchers are welcome to join us in building this foundation!
- [ ]  **The dual-system compatibility** of VLA-Adapter is under exploration!


<br/>

## 🌟 Table of Contents

- [:rocket: Quick Start](#rocket-quick-start) 
  - [Conda Environment of VLA-Adapter](#conda-environment-of-vla-adapter)
  - [Install Dependencies](#install-dependencies)
- [:pencil: Data Preparation](#pencil-data-preparation) 
  - [LIBERO Benchmark](#libero-benchmark)
  - [CALVIN Benchmark](#calvin-benchmark)
  - [:video_game: Our Dependencies](#video_game-our-dependencies)
  - [:pushpin: Benchmark Location](#pushpin-benchmark-location)
- [⚓ VLM backbone](#vlm)
- [:fire: Training for Different Configurations](#fire-training-for-different-configurations) &emsp; => Provides **training configurations** for GPUs ranging from **10GB** to **80GB** of VRAM.
  - [:books: Related File for Training](#books-related-file-for-training)
  - [:ledger: How to Train on Extremely Limited VRAM GPUs](#ledger-how-to-train-on-extremely-limited-vram-gpus) &emsp; => A card with 10GB-12GB *(e.g. NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, and 5070)*
  - [:ledger: How to Train on Low VRAM GPUs](#ledger-how-to-train-on-low-vram-gpus) &emsp; => A card with 24GB *(e.g. NVIDIA GeForce RTX 3090 and 4090)*
  - [:ledger: How to Train on Larger VRAM GPUs](#ledger-how-to-train-on-larger-vram-gpus) &emsp; => A Consumer GPU with 32GB *(e.g. NVIDIA GeForce RTX 5090)* &emsp; A Professional-Grade GPU with 40GB-48GB *(e.g. NVIDIA A100-40GB, A800-40GB, L20, and RTX A6000).*
  - [:ledger: How to Train on Sufficient VRAM GPUs](#ledger-how-to-train-on-sufficient-vram-gpus) &emsp; => Professional-Grade GPUs with ≥80GB *(e.g. NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, and GB200).*
- [:mechanical_arm: Inference](#mechanical_arm-inference)
  - [:books: Related File for Inference](#books-related-file-for-inference)
  - [🤗 Checkpoint of VLA-Adapter](#ckpts)
  - [:notebook: How to Eval](#evals)
- [🌈 Success Rate Comparison](#results)
- [📝 Citation](#cite)
- [:heart: Acknowledgment](#heart-acknowledgment)

<br/>

## :rocket: Quick Start


### Conda Environment of VLA-Adapter

```bash
# Create and activate conda environment
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
```

### Install Dependencies

```bash
# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone vla-adapter repo and pip install to download dependencies
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation
# If you run into difficulty, try `pip cache remove flash_attn` first, or visit the
# website to download it. (https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5)
# You can download the corresponding `.whl` file according to the cuda version of `nvidia-smi`,
# and then run `pip install flash_attn-2.5.5+cuXX...whl` to install it. 
# We use the `flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` file.
```

<br/>
<br/>


## :pencil: Data Preparation

### LIBERO Benchmark

- **(Optional)**

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From vla-adapter base dir
```

To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below. This will download the `Spatial`, `Object`, `Goal`, and `Long` datasets in `RLDS` format, i.e., `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out). These datasets require `~10GB` of memory in total. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup). You can use these to fine-tune Prismatic-VLMs (built on Qwen2.5-0.5B) or other VLMs.

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

🌟 Attention! The dataset downloaded in this way needs to remove of the ``modified_`` word to adapt to the path of - [:pushpin: Benchmark Location](#pushpin-benchmark-location)!!!

When using LIBERO, you may get an error message like `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. You can use:

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

### CALVIN Benchmark

- **(Optional)**

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT

# Installation of `pyhash` may fail on some machines. If it fails, you can solve it by lowering the `setuptools` version: `pip install setuptools==57.5.0`
sh install.sh
```

To download the [CALVIN ABC→D datasets](https://github.com/mees/calvin/tree/main/dataset) that we used in our fine-tuning experiments, run the command below. 

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

If you want to download the RLDS format, you can visit [here](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds) to download it. This dataset requires `~50GB` of memory.

When using CALVIN, you may get an error message like `AttributeError: 'NoneType' object has no attribute 'eglQueryString'`. You can use:

```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```


### :video_game: Our Dependencies 

- **(including LIBERO and CALVIN)**

At this point, the environment is fully installed. If you want to confirm whether the environment is correct, you can see the `our_envs.txt` file we released.


### :pushpin: Benchmark Location

The downloaded dataset can be placed in the `/data` folder. The overall directory structure is as follows:

```
·
├── data
·   ├── libero
    │   ├── libero_10_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_goal_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │   ├── libero_object_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_spatial_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │
    ├── calvin_abc
    │   └── 1.0.0  (It contains some json files, 512 train tfrecord files, and 32 valid tfrecord files)
    │
    └── other benchmarks ...
```

<br/>
<br/>

## ⚓ VLM backbone <a name="vlm"></a>
We use the `Prismatic-VLMs` architecture. Since the file is large, please download it from [here](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b). Then put it in the `/pretrained_models` folder. The file structure is:

```
·
├── pretrained_models
·   ├── configs
    └── prism-qwen25-extra-dinosiglip-224px-0_5b
```


<br/>
<br/>

## :fire: Training for Different Configurations

**We provide different training configurations for different users. You can choose the configuration suitable for training based on your GPU card type.**

### :books: Related File for Training
* `vla-scripts/finetune.py`: VLA fine-tuning script


### :ledger: How to Train on Extremely Limited VRAM GPUs

***=> Extremely Limited VRAM (A card with 10GB-12GB) (e.g. NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, and 5070).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If your resources are extremely limited, you can set `--batch_size 1` and `--lora_rank 64`, it only requires `9.6GB` of VRAM. Certainly, `batch size = 1` will cause gradient updates to be greatly affected by extreme values, and loss convergence will be unstable. In this case, you can modify the `grad_accumulation_steps` parameter to simulate a similar effect. For example, `--batch_size 1` with `--grad_accumulation_steps 8` has a similar effect to `--batch_size 8`, but the training speed will be slower. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `10GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version`, requiring only `8.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 400000 \
--max_steps 400005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 1 \
--grad_accumulation_steps 8 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

<br/>

### :ledger: How to Train on Low VRAM GPUs

***=> Low VRAM (A card with 24GB) (e.g. NVIDIA GeForce RTX 3090 and 4090).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 4` and `--lora_rank 64`. This only takes nearly `20GB`. This is consistent with the rank in our paper. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `24GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch), requiring only `17.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.


 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 4 \
--grad_accumulation_steps 4 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>

### :ledger: How to Train on Larger VRAM GPUs

***=> A Consumer GPU with 32GB (e.g. NVIDIA GeForce RTX 5090) <br/> => A Professional-Grade GPU with 40GB-48GB (e.g. NVIDIA A100-40GB, A800-40GB, L20, and RTX A6000).***


>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 8` and `--lora_rank 64`. This only takes nearly `29GB`. 

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

With this configuration, you can achieve the same results as in our paper on the `LIBERO-Object` benchmark, achieving a `99.2%` success rate, in just `8 hours`. The `LIBERO-Spatial` benchmark requires approximately 10 hours of training. However, the `LIBERO-Long` benchmark takes longer because its tasks are longer and more difficult, requiring more training steps to achieve superior performance.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>

### :ledger: How to Train on Sufficient VRAM GPUs

***=> Professional-Grade GPUs with ≥80GB (e.g. NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, and GB200).***

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

You can use 1 to 8 GPUs for training by changing the number of `CUDA_VISIBLE_DEVICES` to the GPU number and the number of GPUs after `--nproc-per-node`. In our paper, we use 4×H100 GPU for training. In this configuration, the four suites of the LIBERO benchmark, `Spatial` (only five hours), `Object` (less than one hour), `Goal` (three hours), and `Long` (half a day); the `CALVIN` benchmark (eight hours)

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.


>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 150000 \
--max_steps 150005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 16 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--spatial--$current_time \
> logs/VLA-Adapter--spatial--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

## :mechanical_arm: Inference

### :books: Related File for Inference
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: VLA-specific eval utils
  * `robot_utils.py`: Other eval utils

<br/>

### 🤗 Checkpoint of VLA-Adapter <a name="ckpts"></a>
We fine-tuned `Qwen2.5-0.5B` with our adapter bridge paradigm on four LIBERO task suites independently: `LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, and `LIBERO-Long`. 
The four VLA-Adapter checkpoints for LIBERO are available on Hugging Face:
* [VLA-Adapter/LIBERO-Spatial](https://huggingface.co/VLA-Adapter/LIBERO-Spatial) 
* [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object)
* [VLA-Adapter/LIBERO-Goal](https://huggingface.co/VLA-Adapter/LIBERO-Goal)
* [VLA-Adapter/LIBERO-Long](https://huggingface.co/VLA-Adapter/LIBERO-Long)

In addition, we also provide a `Pro` version, we used `4*H100` GPUs for training, `--batch_size 16`, `--lora rank 64`, and the `--max_steps 100000`. The Pro checkpoints is:

* [VLA-Adapter/LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro) `(97.8 -> 99.6)`
* [VLA-Adapter/LIBERO-Object-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Object-Pro) `(99.2 -> 99.6)`
* [VLA-Adapter/LIBERO-Goal-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Goal-Pro) `(97.2 -> 98.2)`
* [VLA-Adapter/LIBERO-Long-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro) `(95.0 -> 96.4)`
* [VLA-Adapter/CALVIN-ABC-Pro](https://huggingface.co/VLA-Adapter/CALVIN-ABC-Pro) `(4.42 -> 4.50)`

These files need to be placed in the `/output` folder. If you trained your own models, it will also be stored here. The subsequent eval code will call the model in this folder for inference.


<br/>


### :notebook: How to Eval <a name="evals"></a>

**We strongly recommend that you use our open source `Pro` version of the model, which has stronger performance.** To start evaluations with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above. If you want to use the original version of the model, you only need to adjust the `-- use_pro_version` parameter to `False` and pass the original version of the model to the `--pretrained_checkpoint` parameter. Finally, the inference results will be displayed in the `/eval_logs` folder, and the inference video will be displayed in the `/rollouts/vla-adapter` folder. 


```bash
# Launch LIBERO-Spatial-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial--chkpt.log 2>&1 &


# Launch LIBERO-Object-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object \
  --use_pro_version True \
  > eval_logs/Object--chkpt.log 2>&1 &


# Launch LIBERO-Goal-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Goal-Pro \
  --task_suite_name libero_goal \
  --use_pro_version True \
  > eval_logs/Goal--chkpt.log 2>&1 &


# Launch LIBERO-Long-Pro (LIBERO-10) evals (Background running)
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-long-Pro \
  --task_suite_name libero_10 \
  --use_pro_version True \
  > eval_logs/Long--chkpt.log 2>&1 &


# Launch CALVIN ABC→D-Pro evals (Background running)
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN-ABC-Pro \
  > eval_logs/CALVIN--ABC.log 2>&1 &
```

If you want to get the inference **throughput**, you can run it in the `run_libero_eval.py` file. You can add  `start = time.time()` and `end = time.time()` before and after `lines 334--345` and calculate the difference between the two. This difference is the time it takes to generate `8 chunks`. This gives you the inference throughput. We measured it multiple times and took the average value of `0.036s`.

<br/>

## 🌈 Success Rate Comparison <a name="results"></a>

All our results are inferred on `H100`. You can find the inference `log` file in the model released on [HF](https://huggingface.co/VLA-Adapter) for viewing. The evaluation script will run 500 trials by default (10 tasks x 50 episodes each) in LIBERO and 1,000 task sequences in CALVIN. Use the same card for training and inference whenever possible. **Note that results may vary slightly if you use a different GPU than the H100.** This phenomenon is also mentioned in the OpenVLA-OFT readme file.

### Performance on LIBERO benchmark. 

<b><i>XX</i></b> represents the best performance, <b>XX</b> represents the second best performance, and <i><u>XX*</u></i> represents the third best performance.
<table>
  <tr>
   <td><strong>LIBERO</strong></td>  <td><strong>Methods</strong></td>
   <td><strong>Scale</strong></td>  <td><strong>Spatial</strong></td>
   <td><strong>Object</strong></td>  <td><strong>Goal</strong></td>
   <td><strong>Long</strong></td>  <td><strong>Avg.</strong></td>
  </tr>

  <tr><td rowspan="10">Large-scale</td><td>FlowVLA (Zhong et al., 2025)</td>
   <td>8.5B</td><td>93.2</td><td>95.0</td><td>91.6</td><td>72.6</td><td>88.1</td></tr>

  <tr><td>UnifiedVLA (Wang et al., 2025)</td>
   <td>8.5B</td><td>95.4</td><td><i><u>98.8*</u></i></td><td> 93.6 </td><td>94.0 </td><td>95.5</td></tr>

  <tr><td>OpenVLA (Kim et al., 2024)</td>
   <td>7B</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td>
   <td>7B</td><td><i><u>97.6*</u></i></td><td>98.4</td><td><b>97.9</b></td><td><i><u>94.5*</u></i></td><td><i><u>97.1*</u></i></td></tr>

  <tr><td>UniVLA (Bu et al., 2025)</td>
   <td>7B</td><td>96.5</td><td> 96.8</td><td> 95.6 </td><td>92.0 </td><td>95.2</td></tr>

  <tr><td>CoT-VLA (Zhao et al., 2025)</td>
   <td>7B</td><td>87.5 </td><td>91.6 </td><td>87.6</td><td> 69.0</td><td> 81.1</td></tr>

  <tr><td>WorldVLA (Cen et al., 2025)</td>
   <td>7B</td><td>87.6</td><td> 96.2</td><td> 83.4</td><td> 60.0</td><td> 81.8</td></tr>

  <tr><td>TraceVLA (Zheng et al., 2025)</td>
   <td>7B</td><td>84.6</td><td> 85.2</td><td> 75.1</td><td> 54.1</td><td> 74.8</td></tr>

  <tr><td>MolmoAct (Lee et al., 2025)</td>
   <td>7B</td><td>87.0</td><td> 95.4 </td><td>87.6</td><td> 77.2 </td><td>86.6</td></tr>

  <tr><td>ThinkAct (Huang et al., 2025)</td>
   <td>7B</td><td>88.3 </td><td>91.4</td><td> 87.1</td><td> 70.9</td><td> 84.4</td></tr>

  <tr><td rowspan="7">Small-scale</td><td>4D-VLA (Zhang et al., 2025)</td>
   <td>4B</td><td>88.9</td><td> 95.2</td><td> 90.9</td><td> 79.1 </td><td>88.6</td></tr>

  <tr><td>SpatialVLA (Qu et al., 2025)</td>
   <td>4B</td><td>88.2</td><td> 89.9</td><td> 78.6</td><td> 55.5 </td><td>78.1</td></tr>

  <tr><td>π0 (Black et al., 2024)</td>
   <td>3B</td><td>96.8</td><td><i><u>98.8*</u></i></td><td>95.8</td><td> 85.2</td><td> 94.2</td></tr>

  <tr><td>π0-FAST (Pertsch et al., 2025)</td>
   <td>3B</td><td>96.4</td><td> 96.8 </td><td>88.6</td><td> 60.2</td><td> 85.5</td></tr>

  <tr><td>NORA (Hung et al., 2025)</td>
   <td>3B</td><td>92.2 </td><td>95.4 </td><td>89.4</td><td> 74.6 </td><td>87.9</td></tr>

  <tr><td>SmolVLA (Shukor et al., 2025)</td>
   <td>2.2B</td><td>93.0</td><td> 94.0 </td><td>91.0</td><td> 77.0 </td><td>88.8</td></tr>

  <tr><td>GR00T N1 (NVIDIA et al., 2025)</td>
   <td>2B</td><td>94.4</td><td> 97.6 </td><td>93.0 </td><td>90.6</td><td> 93.9</td></tr>

  <tr><td rowspan="5">Tiny-scale</td><td>Seer (Tian et al., 2025)</td>
   <td>0.57B</td><td>-</td><td> - </td><td>- </td><td>78.7</td><td> 78.7</td></tr>

  <tr><td>VLA-OS (Gao et al., 2025)</td>
   <td>0.5B</td><td>87.0 </td><td>96.5</td><td> 92.7 </td><td>66.0</td><td> 85.6</td></tr>

  <tr><td>Diffusion Policy (Chi et al., 2023)</td>
   <td>-</td><td>78.3</td><td> 92.5</td><td> 68.3 </td><td>50.5 </td><td>72.4</td></tr>

  <tr><td><b>VLA-Adapter (Ours)</b></td>
   <td><b>0.5B</b></td><td><b>97.8</b></td><td><b>99.2</b></td><td><i><u>97.2*</u></i></td><td> <b>95.0 </b></td><td><b>97.3</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (Ours)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.6</i></b></td><td><b><i>99.6</i></b> </td><td><b><i>98.2</i></b></td><td><b><i>96.4</i></b></td><td><b><i>98.5</i></b></td></tr>
  
</table>

### Performance on CALVIN ABC→D benchmark. 

<b><i>XX</i></b> represents the best performance, <b>XX</b> represents the second best performance, and <i><u>XX*</u></i> represents the third best performance.

<table>
  <tr>
   <td><strong>CALVIN</strong></td>  <td><strong>Methods</strong></td>
   <td><strong>Scale</strong></td>  <td><strong>1</strong></td>
   <td><strong>2</strong></td>  <td><strong>3</strong></td>
   <td><strong>4</strong></td>  <td><strong>5</strong></td> <td><strong>Avg. len</strong></td>
  </tr>

  <tr><td rowspan="8">Large-scale</td><td>UniVLA (Bu et al., 2025) </td><td>7B </td><td>95.5 </td><td>85.8 </td><td>75.4</td><td> 66.9 </td><td>56.5 </td><td>3.80</tr>

  <tr><td>OpenVLA (Kim et al., 2024) </td><td> 7B</td><td> 91.3</td><td> 77.8 </td><td>62.0 </td><td>52.1 </td><td>43.5</td><td> 3.27</td></tr>

  <tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td> 7B</td><td> 96.3</td><td> 89.1 </td><td>82.4</td><td> 75.8</td><td> 66.5</td><td> 4.10</td></tr>

  <tr><td>VLAS (Zhao et al., 2025b) </td><td> 7B</td><td> 87.2 </td><td>64.2</td><td> 40.9 </td><td>28.1</td><td> 19.6 </td><td>2.40</td></tr>

  <tr><td>LCB (Shentu et al., 2024) </td><td> 7B</td><td> 73.6 </td><td>50.2 </td><td>28.5 </td><td>16.0 </td><td>9.9 </td><td>1.78</td></tr>

  <tr><td>RoboDual (Bu et al., 2024a) </td><td> 7B</td><td> 94.4</td><td> 82.7</td><td> 72.1</td><td> 62.4 </td><td>54.4</td><td> 3.66</td></tr>

  <tr><td>OpenHelix (Cui et al., 2025)  </td><td> 7B</td><td> <i><u>97.1*</u></i> </td><td>91.4 </td><td>82.8</td><td> 72.6</td><td> 64.1 </td><td>4.08</td></tr>

  <tr><td>ReconVLA (Song et al., 2025c)  </td><td> 7B</td><td> 95.6 </td><td>87.6 </td><td>76.9</td><td> 69.3</td><td> 64.1 </td><td>3.95</td></tr>

  <tr><td rowspan="4">Small-scale</td><td>DeeR (Yue et al., 2024) </td><td> 3B</td><td> 86.2</td><td> 70.1 </td><td>51.8</td><td> 41.5</td><td> 30.4 </td><td>2.82</td></tr>

  <tr><td>RoboFlamingo (Li et al., 2024b) </td><td> 3B</td><td> 82.4 </td><td>61.9</td><td> 46.6 </td><td>33.1</td><td> 23.5</td><td> 2.48</td></tr>

  <tr><td>VPP (Hu et al., 2025)</td><td>  1.5B</td><td>  95.7</td><td>  91.2</td><td>  <i><u>86.3*</u></i></td><td>  <i><u>81.0*</u></i></td><td>  <i><u>75.0*</u></i></td><td>  <i><u>4.33*</u></i></td></tr>

  <tr><td>SuSIE (Black et al., 2024)</td><td>1.3B</td><td> 87.0</td><td> 69.0</td><td> 49.0 </td><td>38.0</td><td> 26.0</td><td> 2.69</td></tr>

  <tr><td rowspan="5">Tiny-scale</td><td>Seer-Large (Tian et al., 2025)</td><td>0.57B</td><td> 96.3 </td><td><i><u>91.6*</u></i></td><td> 86.1 </td><td>80.3 </td><td>74.0</td><td> 4.28</td></tr>

  <tr><td>MoDE (Reuss et al., 2025) </td><td> 0.44B </td><td>96.2</td><td> 88.9</td><td> 81.1</td><td> 71.8 </td><td>63.5 </td><td>4.01</td></tr>

  <tr><td>Seer (Tian et al., 2025) </td><td> 0.32B</td><td> 94.4 </td><td>87.2 </td><td>79.9 </td><td>72.2 </td><td>64.3</td><td> 3.98</td></tr>

  <tr><td><b>VLA-Adapter (Ours)</b></td>
   <td><b>0.5B</b></td><td><b><i>99.1</i></b> </td><td><b>94.6</b> </td><td><b>88.8</b></td><td> <b>82.8</b> </td><td><b>76.5</b> </td><td><b>4.42</b></td></tr>

  <tr><td><b>VLA-Adapter-Pro (Ours)</b></td>
   <td><b>0.5B</b></td><td><b>98.5</b></td><td><b><i>95.0</i></b> </td><td><b><i>90.5</i></b></td><td><b><i>85.3</i></b></td><td><b><i>80.0</i></b></td><td><b><i>4.50</i></b></td></tr>
  
</table>


<br/>


## 📝 Citation <a name="cite"></a>

### 🫶 If you feel that this paper, models, or codes are helpful, please cite our paper, thanks for your support of VLA-Adapter!

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```

## :heart: Acknowledgment

We thank [OpenVLA-OFT](https://github.com/moojink/openvla-oft), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), and [RoboDual](https://github.com/OpenDriveLab/RoboDual) for their open-sourced work!

## 🌟 Star History

<a href="https://www.star-history.com/#OpenHelix-Team/VLA-Adapter&Date">
  <img src="https://api.star-history.com/svg?repos=OpenHelix-Team/VLA-Adapter&type=Date" width="400" height="250" />
</a>

