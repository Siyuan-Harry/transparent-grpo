# Transparent GRPO

## [English Version](./README.md) | [Chinese Version](./README-cn.md)

**Group Relative Policy Optimization (GRPO) 的极简、单文件实现。**

目前的 RLHF/RL 库（如 TRL, OpenRLHF）虽然功能强大，但往往将数学逻辑隐藏在层层抽象、回调函数和复杂的类继承之下。**Transparent GRPO** 专为希望实现以下目标的视觉研究者和工程师设计：

1. **一气呵成地阅读：** 无需为了寻找 `loss` 函数的定义位置在 10 个文件之间反复横跳。
2. **更清晰的算法细节：** 清晰地看到 Advantages 是如何归一化的，以及 KL 惩罚（penalty）是如何应用的。
3. **快速原型设计：** 只需修改一行代码即可调整核心算法（例如更改 KL 估计器 estimator）。

### ✨ 核心特性

* **代码量 < 400 行：** 整个逻辑（环境、数据集、模型、训练循环）都集成在一个文件内。
* **线性逻辑流：** 代码从上到下顺序阅读：`Generate` -> `Reward` -> `Advantage` -> `Update`。
* **前沿算法实现：** 实现了 *K1 Estimator in Rewards*（无偏 KL 近似），而非旧的 K3-in-Loss 方法。这与最近的研究发现 (*[Shah et al., 2026](https://arxiv.org/abs/2512.21852)*) 一致，在推理任务中具有更低的方差和更好的稳定性。
* **极简轻量：** 仅依赖 `torch`, `transformers`, `accelerate` 和 `numpy`。无需繁重的 RL 框架。
* **开箱即运行：** 脚本内置了一个玩具数学环境（分段函数连续性问题），确保能在单显卡上快速训练，让你在几分钟内验证实现的正确性 ie 看到 llm 答案的质量在5个 epochs 内快速攀升。

## 仓库结构说明

* `transparent_grpo.py`: 包含完整 GRPO 实现的主脚本。建议从头到尾阅读以理解流程。
* `logs/train_grpo_20260209_172905.log`: 示例训练日志，展示了 reward 和 loss 的变化过程。

无需担心其他内容。没有隐藏文件，没有复杂的目录结构。只有一个文件供你阅读和理解。

## 🚀 快速开始

### 0. 硬件要求

使用 `Qwen2.5-3B-Instruct`（代码内置）、`group_size=4`、`max_new_tokens=512` 以及 `DeepSpeed ZeRO-2` 时的预估显存（VRAM）占用：

| GPU 配置 | 每张显卡预估显存 | 状态 | 备注 |
| --- | --- | --- | --- |
| **1x A100 (80GB)** | ~35 GB | ✅ 轻松 | 非常适合单卡调试。 |
| **2x RTX 3090/4090 (24GB)** | ~22 GB | ⚠️ 紧凑 | 需要 ZeRO-2 来分片优化器状态。 |
| **4x L4 (24GB)** | ~16 GB | ✅ 安全 | 优化器状态分布均匀。 |
| **1x RTX 4090 (24GB)** | **OOM** | ❌ 失败 | 除非开启 ZeRO-Offload (CPU)，否则会 OOM。 |

### 1. 安装

你只需要标准的 PyTorch 生态库：

```bash
pip install torch transformers accelerate deepspeed numpy
```

### 2. 运行训练

脚本配置为直接配合 Hugging Face Accelerate 和 DeepSpeed 运行（针对 3B 模型建议使用 ZeRO-2）。

```bash
# 如果还没配置过 accelerate，请先配置（选择 DeepSpeed/Zero2）
accelerate config

# 运行训练脚本
nohup stdbuf -oL accelerate launch --use_deepspeed --zero_stage 2 transparent_grpo.py > logs/train_grpo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 🧠 理解代码实现

本仓库剥离了所有“魔法”。以下是 `transparent_grpo.py` 中核心 GRPO 机制对应的位置：

| 组件 | 描述 |
| --- | --- |
| **Generation** | 在循环内直接调用标准的 `model.generate()` |
| **Group Sampling** | `Config.group_size = 4`。为每个 prompt 生成 4 个输出。 |
| **Reward Function** | 针对特定代数问题（代码中的 `ToyEnv`）定制的基于正则表达式的奖励系统。 |
| **Ref Model LogProbs** | 计算 $\pi_{\text{ref}}(y \| x)$  |
| **KL Divergence** | 计算逐 token 的 KL: $\log \pi_\theta(y \| x) - \log \pi_{\text{ref}}(y \| x)$ |
| **Advantage** | 计算方式极其简单： `(rewards - mean) / std`。 |
| **Loss Function** | 标准的 PPO 截断（clipping）：$\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$ |

与 [DeepSeekMath](https://arxiv.org/abs/2402.03300)（原始 GRPO 论文）相比，本实现：

1. 使用玩具环境（Toy Environment）代替大型数学数据集，以确保简洁性。
2. 实现了 K1 估计器（KL 注入 Reward），而非 K3（KL 注入 Loss），以获得更好的稳定性和性能（参见 *[Shah et al., 2026](https://arxiv.org/abs/2512.21852)*）。
3. 保留了标志性的 Group Relative Advantage 归一化和 PPO 风格的 Clipping，确保算法行为完全符合 GRPO。

### "K1 in Reward" 逻辑

不同于将 KL 散度放入损失函数的旧版 PPO 实现，本实现遵循了 *[Shah et al., 2026](https://arxiv.org/abs/2512.21852)* 的方法，直接从奖励中减去 KL 惩罚：

```python
# 摘自 transparent_grpo.py 的代码片段
per_token_kl = token_log_probs.detach() - ref_token_log_probs.detach()
kl_penalty = (per_token_kl * loss_mask).sum(dim=1)
rewards_with_kl = rewards - Config.beta * kl_penalty
```

这种方法（无偏 K1 估计器）提供了更好的方差与偏差平衡梯度（训练更稳定），并带来了更好的性能。

![alt text](img/rl_impl_contrast.png)

<small>*图片来自 *[Shah et al., 2026](https://arxiv.org/abs/2512.21852)*，展示了 K1 in Rewards 相比 K3 in Loss 在降低方差方面的优势。*</small>

### 玩具环境 `ToyEnv`

为了确保脚本能在普通硬件上快速运行，它被设计用于解决一个特定的**分段函数连续性**问题。

* **目标：** 找到 $a+b$ 的值，使函数在全域连续。
* **标准答案：** $a=-3, b=3 \implies a+b=0$。
* **奖励机制：** 针对识别出连续点（$x=2, x=-2$）、列出正确方程以及解出变量给出部分信用分（Partial credits）。这对于小规模测试非常必要。

*注：这相当于算法的单元测试。如果 loss 下降且 reward 达到 1.0，则证明 GRPO 实现是正确的。*

## 局限性与未来 Roadmap

本实现优先考虑可读性和教育价值，而非生产环境的效率。

* **生成速度：** 使用标准的 HF `generate()`。对于高吞吐量训练，请集成 `vLLM` 或 `SGLang`。
* **内存效率：** 使用标准 Padding。生产运行应使用序列打包（Sequence Packing/Flash Attention varlen）以避免在 pad tokens 上进行计算。
* **可扩展性：** 非常适合在 1-8 张 GPU 上学习和调试。对于在集群上训练 70B+ 模型，建议将此脚本作为逻辑参考，去修改/补丁像 OpenRLHF 或 Verl 这样的库。

## 📚 参考文献

* [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
* [A Comedy of Estimators: On KL Regularization in RL Training of LLMs](http://arxiv.org/abs/2512.21852)
* [JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://arxiv.org/abs/2512.16649)