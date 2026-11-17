---
title: "LLM 系统学习与复习手册"
subtitle: "从数学与神经网络，到 Transformer、后训练、KV Cache、量化与部署"
author: "根据本会话的学习问答系统整理"
date: "2026 年 10 月"
lang: zh-CN
---

> **文档定位**：这不是聊天记录的逐句摘抄，而是把本会话讨论的内容重新组织成一套可从头学习、也可按章节复习的知识体系。约定：重点讲 **Decoder-only 自回归大语言模型**（如 GPT 类架构），遇到其他架构会特别注明。公式里的矩阵统一采用**每个 Token 是一行**的约定。
>
> **建议阅读路线**：第一次按 ① 数学/训练基础 → ② Token/Embedding → ③ Attention/Block → ④ 训练与对齐 → ⑤ 推理/KV → ⑥ 量化/部署 阅读；面试复习优先看各章的“易错点”、第 22 章的辨析以及最后的速记表。

# 0. 一页建立全局地图

一个现代 Chat LLM 可以从 **数据、模型、训练、推理、服务** 五条线认识：

```text
文本数据 → Tokenizer → Token IDs → Embedding
                                  ↓
                         多层 Transformer Block
                      (Attention + FFN/MLP)
                                  ↓
                            Final Norm
                                  ↓
                       LM Head → Vocabulary logits
                                  ↓
                     生成下一个 Token，重复执行
```

**模型学习路径**：

```text
预训练 Pretraining（海量文本、next-token CE）
    ↓
Base Model（会续写，但未必擅长按指令回答）
    ↓
SFT（Prompt → 理想回答；可全参微调 / LoRA / QLoRA）
    ↓
偏好对齐（经典 RLHF 或 DPO 等）
    ↓
Chat / Instruct Model
```

**上线推理路径**：

```text
模型权重 + Config + Tokenizer
    ↓ 可选量化（W8、W4、W8A8 等）
推理引擎（vLLM / SGLang / TensorRT-LLM / llama.cpp 等）
    ↓
请求 → Tokenize → Prefill（处理整个已知 Prompt，建立 KV Cache）
    ↓
Decode（逐 Token 生成，读取/扩展各层 KV Cache）
    ↓
Sampling / Tool call / JSON / Streaming / API
```

**最重要的三组区别**：

| 容易混淆 | 真正区别 |
|---|---|
| Attention vs FFN | Attention **跨 Token** 汇聚信息；FFN **逐 Token** 加工特征。 |
| 训练 vs 推理 | 训练时多个位置可并行预测、反向传播更新参数；推理时权重固定、自回归逐 Token 生成。 |
| Weights vs Activations vs KV Cache | 权重是训练得到的固定参数；Activation 是运行时中间数据；KV Cache 是每层历史 Token 的 K/V。 |

# 第一部分　数学与神经网络基础

## 1. 向量、矩阵与 Linear 的维度

- **标量**：一个数，如损失值 `0.8`。
- **向量**：有顺序的数字序列，如某个 Token 的 `[0.1, -0.2, ...]`。
- **矩阵**：若干行、列组成的二维数组；一段序列的 hidden states 就是矩阵。

矩阵相乘的规则：

\[
(m\times n)(n\times p)=(m\times p)
\]

例如一段 Prompt 有 \(n=4\) 个 Token，每个 Token \(d=4096\) 维，输入矩阵：

\[
X\in\mathbb R^{4\times4096}
\]

一个 Linear 层把 4096 维映射到 16384 维：

\[
Y=XW+b,\qquad W\in\mathbb R^{4096\times16384}
\]

得到 \(Y\in\mathbb R^{4\times16384}\)。偏置 \(b\) 是 \(16384\) 维，会广播到所有 Token 行。**Token 数不变，每个 Token 的特征维度改变。**

对于第 \(i\) 个 Token 的第 \(j\) 个输出维度：

\[
y_{ij}=\sum_{k=1}^{d_{in}}x_{ik}W_{kj}+b_j
\]

因此**一个 Linear 只有一个权重矩阵 \(W\)**（以及可选偏置），并不是 \(XW_1+XW_2+\cdots\)。矩阵的每一列可以理解为“一个输出维度的权重向量”。所有 Token 使用**同一套** \(W,b\)，可由一次矩阵乘法并行完成。

> **易错点**：矩阵的**行数 \(n\)** 表示 Token 个数，**列数 \(d\)** 表示一个 Token 有多少特征。Linear/FFN 通常混合的是单个 Token 内部的 **d 个特征**，而非不同 Token 的行。

## 2. MLP、FFN 与激活函数

**MLP** = Multi-Layer Perceptron，多层感知机。**FFN** = Feed-Forward Network，前馈神经网络。在 Transformer 中，FFN 通常就是**对每个 Token 独立应用的一个 MLP**；两者在这个场景里经常混用，但 MLP 是更宽泛的模型结构称呼。

经典 FFN：

\[
FFN(X)=\operatorname{Linear}_2\big(\phi(\operatorname{Linear}_1(X))\big)
\]

典型维度：

\[
n\times4096\to n\times16384\to n\times4096
\]

其中 \(\phi\) 是非线性激活函数。**如果没有非线性，两个 Linear 会合并成一个 Linear**，因此下面的无激活手算例子仅用于说明维度，不代表完整 FFN。

**逐 Token 手算例子（省略 Bias、激活）**：

\[
X=\begin{bmatrix}1&2\\3&1\\2&4\end{bmatrix}_{3\times2},\quad
W_1=\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}_{2\times3}
\]

\[
XW_1=\begin{bmatrix}1&2&4\\3&1&7\\2&4&8\end{bmatrix}_{3\times3}
\]

再取 \(W_2=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}_{3\times2}\)，得到：

\[
(XW_1)W_2=\begin{bmatrix}5&6\\10&8\\10&12\end{bmatrix}_{3\times2}
\]

这等价于三行分别经过**同一组** \(W_1,W_2\)。每行都只取决于**自己的输入行**：

\[
FFN(x_i)=\operatorname{Linear}_2(\phi(\operatorname{Linear}_1(x_i)))
\]

**FFN 不直接读取其他 Token 的行**，但它的输入 \(x_i\) 可能已通过前面的 Attention 融合了历史上下文。因此“不直接跨 Token 计算”不等于“完全与上下文无关”。

常见激活函数：

| 激活 | 直觉与用途 |
|---|---|
| ReLU | \(\max(0,x)\)，简单、非线性。 |
| GELU | 平滑的非线性，经典 Transformer FFN 常见。 |
| Sigmoid | 映射到 \((0,1)\)，常用于门控、二分类输出。 |
| Tanh | 映射到 \((-1,1)\)，经典循环网络常见。 |
| SiLU / SwiGLU | 现代 LLM FFN 常用的平滑门控结构。 |

Sigmoid/Tanh 在输入绝对值较大时容易**饱和**，导数很小，会加剧深层网络的梯度消失。

## 3. 导数、偏导、梯度与梯度下降

模型有参数 \(W\)，输入数据固定时，训练损失可视为关于参数的函数 \(L(W)\)。

- **导数**：一维函数某点的变化率、局部斜率。
- **偏导**：只改变一个变量、其他变量保持不变的导数。
- **梯度**：所有参数偏导数组成的向量，指向函数局部上升最快的方向。

\[
\nabla L=\left[\frac{\partial L}{\partial w_1},\frac{\partial L}{\partial w_2},\ldots\right]
\]

梯度下降的核心更新：

\[
W_{new}=W_{old}-\eta\nabla L(W_{old})
\]

\(\eta\) 是学习率。**Loss 是高度，Gradient 是局部斜率**：Loss 值高不代表梯度一定大；梯度接近 0 也不保证 Loss 接近 0。

例如把 Loss 整体加 1000，梯度不会变；乘以 10，梯度会扩大 10 倍。

**为什么 MSE 也可能出现高 Loss、零梯度？**

若模型是简单线性 \(\hat y=wx\)，单样本 MSE \((wx-y)^2\) 对 \(w\) 是凸抛物线（\(x\ne0\) 时有唯一最小点）。但神经网络内部函数更复杂，例如 \(\hat y=w^2,\ y=1\)：

\[
L(w)=(w^2-1)^2,\qquad L'(w)=4w(w^2-1)
\]

在 \(w=0\) 时梯度为 0，Loss 却为 1；在 \(w=\pm1\) 时 Loss 为 0。**外层使用 MSE 不意味着整个参数空间的优化地形是简单凸函数。**

## 4. Logits、Softmax 与各类 Loss

**Logits** 是模型输出的原始分数，不是概率。若某任务有 \(K\) 个互斥类别，一般输出 \(K\) 个 logits，再用 Softmax 变成概率：

\[
p_i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
\]

它们非负且总和为 1。对正确类别 \(t\) 的交叉熵：

\[
L_{CE}=-\log p_t
\]

**Logits 个数由输出任务决定，不是永远等于词表大小。** 3 分类输出 3 个 logits；GPT 下一 Token 预测有 \(V\) 种词表 Token，才输出 \(V\) 个 logits。

| Loss | 典型用途 | 直觉 |
|---|---|---|
| MSE | 回归，预测连续值 | 平方误差，大误差惩罚更重。 |
| MAE | 回归 | 绝对误差，较不受离群值影响。 |
| Huber | 回归 | 小误差近似 MSE，大误差近似 MAE。 |
| CE + Softmax | 多类互斥分类 | 提升正确类别的相对概率。 |
| BCE + Sigmoid | 二分类、多标签独立判断 | 每一类分别预测“是/否”。 |
| Focal Loss | 类别不均衡等 | 相对强调困难样本。 |
| KL Divergence | 分布对齐、蒸馏、正则 | 衡量两个分布差异。 |
| InfoNCE/对比学习 | 向量表示学习 | 拉近匹配样本，推远不匹配样本。 |

BCE 对一个二元标签的公式：

\[
L_{BCE}=-\left[y\log p+(1-y)\log(1-p)\right]
\]

**Softmax 让类别竞争、总和为 1；Sigmoid 独立输出，不要求总和为 1。** 对二分类而言，1 个 Sigmoid logit + BCE 也可以与 2 个 Softmax logits + CE 数学上等价。实际框架的 `CrossEntropyLoss` 往往**直接接收 logits**，内部采用数值稳定的 LogSoftmax 实现，不用自己先显式 Softmax。

## 5. Forward、Backward、Optimizer 与训练集划分

训练的一次基本闭环：

```text
输入数据 → Forward（算预测与 Loss）
         → Backward（链式法则算参数梯度）
         → Optimizer.step()（根据梯度修改权重）
         → zero_grad()（清理累积梯度）
```

**反向传播算梯度，不负责实际更新参数。** 最普通 SGD 沿负梯度更新；Adam 综合一阶、二阶梯度统计来调节不同参数的步长；AdamW 常用于 Transformer，权重衰减处理与 Adam 的梯度更新相区分。

```python
optimizer.zero_grad()
logits = model(inputs)         # Forward
loss = criterion(logits, labels)
loss.backward()               # 只计算梯度
optimizer.step()              # 真正更新参数
```

- **Batch**：一次更新使用的一批样本；**Epoch**：训练集完整遍历一次，通常包含许多 Batch。
- **Train**：计算梯度并更新权重；**Validation**：观察泛化与选超参数/检查点；**Test**：最终评估，不宜频繁拿来调参。
- **Overfitting**：Train 表现持续提高，但验证集/真实场景变差。可通过合理数据、早停、Dropout、权重衰减等方式缓解。
- **Dropout**：训练时随机屏蔽部分激活，推理时通常关闭；**Weight Decay**：约束参数规模以改善泛化。它们都是通用 DL 技术。

# 第二部分　从文本到向量

## 6. Tokenizer、Token、Embedding、Hidden State

典型流转：

```text
文本 → Tokenizer → Tokens → Token IDs → Embedding → Transformer
```

Token **不一定是一个汉字或一个英语单词**，也可能是子词、标点、字节序列等。Token ID 只是**词表索引**，不是语义向量。

**BPE**：从较小的单位开始反复合并高频相邻序列，学出子词词表；**WordPiece**：另一类经典子词方法，常与 BERT 联系；**SentencePiece**：可提供 BPE/Unigram 等分词方案的工具体系；**Byte-level BPE**：以 UTF-8 字节为基本单位，有较强的文本覆盖能力。Tokenizer 的词表学习是离散统计过程，不是每次通过 Transformer 反向传播动态分词。

设词表规模 \(V=50000\)，隐藏维 \(d=4096\)：

\[
E\in\mathbb R^{50000\times4096},\qquad e_i=E[\text{token\_id}_i]
\]

同一 Token ID 的**初始 Embedding**相同；经过 Attention 后，它在不同上下文中的 **Hidden State** 可以不同。

- **Embedding**：把离散 Token ID 映射成可训练的实数向量。
- **Hidden State \(h_i\)**：模型某层对位置 \(i\) 的内部表示，可已融合前文信息；它**不是** Transformer Encoder。
- **Position**：序列中的先后顺序，与 Embedding 空间中的坐标是两个概念；模型需要位置编码或 RoPE 等机制感知顺序。

一个可运行的大语言模型通常至少需要：**模型架构/Config + 权重/Weights + 与训练匹配的 Tokenizer**；模型不一定只存在一个 `.pth` 文件里。


# 第三部分　Transformer 核心原理

## 7. Self-Attention：Q/K/V 到底在算什么

设当前一层输入 \(X\in\mathbb R^{n\times d_{model}}\)。对于一个 Head，经过三个**独立可训练的 Linear**：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

假设每个 Head 的 Query/Key 维度是 \(d_k\)，Value 维度是 \(d_v\)，则：

\[
Q,K\in\mathbb R^{n\times d_k},\quad V\in\mathbb R^{n\times d_v}
\]

缩放点积 Attention：

\[
\boxed{O=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V}
\]

其中 \(M\) 是 Mask。\(QK^\top\) 的维度为 \(n\times n\)，**第 \(i\) 行是第 \(i\) 个 Token 的 Query 对所有 Key 的分数**：

\[
[q_i\cdot k_1,\ q_i\cdot k_2,\dots,q_i\cdot k_n]
\]

然后对**该行** Softmax 得到权重 \(\alpha_{ij}\)，用这些权重加权求和各 Token 的 Value：

\[
o_i=\sum_{j\,\text{允许被看到}}\alpha_{ij}v_j
\]

**角色定位**：

| 符号 | 直觉 | 为什么需要 |
|---|---|---|
| Q / Query | “我当前位置想找什么信息？” | 决定这一行要关注哪些位置。 |
| K / Key | “我这边能被别人怎样匹配？” | 给未来的 Query 提供匹配对象。 |
| V / Value | “我真正提供什么内容？” | 被注意力权重加权，构成输出。 |
| O / Output | “我综合查询结果得到的表示” | 进入当前层后续的 Residual/FFN/下一层。 |

**\(\sqrt{d_k}\)**：避免维度增大导致点积分数尺度太大、Softmax 过分尖锐。这里的 \(d_k\) 是一个 Head 的 Key 维度，**不一定等于 \(d_{model}\)**。

> **最容易误会的方向**：不是“当前位置的 K 和其他所有 Q 算、然后更新所有旧 Token”。求第 \(i\) 个位置的 \(o_i\)，是**当前位置的 \(q_i\)** 询问所有允许的 \(k_j\)，再汇总 \(v_j\)。每一行对应一个要计算输出的 Query。

## 8. Causal Mask、Padding Mask 与二次复杂度

Decoder-only 的下一个 Token 预测不能偷看未来。对 3 个 Token，Causal Mask 可写为：

\[
M=\begin{bmatrix}
0&-\infty&-\infty\\
0&0&-\infty\\
0&0&0
\end{bmatrix}
\]

在 Softmax 之前把未来位置的分数加上 \(-\infty\)，相应权重就变为 0：

```text
位置 1：只能看 1
位置 2：可以看 1、2
位置 3：可以看 1、2、3
```

**Padding Mask** 是另一种限制：不让模型关注对齐 Batch 时添加的无效 PAD 位置。**Self-Attention** 指 Q/K/V 来自同一序列；**Causal Self-Attention** 指同一序列内还限制不能看未来。

若完整计算 \(n\) 个位置的 Attention，位置间分数矩阵规模为 \(n\times n\)，典型复杂度有 \(O(n^2)\) 的长度项；有因果三角 Mask 不改变渐近阶。真实计算还涉及隐藏维度，常写为 \(O(n^2d)\)。

## 9. Multi-Head Attention：为什么不是一个大 Head 就够了

假设 \(d_{model}=4096\)，用 32 个 Head，每个 Head 的维度为 128：

\[
4096=32\times128
\]

工程上常一次得到总 Q/K/V，然后 reshape（以 Q 为例）：

\[
[n,4096]\to[n,32,128]
\]

每个 Head 有自己的投影参数、得到自己的 \(n\times n\) 注意力分布：

\[
head_h=Attention(Q_h,K_h,V_h)
\]

再把 32 个 Head 的输出拼接、过输出投影 \(W_O\)：

\[
\operatorname{Concat}(head_1,\dots,head_{32})W_O
\]

得到 \(n\times4096\)。**一个 4096 维的大 Head 只有一个位置间注意力分布；32 个 Head 有 32 组不同的分布/特征子空间。** 但 Head 也不是越多越好：固定总维度下，Head 太多会令每个 Head 太窄，还有系统开销。

现代模型可能使用 **MQA/GQA**：Query Head 仍很多，但多个 Query Head 共享较少的 KV Head，主要有利于**降低 KV Cache 和读取开销**；不必强行要求 \(N_{Q-head}=N_{KV-head}\)。

## 10. Transformer Block、Residual、Norm、FFN

一个典型现代 Decoder-only 的 **Pre-Norm Block** 可以简写为：

\[
X_1=X+Attention(Norm(X))
\]

\[
X_2=X_1+FFN(Norm(X_1))
\]

其职责分工：

```text
X [n×d]
  ├─ Norm → Causal Multi-Head Attention ─┐
  └──────────────── Residual Add ←───────┘  → X1 [n×d]
  ├─ Norm → FFN / MLP ───────────────────┐
  └──────────────── Residual Add ←───────┘  → X2 [n×d]
```

- **Attention**：不同 Token 之间交换/汇聚信息；第 \(i\) 行只能受允许的 \(j\) 行影响。
- **FFN**：单个 Token 内的特征加工；共享同一套参数，不直接混合不同 Token。
- **Residual**：\(x+F(x)\)，保留原信息并为梯度提供较直接的通路。
- **Norm**：稳定向量的数值尺度，有助于深层训练。

**RMSNorm** 是常见的 Norm 实现；**SwiGLU** 是 FFN 内部常见的门控结构。RMSNorm 不是新的 Attention；SwiGLU 也不是 Transformer 外面额外叠的一种网络，都是相应模块的具体实现。

> **“横向/纵向”记忆法**：在一个 \(n\times d\) 张量上，Attention **跨行（Token 轴）**做信息汇聚；FFN/Linear **每行内部跨列（特征轴）**做变换。但矩阵乘法在 GPU 上通常并行执行，而不是一行一行串行跑。

## 11. 位置编码：RoPE 和 Encoder/Decoder 的定位

Self-Attention 单看 Q/K/V 本身，不会自然知道“先后顺序”。常见引入位置信息的方法：

1. **Learned positional embedding**：把学到的位置向量加在 Token 向量上。
2. **Sinusoidal positional encoding**：原始 Transformer 的固定正弦/余弦位置编码。
3. **RoPE（Rotary Positional Embedding）**：对 Q/K 引入与位置相关的旋转，使注意力分数含有位置信息；现代 LLM 常见。

架构对照：

| 架构 | 能看到什么 | 常见任务 |
|---|---|---|
| Encoder-only（如 BERT） | 典型为双向上下文 | 分类、文本表示、理解任务。 |
| Decoder-only（如 GPT） | Causal Self-Attention，仅自己及前文 | 自回归文本生成。 |
| Encoder–Decoder（如 T5） | Encoder 读输入；Decoder Causal 生成并 Cross-Attend 输入 | 输入到输出序列变换。 |

**GPT 的“Decoder-only”不是指最后的 LM Head**；它指其主干采用自回归的 Decoder 风格 Block。原始 Encoder–Decoder 的 Decoder 有 Cross-Attention，而纯 Decoder-only LLM 通常没有独立 Encoder 供它 Cross-Attend。

## 12. 一个 Prompt 从输入到输出：完整维度追踪

设：\(n=4\) 个 Token、\(d_{model}=4096\)、\(d_{ff}=16384\)、32 Heads、词表 \(V=50000\)，忽略 Batch 维（真实往往是 \(B\times n\times d\)）。

| 阶段 | 张量形状 | 解释 |
|---|---|---|
| Token IDs | \(4\) | 4 个离散整数。 |
| Embedding | \(4\times4096\) | 每行一个 Token 的初始向量。 |
| Q/K/V 总投影（传统 MHA 示例） | 各 \(4\times4096\) | 对每一行应用相应 Linear。 |
| 拆 32 Heads | \(4\times32\times128\) | 重新组织维度。 |
| 每个 Head 的 Attention 分数 | \(4\times4\) | 每个 Query 对所有 Key 的关系。 |
| 拼接、输出投影 | \(4\times4096\) | 每行已融合允许的上下文。 |
| FFN 第一层 | \(4\times16384\) | 每个 Token 的特征升维。 |
| FFN 第二层 | \(4\times4096\) | 每个 Token 降回模型维度。 |
| 多个 Block + Final Norm | \(4\times4096\) | 最终隐藏状态 \(h_1\ldots h_4\)。 |
| 全位置 LM Head（可用于训练） | \(4\times50000\) | 每行对应一个位置的下一 Token 预测。 |
| **推理首 Token 的 LM Head** | **\(1\times50000\)** | 只需要最后一行 \(h_4\)。 |

**并行性**：同一 Block 内已知的 4 个 Token 可通过矩阵运算并行处理；但 Block 1 的输出必须先产生，Block 2 才能运行。**Token 轴可并行，层与层有先后依赖。**

**为什么推理首 Token 只要 \(h_n\)，却不能只算最后一个 Token？** 因为第 2 层的第 \(n\) 个位置要读第 2 层其他历史位置的 K/V，而这些 K/V 依赖**前一层历史 Token 的 hidden states**。因此在原始 Prefill 里，所有 Prompt Token 都要逐层参与计算；最终输出预测只用 \(h_n\)，但各层历史 K/V 也被算好并保存。

# 第四部分　从预训练到对齐

## 13. Next-token Prediction、Teacher Forcing 与 PPL

自回归语言模型在位置 \(t\) 根据 \(x_{1:t}\) 预测 \(x_{t+1}\)。概率分解：

\[
P(x_1,\ldots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t})
\]

训练时使用现成文本，输入与标签错开一位：

```text
输入位置： 我 / 喜欢 / 吃 / 苹果
监督标签： 喜欢 / 吃 / 苹果 / <EOS>
```

位置 \(t\) 输出词表大小 \(V\) 的 logits，针对真实下一个 Token 算 CE，再对有效 Token 的 Loss 汇总/求平均。语言模型预训练因此可以理解成**每个位置都做一次 V 类分类**。

**Teacher Forcing**：训练时后续位置仍喂**真实前文 Token**，而不是模型前一步采样到的错误 Token。配合 Causal Mask，可以对各位置**并行计算**；推理时生成的下一个 Token 依赖上一个采样结果，所以必须**跨生成步串行**。

**Perplexity（困惑度）**：通常 \(PPL=\exp(\text{平均 token CE})\)。越低表示模型给真实后续 Token 的概率整体越高（比较前要留意语料与 Tokenizer 一致性）。

## 14. Pretraining、Base Model、SFT 与 Chat Template

**预训练**：大规模文本自身提供“下一 Token”标签，使用 next-token CE 做自监督学习，得到 **Base Model**。Base Model 有语言/知识/代码等能力，但训练目标主要是“续写”，不一定总能按用户指令像助手一样回应。

**SFT（Supervised Fine-Tuning）**：用优质的“指令/上下文 → 理想回答”样本继续训练，让模型更擅长按指令作答。SFT 通常还是 next-token CE；很多管线只在 **Assistant 回答 Token** 上计算训练 Loss，System/User 部分作为上下文，具体实现可不同。

**Chat Template**：将结构化消息（System、User、Assistant、工具消息）序列化为模型熟悉的特殊 Token/字符串协议。它不是新的 Transformer Block，推理时必须尽量与模型训练时的模板匹配。

```text
[system] 你是一个助手
[user] 请讲解梯度下降
[assistant] 梯度下降是……
```

模型实际使用的特殊 Token 名称和边界格式因家族而异，上面只是示意。

**记忆**：Pretraining 更多教“语言和知识如何分布”；SFT 更多教“面对指令该如何组织回应”。

## 15. Full Fine-tuning、LoRA、QLoRA：改哪些参数

**全参数微调**：更新模型里大量/全部权重，效果灵活，但权重梯度、优化器状态和激活缓存的成本高。

**LoRA（Low-Rank Adaptation）**：冻结原本的 \(W\)，只训练低秩增量。在本手册“行向量右乘权重”约定下：

\[
W'=W+\Delta W,\qquad \Delta W=AB
\]

\[
A\in\mathbb R^{d_{in}\times r},\qquad B\in\mathbb R^{r\times d_{out}}
\]

标准 LoRA 常用缩放：

\[
\boxed{W'=W+\frac{\alpha}{r}AB}
\]

若 \(W\) 是 \(4096\times4096\)，原权重约 1678 万参数；设 \(r=8\)，新增可训练参数为 \(4096\cdot8+8\cdot4096=65536\)，针对这个矩阵约少 256 倍。

| 超参数 | 意义 | 直觉 |
|---|---|---|
| `rank` / `r` | 低秩瓶颈维度 | 决定增量的容量；越大通常更灵活但耗资源。 |
| `alpha` | LoRA 增量缩放 | 标准形式 \(\alpha/r\)；不是直接决定参数个数。 |
| `target_modules` | 选择挂 LoRA 的 Linear | 如 `q_proj`、`v_proj`，或 Q/K/V/O 和 FFN。 |

```python
# 典型配置示意
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
```

这表示给指定的 Q/V 投影添加低秩增量，并只训练相应 Adapter。推理时在兼容情形下可把增量合并回 \(W\)。部分变体（如 rsLoRA）会使用不同缩放约定，不能把 \(\alpha/r\) 视为所有方法的绝对规则。

**QLoRA**：进一步把**冻结的 Base Model** 以低 bit（经典方案常见 4bit）保存，再训练较高精度的 LoRA Adapter。QLoRA 的重点是**省显存微调**，不意味着全流程所有乘法都直接 INT4 计算。

> **区分训练目标与训练方法**：**SFT 是“用什么目标/数据微调”**；**Full FT、LoRA、QLoRA 是“更新哪些参数、用什么存储策略”**。因此 SFT 可以用全参、LoRA 或 QLoRA 实现。

## 16. Preference Alignment：经典 RLHF 与 DPO

SFT 通常学“模仿给出的好答案”；偏好对齐进一步学**两个回答哪个更受偏好**。一条偏好数据可写成：

```text
Prompt：解释 Attention
Chosen：准确、具体、清楚的答案
Rejected：模糊、不完整的答案
```

**经典 RLHF（Reinforcement Learning from Human Feedback）** 常见路线：

```text
SFT 模型
 + 多组人类/标注者偏好比较
    ↓
训练 Reward Model（输入 Prompt+回答 → 偏好分数）
    ↓
使用 PPO 等 RL 方法调整 LLM，让期望 Reward 更高
    ↓
通常配合 KL 惩罚，限制模型过度偏离参考策略
```

Reward Model 是一个**学习出来的自动评分器**，并不完美；模型可能针对其弱点“刷分”（Reward Hacking），所以偏好质量、约束和评估非常重要。

**DPO（Direct Preference Optimization）**：仍使用 Chosen/Rejected 偏好对，但**不必单独训练一个 Reward Model，也无需在这个训练过程里运行 PPO**。它直接构造偏好 Loss，优化模型对两者的**相对倾向**，并用一个参考模型/参考分布形成约束。

| | SFT | 经典 RLHF | DPO |
|---|---|---|---|
| 核心数据 | 示例回答 | 偏好对 + RL 反馈 | 偏好对 |
| 优化重点 | 模仿目标序列的 Token | 通过奖励优化生成策略 | 优化偏好对的相对概率 |
| 单独 Reward Model | 不需要 | 典型方案需要 | 不需要 |
| PPO 等 RL 阶段 | 不需要 | 典型方案需要 | 不需要 |

**精确一点的 DPO 直觉**：它提升的是 Chosen 相对 Rejected、且相对参考模型的优势，**不能机械理解为每次更新都必然让 Chosen 的绝对概率上升、Rejected 的绝对概率下降**。

> **记忆**：SFT = 学一份示范；RLHF = 训练评分器后用 RL 优化；DPO = 用偏好对直接优化策略。RLHF 是更宽泛的“从反馈中进行强化学习”范畴，PPO 只是经典做法，并不是 RLHF 唯一算法。


# 第五部分　自回归推理、Prefill、Decode、KV Cache

## 17. 生成新 Token：Logits 到 Decoding Strategy

假设输入 Prompt 是 \(x_1,\ldots,x_n\)。最后一层第 \(n\) 个 Token 的 Hidden State：

\[
h_n\in\mathbb R^{1\times d_{model}}
\]

经过 LM Head（设词表大小 \(V\)）：

\[
\text{logits}=h_n W_{LM}\in\mathbb R^{1\times V}
\]

可经 Softmax 得到每个候选 Token 的概率，再选择/采样一个 \(x_{n+1}\)，追加到上下文，循环继续。**自回归**指“前一步输出会成为后一步输入条件”。

不同的 Decoding 策略：

| 方法 | 做法 | 关注点 |
|---|---|---|
| Greedy | 每步选概率最大 Token | 确定性强，不保证整体最优文本。 |
| Temperature | 计算 Softmax 前用 \(z/T\) 调节分布 | \(T<1\) 更尖锐；\(T>1\) 更平坦。 |
| Top-k | 仅保留概率最高的 k 个候选后采样 | 固定候选数量。 |
| Top-p | 保留累计概率达到阈值 p 的最小候选集后采样 | 候选数量可动态变化。 |

**Decoding** 这里指生成阶段的**Token 选择策略/逐 Token 生成**，不同于 `tokenizer.decode`（ID 转文本），也不同于 Transformer 架构中的 **Decoder**。

一个固定的网络及相同的输入产生相同 logits；**参数多不意味着函数本身具有随机性**。如果严格使用确定性算子与 Greedy，理论上同输入应得同输出；实践中的微小算子非确定性、服务环境、隐藏上下文或采样设置可能改变结果。

## 18. Prefill 与 Decode：一次处理整段 vs 每次只处理新 Token

假设 Prompt 长度 \(n=1000\)。**Prefill** 是第一次处理已经全部给出的 1000 个 Token：

\[
X:1000\times4096
\]

它们在各层被矩阵方式并行处理，最后得到 \(h_1,\ldots,h_{1000}\)，**预测第一个新增 Token 只用 \(h_{1000}\)**。同时，各层将 1000 个历史 Token 的 Key 和 Value 放入 KV Cache。

之后模型已采样生成 \(x_{1001}\)，**下一次 Decode 计算**只把这个新 Token 的当前向量作为新输入行：

\[
X_{new}:1\times4096
\]

该新行逐层经过 Q/K/V 投影、Attention、FFN，结合先前缓存的历史 K/V，形成其自身的最终 Hidden State，用于预测 \(x_{1002}\)。下一轮再把 \(x_{1002}\) 送入。因为未来 Token 不会改变过去 Token 的因果表示，**旧 Token 的各层计算可完全跳过**。

**时间先后需分清**：Prefill 的 \(h_n\) 已可预测/采样 \(x_{n+1}\)；然后才把这个新 Token 输入 Decode，计算它的位置表示以预测 \(x_{n+2}\)。实现对“首次 Decode step”的命名略有差异，但计算依赖不变。

```text
Prefill：输入 x1...xn
每层：完整 Prompt → Q/K/V + Attention/FFN；缓存历史 K/V
最后：取 hn → LM Head → 采样 x(n+1)

下一步 Decode：只输入 x(n+1)
每层：只计算新 Token 的 q/k/v、o、FFN；读取本层旧 K/V
最后：取 h(n+1) → LM Head → 采样 x(n+2)
```

## 19. 为什么只缓存 K/V，为什么不缓存 Q/O/H？

对于某一层、某个新位置 \(t\)：

\[
\boxed{o_t=\operatorname{Softmax}\left(\frac{q_tK_{1:t}^{\top}}{\sqrt{d_k}}\right)V_{1:t}}
\]

这个计算会用到**当前 \(q_t\)** 和**历史以及当前的 \(K,V\)**，却**不使用历史 \(q_i\) 或 \(o_i\)**。

| 中间量 | 当前 Token 算好后，下一个新 Token 是否直接需要？ | 典型处理 |
|---|---|---|
| \(q_i\) | 不需要；未来使用自己的 \(q_t\)。 | 用后可丢。 |
| \(k_i\) | 需要，未来每个 \(q_t\) 都可能匹配它。 | **缓存**。 |
| \(v_i\) | 需要，未来 Attention 权重会加权它。 | **缓存**。 |
| \(o_i\) | 不需要，未来新位置只计算自己的 \(o_t\)。 | 用于当次后续层运算后可释放。 |
| 历史 \(h_i\) | 通常不需要再次跑历史层。 | 其后续层 K/V 已在 Prefill 中生成。 |

**核心误区**：“每产生一个新 Token，Attention 都要再计算以前所有 Token 的 \(o_1,\ldots,o_{t-1}\)。”——**错**。有 KV Cache 时只算**新增位置的那一行** Attention：

\[
[q_tk_1^{\top},q_tk_2^{\top},\ldots,q_tk_t^{\top}]
\]

旧 \(o_i\) **既不重新计算，也没有必要缓存起来复用**。这依赖 **Causal Mask**：新增未来 Token 不会改变任何更早位置的 Attention 输出。

### 两层模型的逐层理解

```text
Prefill（历史 Token x1...xn）：
  Layer 1：算全部旧 Token 的中间表示 + 保存 Layer1 K/V
       ↓ 旧 Token 的 Layer1 输出进入 Layer2
  Layer 2：算全部旧 Token 的中间表示 + 保存 Layer2 K/V
       ↓
  最后一行 hn 用于生成第一个新 Token

Decode（新 Token x(n+1)）：
  Layer 1：新 Token q → 查 Layer1 历史 K/V + 自己 K/V
       ↓ 只得到新 Token 的 Layer1 输出
  Layer 2：新 Token q → 查 Layer2 历史 K/V + 自己 K/V
       ↓ 只得到新 Token 的 Layer2 输出
  最后一行 h(n+1) 用于预测下一个 Token
```

**为什么不能只在第一层缓存 K/V？** 后面各层都有各自的 \(W_Q^{(l)},W_K^{(l)},W_V^{(l)}\)，也有不同的历史 Hidden State，所以**每一层需要自己的一套 K/V**。

### 数学形状：Prefill 和 Decode 对照

假设一个 Head 维度 \(d_k=d_v=128\)。

| | Prefill，\(n=1000\) | Decode，已有 1000 个、计算新 Token |
|---|---|---|
| 新 Query 的行数 | \(1000\) | **1** |
| Q 形状 | \(1000\times128\) | **\(1\times128\)** |
| 可参与的 K/V | \(1000\times128\) | **\(1001\times128\)**（历史 + 当前） |
| 注意力分数 | \(1000\times1000\) | **\(1\times1001\)** |
| 本次产生的 O | \(1000\times128\) | **\(1\times128\)** |

注意 Decode **并非所有开销都变成常数**：每个新 Query 仍需与越来越长的历史 K 计算相关性，并读取历史 V；KV Cache 节省的是历史 Token 的**重复投影与整个网络重算**。单个新 Token 的 Attention 读取开销仍随上下文长度增长。

## 20. KV Cache 显存、TTFT、TPOT 与推理瓶颈

一个常用的 KV Cache 显存估算（不含管理开销）：

\[
\boxed{M_{KV}\approx2\times L\times B\times T\times H_{KV}\times d_{head}\times bytes}
\]

- \(2\)：K 与 V 两份；\(L\)：层数；\(B\)：同时缓存的序列数；\(T\)：缓存的 Token 长度。
- \(H_{KV}\)：**KV Head 数**，不一定等于 Query Head 数；\(d_{head}\)：每个 KV Head 的维度。
- `bytes`：每个缓存元素的存储字节数，如 FP16 是 2。

**举例**：32 层、单请求、4096 Token、8 个 KV Heads、128 维、FP16：

\[
2\cdot32\cdot1\cdot4096\cdot8\cdot128\cdot2
=536870912\text{ Bytes}\approx512\text{ MiB}
\]

如果改成 32 个 KV Heads 而其他不变，约为 2 GiB。可见 **GQA、KV Cache 量化**为何有价值。

服务指标：

| 指标 | 全称 | 直觉 |
|---|---|---|
| TTFT | Time To First Token | 请求发出到首个生成 Token 出现；含排队、Prefill 等。 |
| TPOT | Time Per Output Token | 首 Token 后平均每 Token 时间，主要受 Decode 影响。 |
| Throughput | 吞吐量 | 每秒整个系统处理/输出多少 Tokens 或请求（须指明口径）。 |
| Latency | 延迟 | 一个请求整体等待时间，可含排队、Prefill、Decode、网络等。 |

**Prefill** 通常是较大规模的并行矩阵计算，对长 Prompt 很敏感；**Decode** 每步只计算新 Token，在许多部署场景中容易受到权重/KV Cache **显存带宽**与并发调度影响。并非绝对：瓶颈取决于批大小、硬件和模型。

# 第六部分　量化与推理部署

## 21. Quantization：存储精度与计算精度不是一回事

权重在训练后有大量 FP16/BF16/FP32 数值。**量化**用较少比特表示它们，以节省模型存储和显存，并可能改善内存带宽或计算效率。

| 表示 | 典型每参数占用 | 对 7B 权重的理想化体积 |
|---|---:|---:|
| FP32 | 4 Bytes | 约 28 GB |
| FP16/BF16 | 2 Bytes | 约 14 GB |
| INT8 | 1 Byte | 约 7 GB |
| INT4 | 0.5 Byte | 约 3.5 GB |

实际还包括 scale、zero point、元数据、对齐开销，以及推理中的 **KV Cache、激活、运行时缓冲区**，不能直接把上表视为整机显存需求。

### 21.1 量化/反量化

简单的对称量化示意：

\[
q=\operatorname{round}(x/s),\qquad \hat x=sq
\]

其中 \(s\) 是 scale，\(q\) 是整数值，\(\hat x\) 是反量化后的**近似值**。更一般的非对称量化还用 zero point：\(\hat x=s(q-z)\)。低 bit 下许多原始浮点数被映射到同一表示，所以反量化后**无法恢复已经丢掉的原始精度**。

例如 \(0.137\) 量化后取整成 \(q=3\)、\(s=0.05\)，再反量化得到 \(0.15\)，不是原始的 \(0.137\)。

### 21.2 W（Weight）和 A（Activation）是什么

在 Linear \(Y=XW+b\) 里：

- **W = Weight**：模型训练得到的权重矩阵，推理时一般固定。
- **X = Activation**：模型运行过程中流动的**输入/中间特征张量**。在 Transformer 中通常是各 Token 的当前 Hidden States，**不是 ReLU/GELU 等激活函数层，也不是 Token ID 本身**。
- **Y**：该层产生的新 Activation；它可以继续作为下一个 Linear 的输入。

**Activation Function** 才是 ReLU/GELU/SiLU 等非线性函数；**Activation Quantization** 指量化运行时中间张量，不是只量化“激活函数的代码”。

**W4A16**：Weight 4-bit，Activation 16-bit；**W8A8**：Weight 8-bit，Activation 8-bit。这里的 W/A 是精度约定，不保证所有模块都严格采用相同类型，实际还看算法和 Kernel。

### 21.3 计算时会不会先全部恢复成 FP16？

**不会恢复原始精确数值**，因为低 bit 信息已丢失；实际常见两种实现路线：

1. **Weight-only，例如 W4A16**：权重低 bit 存储，计算时由融合 Kernel **按块读取 → 即时反量化为近似浮点 → 与 FP16/BF16 Activation 运算**，通常不在显存里展开一个完整 FP16 权重副本。
2. **W8A8 等低精度运算**：硬件可以直接做 INT8×INT8 的低精度矩阵乘法，并用较高精度（例如 INT32）累加，之后结合 scale 得到输出；实际实现依硬件/框架不同。

因此 **“低 bit 存储”不必等于“所有计算都在同样的 bit 数上完成”**；量化也**不是**“启动后把整个模型还原成原始 FP16”。

### 21.4 PTQ、QAT、GPTQ、AWQ 与分组

| 概念 | 意义 | 典型定位 |
|---|---|---|
| PTQ | Post-Training Quantization | 训练结束后量化，部署常见。 |
| QAT | Quantization-Aware Training | 训练时模拟/感知量化误差，帮助模型适应。 |
| GPTQ | 常见的权重量化方法 | 借助校准信息减轻量化带来的模型误差。 |
| AWQ | Activation-aware Weight Quantization | 利用 Activation 信息识别/保护重要权重。 |
| Per-tensor | 较大范围共享 scale | 元数据少，但精度灵活性较低。 |
| Per-channel / Group-wise | 每个通道或小组分别量化 | 更细粒度，通常质量更好但元数据增加。 |

`group_size=128` 可以理解成某种分组量化配置下，每 128 个权重元素共享一组量化参数（具体分组轴视算法而定）。

**QLoRA vs GPTQ/AWQ**：QLoRA 的核心应用是**低显存微调**；GPTQ/AWQ 常见于**训练后权重量化/推理部署**。不能因为都出现 4-bit 就把它们当成同一件事。

## 22. 模型文件、推理引擎与多用户 Serving

需要区分：**模型文件** 和 **能承受真实流量的推理服务**。

```text
Weights + Config + Tokenizer
         ↓
     Inference Engine
         ↓
模型加载、显存分配、KV Cache 管理、Prefill/Decode、采样
         ↓
请求队列、Batching、并发控制、Streaming、HTTP API
         ↓
      Web / App / Agent
```

常见引擎定位：

| 技术 | 典型特点 |
|---|---|
| vLLM | GPU LLM Serving、高吞吐、PagedAttention 风格的 KV 管理、动态批处理、API 服务。 |
| SGLang | 面向 LLM/VLM Serving 的高性能推理与编排生态。 |
| TensorRT-LLM | 偏 NVIDIA GPU 深度优化与加速。 |
| llama.cpp | 本地/CPU/消费级设备友好，经常搭配 GGUF。 |

**GGUF / safetensors 是存储格式，不是某一种固定的量化算法**；GGUF 可以承载不同精度和量化版本。引擎是否支持特定格式/量化 Kernel 要看其兼容性。

### Continuous Batching

多个用户的生成长度不同，传统固定 Batch 若必须等最长请求结束，会浪费 GPU 容量。**Continuous Batching** 动态加入/移出正在生成的请求，使每个 Decode 时刻可以一起处理一批活跃序列，提高设备利用率。

好处：通常提升总体吞吐；代价：排队与调度可能影响个别请求延迟。因此 **Throughput 最大** 与 **单用户 Latency 最小** 不一定同时成立。

### 多 GPU 并行

- **Tensor Parallel（TP）**：一个层内的大矩阵/Head 等分配在多张 GPU 上协作；适合单层工作量大。
- **Pipeline Parallel（PP）**：不同 Layer 分布在不同 GPU；例如 80 层拆成若干阶段。需要管理阶段间传输和流水线空泡。

一个服务的 GPU 显存不只是权重：

\[
M_{total}\approx M_{weights}+M_{KV}+M_{activations}+M_{runtime}+\cdots
\]

请求的并发数、上下文长度、输出长度、量化精度、KV Head 数、Batch 调度都会影响能否装下与速度。

# 第七部分　Agent、RAG 与应用层可靠性

## 23. 工具调用/结构化输出：低温为什么不够

温度 \(T\) 只改变抽样概率分布：

\[
P_i=Softmax(z_i/T)
\]

低温（如接近 0）使模型更倾向高概率选项，因此**通常**能减少字段名和工具参数的随机波动；但如果高概率输出本身格式错误，低温只会让错误更稳定。

例如 JSON 需要：

```json
{"city": "Shanghai"}
```

模型如果偏向输出 `{"city": Shanghai}`（字符串未加引号），Greedy/低温不能自动纠正。

要提高可靠性，还要考虑：**JSON Schema、受约束解码（Grammar/Constrained Decoding）、字段类型校验、工具参数验证、异常处理及必要的重试**。Schema 的约束强度取决于具体 API/推理引擎支持；即使保证**语法正确**，也不代表字段值**语义正确**。

> 记忆：**低温 = 少随机选；受约束生成 = 禁止不合法形式；外部校验 = 再检查业务合理性。**

## 24. RAG 为什么影响 TTFT 与 KV Cache

RAG（Retrieval-Augmented Generation）将检索到的文本放进 Prompt。例如原提问 500 Tokens，新增资料 4500 Tokens，Prefill 长度就由 500 增加到 5000。

这会带来：

- **TTFT 变长**：Prefill 需处理更多 Token（还包括检索本身的耗时）。
- **KV Cache 变大**：历史长度 \(T\) 增长，而 \(M_{KV}\propto T\)。
- **质量未必提升**：无关资料、冲突片段、过长上下文可能干扰回答。

因此 RAG 的优化不仅是“多召回”，还包括**检索质量、去重、Rerank、选取合适的 Top-k、压缩文本、文档分块策略和上下文预算**。只要知识仍需要即时检索，量化和 KV 优化也不能替代检索质量。

## 25. 知识答错：Prompt、RAG、SFT/LoRA 如何选

| 常见根因 | 优先思路 | 能解决什么 | 不能保证什么 |
|---|---|---|---|
| 模型本来懂，却没按要求答 | **Prompt / 输出约束** | 限定格式、要求引用来源、拒答条件、步骤。 | 无法凭空补齐缺失知识。 |
| 缺少内部/最新/特定文件的事实 | **RAG / 调外部系统** | 把可更新的真实资料动态提供给模型。 | 检索错、文档错仍会误答。 |
| 反复不擅长某类任务方式或领域表达 | **SFT / LoRA** | 长期改写模型的响应行为、风格、任务适应性。 | 不能保证精准记住所有动态事实。 |
| 必须读取实时状态、执行事务 | **Tool / API 调用** | 获取实时数据、执行受控动作。 | 仍需权限、验证、错误处理。 |

例如“公司今天的审批规则”优先检索权威制度或调业务系统，不应每次变更都微调；“长期要求某种领域术语和固定解释风格”才可能值得用 SFT/LoRA。实际项目常用 **Prompt + RAG + Tools + Validator** 组合，而不是四选一。

# 第八部分　易错点、复习路径与自测

## 26. 15 个最容易反复混淆的点

| 错误/不严谨理解 | 更准确的理解 |
|---|---|
| 1. Token 就是一个字/单词 | Token 是分词器划定的单位，可为子词、标点、字节等。 |
| 2. Token ID 已经是语义向量 | ID 是索引；Embedding 才是连续向量。 |
| 3. Hidden State 就是 Encoder | Hidden State 是位置的内部表示，Encoder 是一类网络架构。 |
| 4. Linear 是对不同 Token 做加权求和 | Linear 的一行输出由**同一 Token 内各输入维度**加权产生。 |
| 5. 一个 Linear 有无数个 W1、W2 然后相加 | 统一为一个权重矩阵 \(W\)，每列负责一个输出维度。 |
| 6. 两个 Linear 不加激活也能产生复杂非线性 | 连续线性/仿射层可合并；非线性来自 Activation/门控。 |
| 7. Attention 里当前 K 查历史全部 Q | 算当前输出时，**当前 Q 查历史/当前 K**，再汇总 V。 |
| 8. FFN 完全不包含上下文 | FFN 不直接跨 Token，但输入 Hidden State 往往已有上下文。 |
| 9. 每层会把 n 个 Token 串行送进 FFN | 同层各行可矩阵并行计算，跨层须按顺序执行。 |
| 10. Prefill 的 n 个 Hidden State 都必须过 LM Head | 首个生成 Token 通常只需要最后位置 \(h_n\) 过 LM Head。 |
| 11. Decode 每一步重新算所有旧 Token 的 O | 有 KV Cache 时**仅新增位置**过完整网络。 |
| 12. 不缓存 O 会浪费重复计算 | 旧 O **不再计算也不再使用**，所以无需缓存。 |
| 13. KV Cache 只有模型整体一份 | **每层各有**历史所有已处理 Token 的 K/V。 |
| 14. Activation Quantization = 量化 GELU | Activation 指运行时中间数据，不是激活函数本身。 |
| 15. INT4 权重能还原原始 FP16 精度 | 反量化只得到近似值，丢失的精度回不来。 |

## 27. 核心公式速记页

**矩阵与 Layer**：

\[
(n\times d_{in})(d_{in}\times d_{out})=n\times d_{out}
\]

\[
Y=XW+b
\]

**梯度更新**：

\[
W\leftarrow W-\eta\nabla_WL
\]

**Attention（可含 Mask）**：

\[
\operatorname{Attention}(Q,K,V)=Softmax\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
\]

**Decoder-only Block（Pre-Norm 形式）**：

\[
X_1=X+Attention(Norm(X)),\quad X_2=X_1+FFN(Norm(X_1))
\]

**Next-token CE**：

\[
L_t=-\log P(x_{t+1}\mid x_{1:t})
\]

**LoRA**：

\[
W'=W+\frac{\alpha}{r}AB
\]

**单个新 Token 的 Attention**：

\[
o_t=Softmax\!\left(\frac{q_tK_{1:t}^\top}{\sqrt{d_k}}\right)V_{1:t}
\]

**KV Cache 显存估算**：

\[
M_{KV}\approx2LBTH_{KV}d_{head}\,bytes
\]

**量化近似**：

\[
q\approx\operatorname{round}(x/s),\quad \hat x=sq
\]

**一个经典 Decoder-only LLM 的主干维度**：

\[
n\times d\xrightarrow{Attention}n\times d
\xrightarrow{FFN}n\times d
\xrightarrow{LM\ Head}n\times V
\]

推理时通常只取末行投影：

\[
h_n:(1\times d)\rightarrow\text{logits}:(1\times V)
\]

## 28. 自测 12 题（先自己回答，再对照）

1. 4 个 Token、隐藏维 4096，FFN 的第一个 Linear 把 4096 扩到 16384：\(X,W,Y\) 各是什么形状？
2. 为什么 \(W\) 可以理解为很多权重向量，但一个 Linear 仍只写 \(XW+b\)？
3. 当前 Token \(i\) 要计算 \(o_i\)，它的 Q 具体和哪些 K 比较？哪些 V 被汇总？
4. Causal Mask 限制未来位置，为什么 Prefill 仍可同时算所有位置？
5. 32 个 Attention Heads 与 1 个很宽的 Head 的关键区别是什么？
6. FFN 不直接读取别的 Token，为何 FFN 的输出仍可能受前文影响？
7. 为什么推理 Prefill 最后只需 \(h_n\) 预测新 Token，但前 \(n-1\) 个 Token 仍需经过每一层？
8. 为什么在 Decode 时不需要缓存历史 \(Q\) 和 \(O\)？
9. 如果 Prompt 1000 个 Token，Decode 新位置 1001，第一个 Head 的 Attention 分数矩阵有几行、几列？
10. `W4A16` 的 A 是否表示 Activation Function？INT4 参数计算前会恢复原来 FP16 数值吗？
11. SFT 与 DPO 的训练数据/目标信号有什么不同？QLoRA 和 GPTQ 为什么不能画等号？
12. 公司制度每周更新，而模型答错了，应优先 Prompt、RAG 还是 SFT？如果输出 JSON 语法偶尔错，光用低温足够吗？

**简答核对**：

1. \(X:4\times4096\)，\(W:4096\times16384\)，\(Y:4\times16384\)。
2. W 的每列映射到一个输出维度；一整个矩阵乘法已把全部加权求和并行算完。
3. 用 \(q_i\) 匹配位置 \(\le i\) 的 Key，再加权相应 Value。
4. 已知整段序列，可并行形成各位置矩阵计算，然后用 Mask 屏蔽未来信息。
5. 独立的投影空间、独立的注意力分布；单 Head 只有一份注意力分布。
6. FFN 的输入 Hidden State 已经经过 Attention，含前文信息。
7. 下一层历史 K/V 依赖前一层各历史位置的 Hidden State；这些历史 K/V 未来 Decode 还要用。
8. 新位置只使用自己的 Query 和历史 K/V；历史 O 也不被下轮重算或读取。
9. \(1\times1001\)（该层单 Head，新 Query 对 1001 个可见 Key）。
10. A 是运行时 Activation；不能还原原始被舍弃的精度。
11. SFT 模仿示范回答；DPO 利用偏好对优化相对偏好。QLoRA 主要用于低显存微调，GPTQ 常用于训练后权重量化。
12. 优先 RAG/权威数据源；JSON 用 Schema、约束解码和外部校验，低温不够。

## 29. 三种复习节奏

| 用时 | 复习目标 | 建议章节 |
|---|---|---|
| 15 分钟 | 恢复全局框架 | 0、12、19、21、26、27 |
| 45 分钟 | 讲清 Transformer → KV | 7–12、17–20，手画矩阵形状。 |
| 90 分钟 | 面试级串联原理与工程 | 1–5、13–16、17–25，完成第 28 章自测。 |

**推荐练习**：自己画一张 \(n\times d\) 输入在 Attention、FFN、LM Head 中的形状变化图；再画一张两层 Transformer 的 Prefill/Decode 数据流。能不看文档解释“旧 O 为什么不用缓存”，说明核心已经掌握。

# 附录 A　对确定性、推理循环与 AI 意识的延伸讨论

本会话还讨论过“固定权重的复杂网络是否随机”“人脑为何不像一次性函数调用”“有持续内部循环的 AI 是否更像人脑”等问题。这些属于与主线相关但并非 LLM 实现必需的思考。

**确定性、可预测性、可计算性不是同一件事。** 固定权重 + 同一输入 + 理想确定性运算，输出可以确定，但复杂系统依然可能因为初始状态、环境、数值差异而难预测。随机采样也不等于“模型有意识”。

人脑更适合粗略理解成不断随感官输入、身体状态、神经活动与可塑性演化的动态系统：\(S_{t+1}=F(S_t,\text{input})\)。传统“Prompt → 一次推理 → 回答”式 LLM 与这种持续状态演化明显不同。Agent 循环、持久记忆、递归/循环 Transformer、潜在状态推理等研究方向能引入更多持续性，但**不能单凭循环结构或参数规模推出主观意识**。意识与主观体验是否、如何在 AI 中出现，仍是开放研究问题。

# 附录 B　中英术语对照

| 缩写/英文 | 中文/提示 |
|---|---|
| DL / Deep Learning | 深度学习 |
| MLP / Multi-Layer Perceptron | 多层感知机 |
| FFN / Feed-Forward Network | 前馈神经网络 |
| Q / K / V | Query / Key / Value，查询 / 键 / 值 |
| MHA | Multi-Head Attention，多头注意力 |
| LM Head | Language Model Head，语言模型输出映射层 |
| CE / BCE | 交叉熵 / 二元交叉熵 |
| SGD / AdamW | 优化器；SGD 是随机梯度下降，AdamW 是现代自适应优化器 |
| SFT | Supervised Fine-Tuning，监督微调 |
| PEFT | Parameter-Efficient Fine-Tuning，参数高效微调 |
| LoRA / QLoRA | 低秩适配 / 量化 Base Model 的 LoRA 微调 |
| RLHF / DPO | 基于人类反馈的强化学习 / 直接偏好优化 |
| PPO | Proximal Policy Optimization，经典 RL 算法 |
| PPL | Perplexity，困惑度 |
| TTFT / TPOT | 首 Token 延迟 / 每个输出 Token 耗时 |
| PTQ / QAT | 训练后量化 / 量化感知训练 |
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| TP / PP | Tensor Parallel / Pipeline Parallel |
| KV Cache | 各层历史 Token 的 Key/Value 缓存 |
| RoPE / RMSNorm / SwiGLU | 旋转位置编码 / 一种归一化 / 一种门控 FFN |

---

**最终主线，一句话串联**：文本被 Tokenizer 转成 ID，Embedding 把 ID 变成向量；Attention 让 Token 跨位置获取上下文，FFN 对每个 Token 的特征做非线性加工，多层堆叠后通过 LM Head 预测下一个 Token；预训练、SFT 和偏好对齐让模型学会语言与行为；推理时先 Prefill 再逐 Token Decode，利用各层 KV Cache 只计算新位置；上线则通过量化、调度、显存管理和推理引擎优化服务成本、延迟与吞吐。
