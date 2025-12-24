# CViT Model Architecture

CViT (Conditional Vision Transformer) 是一个基于 JAX/Equinox 实现的混合架构模型。它结合了 Vision Transformer (ViT) 强大的特征提取能力和基于坐标的隐式查询机制（Implicit Querying）。

该模型的设计目标通常用于处理物理场模拟、神经算子学习（Neural Operator）或超分辨率任务。它允许模型根据输入的网格数据（如初始条件或某一时刻的场），查询任意时空坐标点 $(x, y, t)$ 处的值。

## 总体架构 (Overall Architecture)

模型由两个主要部分组成：
1.  **Encoder (编码器)**：基于 ViT，负责将输入的网格数据（图像形式）编码为一组潜在的特征向量（Latent Vectors）。
2.  **Decoder (解码器)**：基于 Cross-Attention，负责接收任意的时空坐标查询，并结合编码器的特征预测该坐标处的物理量。

---

## 1. Encoder (编码器)

编码器的作用是理解输入的物理场（例如 $t=0$ 时刻的初始条件），并将其压缩为高维特征表示。

*   **输入 (Input)**: 形状为 `(C, H, W)` 的张量。
    *   $C$: 通道数（物理变量数量）。
    *   $H, W$: 网格的空间分辨率。
*   **Patch Embedding**:
    *   使用 `Conv2d` 将输入图像切分为非重叠的 Patch（例如 $16 \times 16$）。
    *   每个 Patch 被投影到维度 `emb_dim`。
    *   输出形状变换为 `(N_patches, emb_dim)`。
*   **Positional Embedding (位置编码)**:
    *   使用 **2D Sin-Cos Positional Embeddings**。这是固定的（非学习的）位置编码，用于保留 Patch 在原始网格中的空间信息。
    *   直接加到 Patch Embedding 上。
*   **Transformer Blocks (Self-Attention)**:
    *   堆叠多个 `SelfAttnBlock`。
    *   每个块包含 LayerNorm、Multi-head Self-Attention (MHSA) 和 MLP。
    *   通过自注意力机制，模型学习 Patch 之间的全局依赖关系（例如物理场中的长程相关性）。
*   **输出**: 编码后的特征序列，形状为 `(N_patches, emb_dim)`。

## 2. Decoder (解码器)

解码器的设计灵感来自于 Perceiver IO 或 DeepONet 的查询机制。它不是生成一张完整的图像，而是根据查询坐标生成对应的值。这使得模型具有**分辨率无关性 (Resolution Independence)**，可以在任意精细的坐标上进行查询。

*   **输入 (Input)**:
    *   **Context**: 来自 Encoder 的特征 `(N_patches, emb_dim)`。
    *   **Queries**: 查询坐标 `coords`，形状为 `(N_query, coord_dim)`。通常 `coord_dim=3` 对应 $(x, y, t)$。
*   **Fourier Features (傅里叶特征映射)**:
    *   原始的低维坐标 $(x, y, t)$ 直接输入神经网络通常难以学习高频细节（Spectral Bias 问题）。
    *   模型使用 `FourierEmbs` 将坐标映射到高维空间：
        $$ \gamma(v) = [\cos(2\pi \mathbf{B}v), \sin(2\pi \mathbf{B}v)] $$
        其中 $\mathbf{B}$ 是从高斯分布中采样的随机权重矩阵。
    *   这使得解码器能够捕捉物理场中剧烈变化的界面或高频特征。
*   **Cross-Attention Blocks (交叉注意力)**:
    *   这是解码器的核心。
    *   **Query (Q)**: 来源于坐标的傅里叶特征。
    *   **Key (K) / Value (V)**: 来源于 Encoder 的输出特征。
    *   通过 Cross-Attention，每个查询坐标“关注”输入场中与其最相关的 Patch 特征，从而聚合信息。
*   **MLP Head**:
    *   最后通过一个多层感知机（MLP）将聚合后的特征映射到目标输出维度 `out_dim`（例如预测的标量场值 $\phi$）。

## 3. 数据流总结 (Data Flow Summary)

1.  **输入场** $U(x, y)$ $\xrightarrow{\text{PatchEmbed + PosEmb}}$ **Patch 序列**。
2.  **Patch 序列** $\xrightarrow{\text{Encoder (Self-Attention)}}$ **潜在特征 (Latent Context)**。
3.  **查询坐标** $(x, y, t)$ $\xrightarrow{\text{FourierEmbs}}$ **查询向量 (Query Vectors)**。
4.  **查询向量** + **潜在特征** $\xrightarrow{\text{Decoder (Cross-Attention)}}$ **解码特征**。
5.  **解码特征** $\xrightarrow{\text{MLP}}$ **预测值** $\hat{\phi}(x, y, t)$。

## 4. 设计特点

*   **Equinox 框架**: 采用显式的参数传递和类结构，易于调试和通过 `jax.vmap`/`jax.grad` 进行变换。
*   **网格无关性**: 虽然编码器处理固定网格，但解码器是连续的。这意味着你可以用 $64 \times 64$ 的网格训练，但在测试时查询 $128 \times 128$ 的点阵，实现 Zero-shot Super-resolution。
*   **物理先验**: 傅里叶特征的引入非常适合拟合具有波动性质或尖锐界面的物理方程解。
