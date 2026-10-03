# 实现路线

每一关都按“解释 → 预测 → 自己写 → 检查 → 纠错 → 换条件验证”推进，不按视频观看时长推进。

| 关卡 | 亲手完成的内容 | 需要解释或验证的证据 |
|---|---|---|
| L00 | PyTorch 张量、矩阵乘法、softmax、梯度 | 先写预测，再运行张量诊断；实现一次训练步 |
| L01 | 完整模型的数据流 | 源句、memory、右移目标和 logits 的关系 |
| L02–L03 | 单头 Attention 与 Mask | QKᵀ/√d_k、softmax、乘V；因果与padding不变性 |
| L04 | 多头 Attention | 拆头、拼头、不同Q/K长度与输出形状 |
| L05 | Embedding 与正弦位置编码 | 缩放、buffer、位置0、device迁移 |
| L06 | FFN、残差与LayerNorm | 逐位置计算、归一化维度、Post-LN顺序 |
| L07 | Encoder | 两个子层、有效位置与padding |
| L08 | Decoder 与交叉注意力 | 三个子层、Q/K/V来源、未来信息不可见 |
| L09 | 完整模型 | [B,T,V] logits、前向、反向、有限梯度 |
| L10 | 数据、右移与损失 | BOS/EOS/PAD、teacher forcing和loss忽略位置 |
| L11 | 训练与逐token生成 | 小样本过拟合、EOS与长度上限 |
| L12 | 留出数据上的质量 | 固定划分与评价，避免只看训练集 |
| L13 | 消融与泛化 | 一次改变一个因素，固定预算与种子 |
| L14 | 闭卷重建与讲解 | 独立写核心模块、定位mask错误、迁移到新长度 |
| L15 | 可选decoder-only拓展 | 解释它与完整Encoder–Decoder的差异 |

当前所有关卡待本人完成。框架和检查器准备好不代表已经通过验收。
