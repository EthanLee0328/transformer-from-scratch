"""L04–L09 学习接口：原论文 Encoder–Decoder、Post-LN。

这些是练习合同，不是可训练模型。完成一个关卡再打开下一个 TODO。
不得用 nn.Transformer / MultiheadAttention / scaled_dot_product_attention
替代需要手写的主体；Linear、Embedding、LayerNorm、Dropout 可以使用。
"""

from torch import nn


class MultiHeadAttention(nn.Module):
    """L04。d_model % num_heads == 0。

    forward(q, k, v, allowed_mask=None) -> output, weights
    q/k/v: [B, Lq/Lk/Lk, d_model]；输出 [B,Lq,d_model]；
    weights [B,H,Lq,Lk]。投影、拆头、调用自己的 attention、拼头、输出投影。
    Q 可以来自 decoder，K/V 可以来自 encoder，Lq 不必等于 Lk。
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L04: 实现多头注意力。")

    def forward(self, q, k, v, allowed_mask=None):
        raise NotImplementedError("L04")


class SinusoidalPositionalEncoding(nn.Module):
    """L05。forward(x) -> [B,L,D]；x 已经是 sqrt(D)*token embedding。

    PE(pos,2i)=sin(pos/10000^(2i/D))；奇数维用 cos。
    用 register_buffer 保存非训练参数；支持 max_len 内不同序列长度。
    加 PE 后施加 dropout；device 随模型迁移。
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L05: 实现正弦位置编码。")

    def forward(self, x):
        raise NotImplementedError("L05")


class FeedForward(nn.Module):
    """L06。forward(x): [B,L,D] -> [B,L,D]；两层线性+ReLU。

    每个位置独立，参数跨位置共享。d_ff 是中间维度。
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L06: 实现逐位置前馈网络。")

    def forward(self, x):
        raise NotImplementedError("L06")


class EncoderLayer(nn.Module):
    """L07。forward(x, src_allowed_mask=None) -> [B,S,D]。

    原论文 Post-LN：LayerNorm(x + Dropout(Sublayer(x)))。
    自注意力后接 FFN，每个子层分别有残差与自己的 LayerNorm。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L07: 实现 Post-LN 编码器层。")

    def forward(self, x, src_allowed_mask=None):
        raise NotImplementedError("L07")


class DecoderLayer(nn.Module):
    """L08。forward(x, memory, tgt_allowed_mask, src_allowed_mask=None)。

    x [B,T,D]；memory [B,S,D]；输出 [B,T,D]。
    三个 Post-LN 子层依次是 masked self-attention、cross-attention、FFN。
    cross-attention 的 Q 来自前一子层，K/V 来自 encoder memory。
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L08: 实现解码器层与交叉注意力。")

    def forward(self, x, memory, tgt_allowed_mask, src_allowed_mask=None):
        raise NotImplementedError("L08")


class Transformer(nn.Module):
    """L09。完整 encoder–decoder（不是 decoder-only）。

    forward(src_ids, tgt_input_ids, src_allowed_mask, tgt_allowed_mask)
    -> 未归一化 logits [B,T,tgt_vocab_size]，交给 CrossEntropyLoss。
    encode/decode/project 接口方便 L11 逐 token 生成。
    训练目标错位、PAD loss 和 EOS 停止属于 L10–L11，不能靠模型猜。
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, max_len,
                 d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        raise NotImplementedError("L09: 组装自己完成的模块。")

    def encode(self, src_ids, src_allowed_mask=None):
        raise NotImplementedError("L09")

    def decode(self, memory, src_allowed_mask, tgt_input_ids, tgt_allowed_mask):
        raise NotImplementedError("L09")

    def project(self, x):
        raise NotImplementedError("L09")

    def forward(self, src_ids, tgt_input_ids, src_allowed_mask, tgt_allowed_mask):
        raise NotImplementedError("L09")
