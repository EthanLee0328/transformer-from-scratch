"""L02–L03：手写 scaled dot-product attention。

先用纸笔解释：一个 query 得到的一行权重，最后怎样变成输出向量？
再预测 checks/check_attention.py 中已知算例的结果，然后实现本函数。
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    allowed_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[Tensor, Tensor]:
    """实现合同（不要调用 torch 的封装 attention）。

    q: (..., query_len, d_k)，k: (..., key_len, d_k)
    v: (..., key_len, d_v)。前导 batch/head 维须可广播。
    返回 output (..., query_len, d_v) 与 weights (..., query_len, key_len)。

    分数为 Q K^T / sqrt(d_k)，softmax 沿 key_len 维。
    allowed_mask 必须为 bool，且可广播到分数形状；True=允许关注，
    False=屏蔽。必须在 softmax 之前屏蔽；每个 query 至少有一个合法
    key，否则抛 ValueError。None 表示全部允许。

    例如 causal mask: torch.ones(L, L, dtype=torch.bool).tril()
    key padding mask: (src_ids != PAD)[:, None, None, :]。
    组合 mask 用逻辑 AND；padding mask 只屏蔽 key，不自动忽略 PAD loss。

    dropout 作用于 softmax 后的权重，仅 training=True 时启用。
    返回用于乘 V 的权重；dropout=0 时每行权重和为 1。
    保持 dtype/device；保持 q/k/v 的梯度；不要 detach 或转 numpy。
    本合同检查仅用 float32 与 dropout_p=0；其他实现细节可逐步增加。
    """
    raise NotImplementedError("L02–L03: 请先预测算例，再亲手实现 attention。")
