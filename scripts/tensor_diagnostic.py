#!/usr/bin/env python3
"""L00：先预测，再用 --reveal 核对；不能把运行结果当成掌握证明。"""

import argparse
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reveal", action="store_true", help="在预测后显示实验观测")
    args = parser.parse_args()
    import torch
    print("L00：请先口头预测以下 4 项，并说明理由。")
    print("1. x=arange(24).reshape(2,3,4)：x.shape、x[1,0,0]、x.transpose(1,2).shape？")
    print("2. q.shape=(2,3,4)，k.shape=(2,5,4)：q @ k.transpose(-2,-1) 的形状？")
    print("3. logits=[[0, ln(3)]]：softmax(dim=-1) 的两个权重及行和？")
    print("4. x=[1,2,3], loss=sum(x*x)：backward 后 x.grad 是什么？")
    if not args.reveal:
        print("PREDICTION_PENDING：写下预测，再加 --reveal 运行；没有完成/通过判定。")
        return
    x = torch.arange(24).reshape(2, 3, 4)
    print("观测 1:", tuple(x.shape), x[1, 0, 0].item(), tuple(x.transpose(1, 2).shape))
    q, k = torch.zeros(2, 3, 4), torch.zeros(2, 5, 4)
    print("观测 2:", tuple((q @ k.transpose(-2, -1)).shape))
    weights = torch.softmax(torch.tensor([[0.0, math.log(3.0)]]), dim=-1)
    print("观测 3:", weights.tolist(), "行和:", weights.sum(-1).tolist())
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    values.square().sum().backward()
    print("观测 4:", values.grad.tolist())
    print("请解释一项预测差异，并换一个维度或数值重新预测；脚本不替你判定掌握。")


if __name__ == "__main__":
    main()
