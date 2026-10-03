# Transformer from Scratch

用费曼学习法，亲手理解并用 PyTorch 实现 Encoder–Decoder Transformer。

目标是能自己讲清数据流、预测张量形状、手写核心模块，最后完成训练与逐 token 生成。实现目标采用 Post-LN。

**当前是练习框架，核心模型保留 TODO，尚未实现。** 初始脚手架和检查器由 Codex 协助准备；后续提交记录实际的手写实现与实验，不把参考程序跑通计作本人掌握。

## 目录

| 目录 | 内容 |
|---|---|
| `student/` | 自己实现 Attention、多头、位置编码、Encoder、Decoder 与完整模型 |
| `checks/` | 数值、形状、Mask 和梯度检查 |
| `scripts/` | 自有的小型学习实验 |
| `docs/roadmap.md` | 按能力推进的实现路线 |
| `notes/` | 整理后可公开的本人解释与纠错笔记 |
| `experiments/` | 自有实验的配置、指标和复盘 |

这里只保存自有代码与整理后的记录。第三方教程、参考实现、视频、字幕、课程讲义、数据集和模型权重不放进仓库。

## 开始运行

准备 Python 3.11 环境，安装最小依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 先写下预测；核对时再加 --reveal。
python scripts/tensor_diagnostic.py
python scripts/tensor_diagnostic.py --reveal

# 完成自己的 attention 后运行检查。
python checks/check_attention.py
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。Attention 检查使用 CPU float32；MPS/CUDA 训练实验在完成模型后另行记录。

Attention 的约定是 `bool mask`，`True` 表示允许关注。尚未实现时检查返回 **2 / NOT_IMPLEMENTED**；实现错误返回 **1**，环境错误返回 **3**，检查通过返回 **0**。TODO 状态不是检查成功。

## 每次提交什么

先解释问题并预测，再写代码、检查结果和复盘。每次提交一个能说明目的的小步骤；记录真实结果，不写成已经学会尚未独立完成的内容。

```bash
git status --short
git add student/ checks/ scripts/ docs/ notes/ experiments/ README.md requirements.txt
git diff --cached
git commit -m "Implement one Transformer learning step"
git push origin main
```

提交前查看暂存差异，确认是本次的自有代码和记录。训练数据、私人路径与完整学习资料库留在本地；新指标可写成小型 JSON/CSV，模型权重不提交。
