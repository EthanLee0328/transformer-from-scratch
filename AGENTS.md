# Handwritten Transformer learning workspace

- 本仓库只收录自有代码、检查器、整理后的本人学习笔记和实验指标；不复制第三方教程实现、视频、字幕、课件、数据集或模型权重，不同步完整本地资料库。
- 目标是理解并手写完整 Encoder–Decoder Transformer，采用 Post-LN。核心模块不能用 nn.Transformer、MultiheadAttention 或封装的 attention API 替代；可使用 Linear、Embedding、LayerNorm、Dropout 等基础层。
- student/ 是本人练习区，默认保留 TODO。先请学习者解释、预测和尝试，再给分级提示；未经明确请求，不代写整道题或用参考答案覆盖。
- 区分 Codex 准备的脚手架、本人独立实现和实验观测。代理运行成功不等于本人掌握；待实现、失败和通过必须如实记录。
- 每次只做与当前问题有关的检查。Attention checker 的退出码2表示未实现，不冒充通过；源 padding mask 与目标 padding loss 是两个约定。
- 训练记录写 experiments/，解释和复盘写 notes/。不在公开文件里写凭据、个人录音、本机绝对路径或无关私人资料。
- 本仓库是独立 Git 根目录。提交前检查 git diff --cached，仅上传当前授权的自有变更。不要从外层学习资料目录执行全量上传，不创建自动上传任务。
