# Text-conditioned spatial Beta router — CT OOD

## 实验身份

- 实验 ID：`spatial_beta_p07_ct_f0_30ep_20260730`
- 状态：运行中
- 运行日期：`2026-07-30`
- 数据集：`Dataset009_CT_OOD`
- Fold / Seed：`0 / 42`
- 代码分支：`exp/spatial-combination`
- 绑定提交：`44e2adb`
- GPU：`cuda:2`

## 实验目的

验证 text-only prompt-specific 空间 Beta 路由。每类只学习一张最高分辨率
门控，各 Predictor 层通过 area downsample 共享该门控，并用
`beta * P + (1-beta) * S` 做逐像素凸组合。

## 核心设置

- P 路由先验均值：`0.7`
- Epoch / batch / block-z / image：`30 / 1 / 4 / 512`
- LR：`1e-4`，2 epoch warm-up 后 cosine decay，末端比例 `0.1`
- Route warm-up：`5`
- P-OT：epoch 2 开始；S-UOT：epoch 3 开始；OT warm-up：`5`

训练命令与全部参数以 `resolved_config.json` 为准；后台启动输出见
`launcher.log`，训练日志位于 `logs/`。
