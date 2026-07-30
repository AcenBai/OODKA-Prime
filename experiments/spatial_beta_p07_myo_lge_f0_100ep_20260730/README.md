# Text-conditioned spatial Beta router — LGE OOD

## 实验身份

- 实验 ID：`spatial_beta_p07_myo_lge_f0_100ep_20260730`
- 状态：运行中
- 运行日期：`2026-07-30`
- 数据集：`Dataset011_MYO_LGE_BC_OOD`
- Fold / Seed：`0 / 42`
- 代码分支：`exp/spatial-combination`
- 绑定提交：`44e2adb`
- GPU：`cuda:4`

## 实验目的

在 nnUNet crop/resample 完全对齐的 LGE 路径上验证同一空间 Beta
路由，重点观察 Scar 与 Edema。MRI 在前景内用 1--99 分位映射至
0--255，并使用 5 类 LGE 专用 prompts。

## 核心设置

- P 路由先验均值：`0.7`
- Epoch / batch / block-z / image：`100 / 8 / 2 / 256`
- LR：`1e-4`，5 epoch warm-up 后 cosine decay，末端比例 `0.05`
- Route warm-up：`10`
- P-OT：epoch 5 开始；S-UOT：epoch 10 开始；OT warm-up：`15`
- 对齐缓存：`Distangler3/distangler3_output/biomedparse_preprocessed_Dataset011_MYO_LGE_BC_OOD`

训练命令与全部参数以 `resolved_config.json` 为准；后台启动输出见
`launcher.log`，训练日志位于 `logs/`。
