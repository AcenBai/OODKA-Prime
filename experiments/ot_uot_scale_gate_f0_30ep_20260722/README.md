# OT/UOT Scale-Gate — Fold 0, 30 Epochs

## 实验身份

- 实验 ID：`ot_uot_scale_gate_f0_30ep_20260722`
- 原始运行 ID：`scale_gate_fold0_30ep_20260722_005035`
- 状态：已归档
- 数据集：`Dataset009_CT_OOD`
- Fold：`0`
- Epochs：`30`
- Seed：`42`
- 运行日期：`2026-07-22`
- 代码分支：`dev/OT_7_20`
- 绑定提交：`b3a4f9f9d191f9e76aff3398f5513754574a02d6`
- 建议标签：`exp-ot-uot-scale-gate-f0-30ep-20260722-v1`

> 注意：本次运行在 2026-07-22 约 00:50 开始，绑定提交在同日 15:21 创建。
> 因此这是对实验代码的事后归档绑定；现有记录无法独立证明运行开始时工作区完全干净。

## 实验目的

<!-- 在这里补充本次实验希望验证的假设、与上一版的差异。 -->

验证多尺度 P/S 特征解耦、P 分支 balanced OT、S 分支 unbalanced OT，以及 Beta scale gate 在 OODKA 蒸馏训练中的效果。

## 核心设置

- 训练样本：16 cases，702 blocks，4171 real slices
- Validation：4 cases，191 blocks，1134 real slices
- Batch size：1
- Block Z：6
- 输入大小：512
- Optimizer learning rate：`1e-4`
- Weight decay：`1e-4`
- P-OT weight：`0.1`
- S-OT weight：`0.1`
- P-OT start epoch：2
- S-OT start epoch：3
- OT warmup：5 epochs
- Sinkhorn iterations：30
- AMP：float16
- 物理 GPU：7；程序内设备：`cuda:0`

完整参数见 [`resolved_config.json`](resolved_config.json)。

## 运行入口

对应提交中的入口脚本：

```bash
RUN_TAG=scale_gate_fold0_30ep_20260722_005035 \
  bash scripts/run_scale_gate_30ep_gpu7.sh
```

该脚本依次执行训练、validation/test 整卷评估、Beta gate 分析、
P/S 特征分析和 UOT 压力测试。

## 主要结果

| Split | Cases | Mean Dice (GT present) |
|---|---:|---:|
| Validation | 4 | 0.901670 |
| Test | 20 | 0.889814 |

Test 各类别 Dice：

| Class | Dice |
|---:|---:|
| 1 | 0.891843 |
| 2 | 0.848231 |
| 3 | 0.928142 |
| 4 | 0.880382 |
| 5 | 0.878388 |
| 6 | 0.962610 |
| 7 | 0.839101 |

详细结果见 `metrics/`，训练和 OT 路由曲线见 `plots/`。

## 结论与备注

<!-- 在这里写对结果的解释、值得保留的发现、失败点和后续计划。 -->

- 当前版本已达到可作为后续对照实验和消融实验起点的水平。
- `student_deploy_best.pth` 是纯学生部署权重。
- `fusion_disentangle_best.pth` 包含训练期完整模块，适合分析和继续实验。
- `checkpoint_epoch030.pth` 用于恢复第 30 epoch 状态。

## 目录说明

```text
.
├── README.md
├── resolved_config.json
├── metrics/       # 汇总指标和逐病例指标
├── plots/         # 适合直接查看和纳入报告的曲线
├── models/        # 权重；本地保留，不进入普通 Git
├── logs/          # 原始运行日志；本地保留
├── analysis/      # Beta gate、P/S 特征分析及中间数组
├── predictions/   # validation/test 整卷预测
└── artifacts.sha256
```

`models/`、预测、日志和大型分析文件由本目录的 `.gitignore` 排除，
但其 SHA256 会记录在 `artifacts.sha256` 中。Git checkout 负责恢复代码、
配置、指标和说明；大型产物需从实验归档存储恢复后再校验。

## 原始位置

归档时原始运行目录为：

```text
outputs/oodka_ot_experiments/scale_gate_fold0_30ep_20260722_005035
```

本次整理采用复制方式，未删除或移动原始目录。
