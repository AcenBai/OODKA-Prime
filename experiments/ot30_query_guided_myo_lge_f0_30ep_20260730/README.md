# OT/30 query-guided S injection — MYO LGE OOD

## 实验身份

- 实验 ID：`ot30_query_guided_myo_lge_f0_30ep_20260730`
- 状态：已完成训练与 OOD 测试
- 运行日期：2026-07-30
- 数据集：`Dataset011_MYO_LGE_BC_OOD`
- Fold / Seed：`0 / 42`
- 代码分支：`OT/30`
- 绑定提交：`cb2fa9428544c5e1b8e08454f4c35042f93b330e`
- GPU：`cuda:2`

## 核心设置

- LGE MRI 前景内 1–99 分位归一化到 0–255。
- BiomedParse 使用与 nnUNet 一致的 crop/resample 几何。
- 输入尺寸：`256 × 256`
- `block_z=2`，`batch_size=8`
- 5 个 LGE 专用 prompts。
- P-initial-proposal guided S residual，`topk=4`，`S floor=0.2`。
- Beta router 关闭。
- 训练 30 epochs，每 5 epochs 验证。

完整参数见 `resolved_config.json`，完整终端记录见
`train_console.log` 与 `test_eval_console.log`。

## 主要结果

- 最佳验证：Epoch 20，mean Dice `0.560283`
- Epoch 30 验证：mean Dice `0.540572`
- 45 例 LGE-OOD 测试：mean Dice `0.266673`

测试分类型 Dice：

- Scar：`0.017003`
- Edema：`0.129855`
- LV blood pool：`0.705851`
- LV normal myocardium：`0.310111`
- RV blood pool：`0.161811`

## 初步观察

训练集指标持续提高，但 Epoch 20 后验证指标回落，存在过拟合。OOD
测试中血池结构的迁移明显好于 Scar、Edema 和 RV blood pool。此处仅记录
事实结果，后续机制解释与改进结论由实验审计补充。

## 产物

- 最佳完整恢复模型：`fusion_disentangle_best.pth`
- Epoch 30 完整恢复模型：`checkpoint_epoch030.pth`
- 最佳 Student 部署模型：`student_deploy_best.pth`
- Epoch 30 Student 部署模型：`student_deploy_epoch030.pth`
- 验证摘要：`metrics/val_summary.json`
- 测试摘要：`metrics/test_summary.json`
- 逐病例测试指标：`metrics/test_metrics.csv`
- 原始预测：`test_eval/pred_nii/`（不进入 Git）
- 校验：`sha256sum -c artifacts.sha256`
