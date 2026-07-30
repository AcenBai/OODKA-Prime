# Spatial Beta 路由与 LGE OOD

## 路由定义

冻结的 prompt `class_emb` 经过一个小型 MLP，预测平滑二维 RBF
基函数的系数，在归一化坐标 `[-1, 1] × [-1, 1]` 上生成每个 prompt
的一张最高分辨率 Beta 分布场：

```text
class_emb[p] -> MLP -> alpha/beta spatial coefficients
                         |
                         v
              beta[p, y, x] in [0, 1]
```

训练时使用可重参数化的 Beta 采样，推理时使用分布均值。初始先验为
`Beta(7, 3)`，所以 P 分支权重均值为 `0.7`。Pixel Decoder 的 mask
feature 使用原始门控图，其他多尺度特征只从同一张图做 area
downsample，不再为不同层分别学习路由。

每个位置保持凸组合：

```text
F[p, l, y, x] =
    beta[p, l, y, x] * P[l, y, x]
    + (1 - beta[p, l, y, x]) * S[l, y, x]
```

归一化坐标只负责让参数化与输入分辨率无关。路由在前向时直接按实际
Pixel Decoder 尺寸生成张量，因此 CT 512 和 LGE 256 不需要手工做
坐标反变换。

## Dataset011 LGE 对齐预处理

LGE 使用 5 个专用 prompt，并在 MRI 前景内按 1--99 分位映射到
0--255。BiomedParse 与 nnUNet 共用 nnUNet plans 的 crop/resample
几何，避免师生特征错位。

首次运行先生成离线对齐缓存：

```bash
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python \
  scripts/preprocess_biomedparse_aligned.py \
  --dataset_name Dataset011_MYO_LGE_BC_OOD \
  --configuration 2d \
  --split all \
  --output_dir /path/to/biomedparse_preprocessed_Dataset011
```

训练的关键参数为：

```bash
--dataset_name Dataset011_MYO_LGE_BC_OOD \
--norm_mode mri \
--image_size 256 \
--block_z 2 \
--batch_size 8 \
--raw_cache_cases 4 \
--biomedparse_preproc_dir /path/to/biomedparse_preprocessed_Dataset011
```

长训练可启用线性 warm-up 后的 cosine 学习率：

```bash
--lr_schedule cosine \
--lr_warmup_epochs 5 \
--min_lr_ratio 0.05
```

每次训练写出的 `resolved_config.json` 会记录 `source_commit`、
`source_branch` 和 tracked-file dirty 状态，从而把实验和可恢复代码
版本一一绑定。
