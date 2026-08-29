# OODKA 两阶段 ROI：实现、实验结果与当前诊断

更新日期：2026-08-25

当前分支：`codex/lge-roi-five-class`

> 本文是给后续讨论者（包括 ChatGPT）的代码与实验交接。结论必须以本文列出的实际实现和诊断实验为基础，不能把当前问题简单归因为“ROI 框不够准”。

## 1. 目标与实验背景

原始 CT/MRI OODKA 是单阶段全图七类分割：

```text
full volume -> contiguous Z blocks -> OODKA P/S fusion -> seven-class logits
```

WHS 七类为：

1. LV
2. RV
3. LA
4. RA
5. Myo
6. AO
7. PA

为了提高小结构分割质量，引入了两阶段 ROI：

```text
Pass 1: full block + one whole-heart prompt -> whole-heart probability
        -> threshold -> expanded block-level ROI

Pass 2: crop the same ROI from every slice in the block
        -> resize crop to model input resolution
        -> seven refinement prompts -> seven-class local logits
        -> restore logits to full-image coordinates
```

定位 prompt 是七类前景的并集：

```text
whole heart containing LV, RV, LA, RA, Myo, AO and PA
```

CT 使用 CT 文本，MRI 使用 MRI 文本；第二阶段继续使用原来的七个类别 prompt。

## 2. 当前“3D ROI”的精确定义

当前代码中的 3D/block ROI 并不是任意三维包围盒，也不裁剪 Z 轴。

设一个 batch 元素包含连续的 `Z=block_z` 个切片，第一阶段 whole-heart 概率为

\[
P\in[0,1]^{Z\times H\times W}.
\]

先屏蔽 padded tail slices，再沿 Z 取最大值：

\[
P_{\mathrm{block}}(x,y)=\max_{z\in\mathcal V}P(z,x,y).
\]

对 `P_block >= threshold` 的二维并集求包围框并扩张。得到的同一个 XY 框应用于该 block 的全部 Z 切片。因此它具有：

- block 内空间一致性；
- 与 `block_z` 直接关联；
- 不会出现逐切片 ROI 抖动；
- 但不会沿 Z 轴裁剪或扩张。

核心实现位于：

- `oodka/train/lge_roi_engine.py::_online_rois`
- `oodka/data/lge_roi.py::crop_and_resize_batch`
- `oodka/data/lge_roi.py::restore_roi_logits`

## 3. 训练过程

### 3.1 Warm-up

前 `roi_warmup_epochs` 只训练 Pass 1 whole-heart 定位分支。WHS 30-epoch 实验使用 10 epoch warm-up。

### 3.2 Mixed two-pass training

warm-up 后，每个 iteration 同时计算：

\[
\mathcal L
=\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
+\lambda_{\mathrm{refine}}\mathcal L_{\mathrm{refine}}.
\]

- `anchor`：全图 whole-heart prompt；
- `refine`：从预测 ROI 裁剪后的七类 prompt；
- 两个 pass 共享同一套 OODKA student P/S modules 和 router parameters；
- 每个 pass 都可包含 segmentation、reconstruction、orthogonality、route KL、P-OT 和 S-UOT；
- mixed 阶段 regularizer 按两个 pass 各乘 `0.5`，避免简单翻倍。

训练 ROI 来自离线预测 cache；验证和测试使用当前模型在线预测 ROI。`roi_refresh_every=0` 表示 warm-up 后生成一次 cache，正值表示定期刷新。

### 3.3 Prompt 数量归一化

第一阶段只有 1 个 prompt，第二阶段有 7 个 prompt。旧的 `sum` reduction 会导致第二阶段 segmentation loss 天然约为第一阶段的七倍。

新增 `prompt_mean`：先对每个 prompt 的有效样本取平均，再对有效 prompts 取平均。因此 1-prompt 和 7-prompt 分支具有可比较的 loss scale。对应测试：

```text
tests/test_prompt_loss_reduction.py
```

### 3.4 数据增强

增强是独立软开关 `lge_augment`/`--no_augment`。当前实现支持 `Z>1`：同一 block 的全部切片共享一个空间 affine transform，保证 Z 一致；强度扰动可逐切片采样。

WHS 30-epoch CT/MRI 3D ROI 实验开启了增强。ROI、增强与 relative KD 是彼此独立的开关。

## 4. 推理过程

WHS checkpoint 使用格式 `oodka_whs_roi_v1`。

推理对每个连续 Z block 执行：

1. 全图 Pass 1 预测 whole-heart；
2. 有效切片沿 Z 聚合，生成共享 XY ROI；
3. 裁剪每个 Z slice 的相同 XY 区域；
4. crop resize 回 512（CT）或 320（MRI）；
5. Pass 2 输出七类 logits；
6. logits resize 回 ROI 大小并恢复到全图；
7. ROI 外七类 logits 设为低值，背景胜出；
8. 恢复到原始 NIfTI geometry 后做七类 exclusive argmax。

诊断支持三种 ROI 来源：

- `predicted`：第一阶段预测框；
- `ground_truth`：七类 GT 并集的 oracle 框；
- `full`：整幅图作为 ROI，隔离 crop 是否有作用。

### 当前最重要的实现事实

WHS trainer 设置：

```python
refinement_only_output=True
```

因此最终七类结果完全来自 Pass 2。Pass 1 只学习一个 whole-heart 定位任务，不产生也不保存一个可用于兜底的全图七类结果。

这意味着当前结构不是：

```text
strong global seven-class prediction + local correction
```

而是：

```text
whole-heart localization + complete replacement by local seven-class model
```

## 5. 数据、teacher 与划分

### CT

- Dataset：`Dataset009_CT_OOD`
- teacher：Dataset009 对应 frozen nnUNet fold 0 checkpoint
- input：512，CT window level/width = 40/400
- pseudo-RGB：`adjacent`
- train/val：Dataset009 自己的 `splits_final.json`, fold 0
- independent test：Dataset009 `imagesTs/labelsTs`，20 cases

### MRI

- Dataset：`Dataset010_WHS_MRI_OOD`
- teacher：Dataset010 对应 frozen nnUNet fold 0 checkpoint
- input：320，MRI percentile normalization
- pseudo-RGB：`adjacent`
- train/val：Dataset010 自己的 `splits_final.json`, fold 0
- independent test：Dataset010 `imagesTs/labelsTs`，26 cases

两种模态都使用各自任务的固定 split；没有把 CT/MRI 的 train/val/test 病例交叉使用。

### 已修复的 CT preprocessing 问题

早期通用 ROI evaluator 对 CT 错用了 MRI percentile normalization。当前 `AlignedBiomedParsePreprocessor` 已按 `norm_mode` 分流：

- CT：40/400 window -> `[0,255]`；
- MRI：percentile normalization。

这是强度预处理问题，不是 NIfTI header 修复问题。VoxTell 中 `heart_2009/2017` 的 header repaired copies 属于另一条独立实验链，不应与这里混为一谈。

## 6. 实验配置

### CT 3D ROI

```text
experiment: whs_ct_roi3d_z4_b1_promptmean_30ep_20260824
epoch: 30
best epoch: 20
block_z: 4
batch_size: 1
image_size: 512
threshold: 0.3
expand: 1.25
prompt reduction: prompt_mean
pseudo-RGB: adjacent
augmentation: on
```

### MRI 3D ROI

```text
experiment: whs_mri_roi3d_z4_b2_promptmean_30ep_20260824
epoch: 30
best epoch: 25
block_z: 4
batch_size: 2
image_size: 320
threshold: 0.2
expand: 1.4
ROI cache refresh: every 5 epochs
prompt reduction: prompt_mean
pseudo-RGB: adjacent
augmentation: on
```

## 7. 独立测试结果

| Experiment | Test mean Dice | Comparison |
|---|---:|---:|
| CT no-ROI historical baseline | 0.88766 | reference |
| CT old per-slice/2D ROI | 0.85362 | -3.40 pp |
| CT block-level 3D ROI | 0.85524 | -3.24 pp |
| MRI no-ROI historical baseline | about 0.67 | reference |
| MRI old per-slice/2D ROI | 0.62389 | about -4.61 pp |
| MRI block-level 3D ROI | 0.63995 | about -3.01 pp |

结论：block-level 3D ROI 相对旧 2D ROI 有小幅改善，但 CT/MRI 都没有超过 no-ROI。

### CT 3D ROI per class

| LV | RV | LA | RA | Myo | AO | PA | Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8586 | 0.7688 | 0.9189 | 0.8183 | 0.8473 | 0.9482 | 0.8265 | 0.8552 |

### MRI 3D ROI per class

| LV | RV | LA | RA | Myo | AO | PA | Mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7874 | 0.6362 | 0.8129 | 0.7504 | 0.6268 | 0.4416 | 0.4245 | 0.6399 |

### 为什么 LGE 曾经从 ROI 获益

LGE four-class ROI-v2 test `mean_dice_gt_present` 约为 0.65055，而对应 flat four-prompt full-image run 约为 0.50646。LGE 目标集中在小面积 myocardium ROI，放大后局部分辨率收益明显；CT/MRI whole-heart 七类本身占据更大范围，裁剪带来的分辨率收益较小，同时全局上下文和尺度改变成本更高。

这说明“ROI 方法普遍有效”并不成立；收益依赖目标大小、上下文需求、最终融合方式和任务定义。

## 8. CT 三组关键诊断

使用同一个 CT 3D ROI best checkpoint：

| ROI source | Mean Dice | 含义 |
|---|---:|---|
| predicted ROI | 0.85524 | 实际两阶段推理 |
| GT/oracle ROI | 0.85828 | 排除定位框误差后的上限 |
| full-image ROI | 0.78105 | 保留 refinement branch，但取消 crop |
| old no-ROI model | 0.88766 | 原全图七类模型 |

关键差值：

- Oracle - predicted：`+0.30 pp`；
- Oracle - no-ROI：`-2.94 pp`；
- Full ROI - predicted：`-7.42 pp`；
- Full ROI - no-ROI：`-10.66 pp`。

ROI 定位统计：

```text
CT predicted ROI final GT coverage: 0.98599
CT threshold-mask Dice:             0.86526
CT mean ROI area fraction:          0.50202

MRI predicted ROI final GT coverage: 0.98820
MRI threshold-mask Dice:              0.83723
MRI mean ROI area fraction:            0.26193
```

虽然个别病例（例如 CT `heart_2010`）会被预测框明显伤害，但 oracle ROI 在整个测试集上只提高 0.30 pp。因此平均性能缺口主要不是 ROI 框定位造成的。

## 9. 当前失败原因排序

### 9.1 主因：Pass 2 完整替换，而不是增量 refinement

oracle ROI 仍然低于 no-ROI，说明即使给完美框，当前 refinement model 也弱于旧全局七类模型。

Pass 1 只学习 whole-heart，并没有一个全局七类输出可以保留。Pass 2 必须独自完成全部七类判别，因此“加 ROI”实际上替换了原先已经较强的任务，而不是在它上面做修正。

### 9.2 crop/resize 改变了任务分布

ROI crop 被 resize 回固定输入大小，带来：

- 解剖尺度变化；
- 插值误差；
- 全局位置和周围结构上下文丢失；
- 不同病例/blocks 的缩放比例波动。

`full ROI = 0.78105` 又说明 refinement checkpoint 已明显依赖 crop 放大；crop 对这个弱 refinement branch 有帮助，但仍不足以超过原全图模型。

### 9.3 共享参数存在双任务干扰

whole-heart localization 和 seven-class refinement 共用同一套 P/S decomposition 与 router。一个分支是 1 prompt 的全图任务，另一个是 7 prompts 的局部尺度任务。`prompt_mean` 修正了 loss 数量级，却不能消除表征和尺度冲突。

### 9.4 cached training ROI 与 online inference ROI 的差异

CT `roi_refresh_every=0`，warm-up 后训练 cache 不再更新；验证/测试使用在线预测框。该差异可能造成额外损失，但 oracle 诊断说明它不是约 3 pp 缺口的主因。

### 9.5 ROI 外强制背景

恢复后的 refinement logits 在 ROI 外设为低值。少量漏框必然变成假阴性。当前 coverage 接近 99%，所以这是次要但真实存在的风险。

## 10. 下一版 ROI 的建议方向

不要继续把主要精力放在 threshold/expand/LCC 上。首先改变最终建模方式：

### 10.1 保留全局七类预测

第一阶段应同时保留或单独运行一个 global seven-class model：

\[
Z_{\mathrm{global}}\in\mathbb R^{7\times Z\times H\times W}.
\]

ROI branch 只预测局部修正：

\[
Z_{\mathrm{final}}
=Z_{\mathrm{global}}+M_{\mathrm{roi}}\odot\Delta Z_{\mathrm{roi}},
\]

或者做置信度门控：

\[
Z_{\mathrm{final}}
=(1-g)\odot Z_{\mathrm{global}}+g\odot Z_{\mathrm{roi}}.
\]

ROI 外严格保留 global；ROI 内只有当 local branch 更可信时才覆盖。

### 10.2 推荐 loss

\[
\mathcal L
=\mathcal L_{\mathrm{global-7}}
+\lambda_{\mathrm{roi}}\mathcal L_{\mathrm{roi-7}}
+\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{consistency}}.
\]

其中 consistency 可以约束恢复到全图后的 ROI logits 与 global logits 在可靠区域一致，或者只学习 residual correction。

### 10.3 必须保留的诊断矩阵

每个新版本至少同时报告：

1. no-ROI global；
2. predicted ROI；
3. oracle ROI；
4. full ROI；
5. global + ROI fusion；
6. 每类 Dice、case-wise delta、ROI coverage/area/fallback。

## 11. Expert-side relative KD：独立于 ROI 的当前改动

旧 expert-side decomposition 只有 reconstruction 和 decorrelation。即使 P/S 非零且不复制，也存在无语义的任意互补拆分。

新增独立开关：

```text
relative_kd=False/True
relative_kd_expert_weight=1.0
```

每个 iteration 先在 `no_grad` 下计算 correspondence：

\[
\pi^{(t)}=\operatorname{Transport}(S^{(t)},E^{(t)}).
\]

固定该 correspondence，保留原 expert-to-student KD，并使用同一个 transport 的转置增加 student-to-expert KD：

\[
\mathcal L_{E\to S}
=d\left(S,\operatorname{sg}(\pi E)\right),
\]

\[
\mathcal L_{S\to E}
=d\left(E,\operatorname{sg}(\pi^\top S)\right).
\]

transport 和 teacher values 均 detach；当前 iteration 更新两侧 adapter，下一 iteration 用更新后的特征重算 transport。P-balanced OT 和 S-unbalanced OT 都支持双向损失。

它与 ROI、augmentation 是三个独立开关。当前正在运行的 CT ablation：

```text
spatial_beta_p07_ct_f0_relativekd_no_roi_noaug_30ep_20260825
```

只开启 relative KD，关闭 ROI 和 augmentation，其他设置复刻旧 CT no-ROI 30-epoch baseline。该实验用于验证 expert P/S 语义性与最终 Dice 是否改善，不应被当作已经解决 ROI 替换问题。

## 12. 当前代码地图

| 文件 | 作用 |
|---|---|
| `run_train_whs_roi.py` | CT/MRI WHS 两阶段训练入口 |
| `oodka/models/prompts.py` | CT/MRI whole-heart localization 与七类 refinement prompts/groups |
| `oodka/train/lge_roi_engine.py` | 通用 two-pass trainer；warm-up、ROI cache、mixed loss、checkpoint |
| `oodka/data/lge_roi.py` | ROI 生成、block-coherent crop/restore、visibility、augmentation |
| `run_eval_lge_roi.py` | LGE/WHS 通用 ROI evaluation；predicted/oracle/full diagnostics |
| `oodka/data/aligned_preprocessing.py` | nnUNet geometry-aligned CT/MRI preprocessing |
| `oodka/train/forward.py` | prompt loss reduction、OODKA forward 与 loss aggregation |
| `oodka/models/ot/objective.py` | P-OT/S-UOT 与可选 bidirectional relative KD |
| `run_train.py` | 标准 no-ROI trainer，支持 `--relative_kd` |

## 13. 可复现实验脚本

```text
shell/run_whs_ct_roi3d_z4_b1_30ep_20260824.sh
shell/run_whs_mri_roi3d_z4_b2_30ep_20260824.sh
shell/eval_whs_ct_roi3d_best_test_20260824.sh
shell/eval_whs_mri_roi3d_best_test_20260824.sh
shell/diagnose_whs_ct_roi3d_best_test_20260824.sh
shell/run_ct_relative_kd_no_roi_noaug_30ep_20260825.sh
```

## 14. 测试状态

当前 repository tests：`45 passed`（2026-08-29 全量回归）。

新增覆盖包括：

- Z-block coherent crop/restore；
- ROI visibility；
- 3D augmentation shape/coherence；
- CT/MRI prompt/group mapping；
- `prompt_mean` 对 prompt 数量不敏感；
- relative KD 关闭时 expert 无 KD gradient；
- relative KD 开启时 student/expert P/S 均有有效 gradient。

## 15. 后续讨论应优先回答的问题

1. global seven-class branch 应该使用独立模型、共享 backbone，还是共享 P/S adapters？
2. ROI branch 更适合作为 residual logits、feature residual，还是 confidence-gated expert？
3. 如何初始化 ROI branch，使 `g=0` 时严格退化到已知 no-ROI baseline？
4. consistency 应作用在 logits、probabilities、features，还是边界区域？
5. 如何保证任何 ROI failure 都不会让结果低于 global baseline？
6. relative KD 是否改善 expert P/S 的 activation semantics，而不仅是 Dice？

当前最可靠的总判断是：

> ROI localization 已经基本可用；CT/MRI 的主要问题是 local refinement 完整替换了一个更强的 global seven-class predictor。下一版应把 ROI 设计成保底全局预测上的局部增量，而不是另起一个必须独自完成任务的替代模型。

## 16. 2026-08-29 实验结果与最新诊断

### 16.1 MRI great-vessel bridge ROI

新策略的 Pass 1 为六类全图预测：LV、RV、LA、RA、Myo、GV，其中
GV 是 AO 与 PA 的并集，只负责产生 ROI；Pass 2 在 ROI 内细分完整七类，
ROI 外保留前五个全图类别。配置为 `B=2, Z=4`、adjacent pseudo-RGB、
augmentation、prompt-mean loss、30 epochs。

独立 test（26 cases，predicted ROI，spatial hard switch）：

| class | Dice |
|---|---:|
| LV | 0.8414 |
| RV | 0.6549 |
| LA | 0.8311 |
| RA | 0.7490 |
| Myo | 0.6409 |
| AO | 0.4073 |
| PA | 0.4310 |
| mean | **0.6508** |

它高于旧 3D whole-heart ROI 的 `0.6399`，但仍低于约 `0.67` 的
historical no-ROI baseline。GV threshold mask Dice 为 `0.4346`，precision
为 `0.3169`，expanded ROI 的 GT recall 约为 `0.9536`，full-image fallback
rate 为 `0.2613`。结论是 GV bridge 有小幅改善，但仍未实现 ROI 正增益，
AO/PA 是主要瓶颈。

### 16.2 CT great-vessel bridge ROI

`B=1, Z=4, 512x512` 在 epoch 10 warm-up 后切换到同时保留 full-image
anchor 和 ROI refinement 两套反向图，训练进程峰值约 `20.50 GiB`。
当时 GPU 另有约 3 GiB 常驻进程，epoch 11 申请额外 448 MiB 时 OOM。
warm-up epoch-10 六类 validation Dice 为 `0.8974`，但没有保存 checkpoint，
不能续训。已提供在完全空闲 GPU 上从头重跑 Z4/B1 的手动脚本。

### 16.3 Relative KD CT

关闭 ROI 和 augmentation、只开启 bidirectional relative KD 的 CT 30-epoch
实验已完成。独立 test mean Dice 为 `0.89318`；旧 no-KD adjacent diagnostic
为 `0.88766`，绝对提升约 `0.00552`。性能有小幅收益，但 Expert P/S 的
可视化仍主要表现为区域内部/边缘、高频/低频的互补分解，没有形成预期的
语义解耦。

KD 日志中的 `P/S` 是 forward 与 reverse 的合计。拆分结果如下：

| epoch | P forward | S forward | P reverse | S reverse |
|---:|---:|---:|---:|---:|
| 5 | 0.2023 | 0.2788 | 0.1345 | 0.1715 |
| 10 | 0.1241 | 0.1193 | 0.0697 | 0.0514 |
| 20 | 0.0886 | 0.0753 | 0.0433 | 0.0214 |
| 30 | 0.0795 | 0.0656 | 0.0378 | 0.0182 |

epoch 30 的 raw forward/reverse 比约为 `2.59:1`。乘以 `wP=wS=0.1`
后，作用于 Expert 的 reverse KD 约为 `0.0056`，只占 validation total
loss `0.5094` 的约 1.1%。transport 数值稳定且 loss 正常下降，但当前
cosine KD 只约束每个空间 token 的通道方向，不约束 channel-RMS energy、
P/S energy share 或空间关系，因此 KD 收敛不等价于 Expert energy maps
语义化。

### 16.4 Expert 平凡解的结构性原因与候选修正

Student 使用两个无 bias、无 branch normalization 的 1x1x1 projections，
直接满足 `P_b + S_b ~= Z_b`，并经过 frozen BiomedParse pixel decoder、
prompt embedding 和 spatial router 接收 segmentation gradient。Expert 则
经过公共 Conv-IN-GELU、两个独立 Conv-IN branch heads，以及两个独立可学习
decoder，满足 `R_p(P_e) + R_s(S_e) ~= Z_e`。Expert P/S 不进入最终 predictor。

因此 Expert 存在独立通道旋转/缩放可由各自 decoder 逆向补偿的 gauge
freedom；orthogonality 只能鼓励互补，不能确定 branch semantics。res2--res4
的独立 branch InstanceNorm 还会把 P/S 分别标准化到相近方差，容易产生接近
0.5 的 S-share 和边缘/内部互补解。

在不让 Expert 承担分割任务、继续以 BiomedParse 为轴体的前提下，建议按顺序
做以下受控消融：

1. 冻结 Student decomposers 与 Beta router，关闭 forward KD，只更新 Expert；
2. 单独将 reverse KD 权重提高到 3--5，transport 仍逐 iteration 重算并 detach；
3. 将 Expert orthogonality 权重从 0.3 暂降到 0.05--0.1；
4. 去掉 res2--res5 Expert branch heads 各自的 InstanceNorm，仅保留公共 adapter norm；
5. 增加 transported S-share/energy matching 与 spatial relational KD，以补足 cosine
   对能量和空间关系不敏感的问题；
6. 若仍存在平凡解，将两个独立 decoder 改为共享 `R(P_e+S_e)`，并增加
   aligned-space `P_e+S_e ~= U_e`，限制独立 decoder 的补偿自由度。

优先记录每层 forward/reverse KD、Expert adapter gradient norm、P/S RMS、
S-share 均值/方差、cross-branch affinity 和 frozen linear-probe semantics，
避免继续只根据总 loss 判断语义传递是否成功。
