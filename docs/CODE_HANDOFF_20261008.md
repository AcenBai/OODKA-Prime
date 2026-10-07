# OODKA 代码交接地图

入口与数据流

| 入口                     | 用途                                        | 主要实现                                                                                                        |
| ------------------------ | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `run_train.py`         | 不使用 ROI 的全图训练                       | `oodka/train/engine.py`                                                                                       |
| `run_train_whs_roi.py` | CT/MRI whole-heart 或 great-vessel ROI 训练 | `oodka/train/lge_roi_engine.py`、`lge_roi_lifecycle.py`                                                     |
| `run_train_lge_roi.py` | LGE ROI v1/v2/v3 或 flat 消融               | 同上                                                                                                            |
| `run_eval_oodka.py`    | 全图 student-only 评估                      | `oodka/eval/eval_oodka.py`                                                                                    |
| `run_eval_lge_roi.py`  | LGE/WHS 离线 ROI 评估                       | `oodka/eval/roi_checkpoint.py`、`roi_inference.py`、`roi_block_diagnostics.py`、`roi_case_reporting.py` |

训练样本由 `oodka/data/slice_dataset.py` 组织成连续 Z-block，同时提供 nnUNet 和 BiomedParse 两路输入。训练时 nnUNet 是冻结 teacher；BiomedParse backbone 也是冻结的，`oodka/train/forward.py` 负责一次 batch 的编排，细节在 `forward_components.py`；OT、relative KD 与 capacity-partial S 的实现位于 `oodka/models/ot/`。评估只加载 BiomedParse student 所需模块，不加载 nnUNet teacher。

ROI 的一个 block 共用一个 XY 框，不裁 Z。`oodka/data/roi_geometry.py` 管 crop/pad/letterbox 和可逆回填；`roi_policy.py` 管预测框、GT 框、jitter、可见性和空间合成；`roi_augmentation.py` 管同步增强；`roi_cache.py` 管训练预测框缓存。

## 软开关及联动

| 分组       | 开关/字段                                                                                                   | 当前语义                                                                                                                                   |
| ---------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 推荐默认   | `--relative_kd`、`--s_transport_mode capacity_partial`                                                  | 三个训练入口默认开启 relative KD + capacity S；恢复旧 checkpoint 时，全图入口优先沿用 checkpoint 的相关配置。                              |
| ROI 来源   | `--roi_train_source predicted/ground_truth/full`                                                          | 预测框是可部署训练路径；GT 框是 Oracle 对照；full 是无裁剪对照。评估对应`--roi_source`，默认 predicted。                                 |
| ROI 几何   | `--roi_transform resize/pad/letterbox`                                                                    | 默认仍为resize；pad 保留模型画布上的像素尺度，图像补 0、训练标签补`-1` 并从分割损失排除。评估默认 `checkpoint`，从权重配置读几何模式。 |
| 定位与缓存 | `--warmup_epochs`、`--roi_threshold`、`--roi_expand`、`--roi_fallback`、`--roi_refresh_every`     | warmup 后才进入 mixed two-pass；预测框训练使用缓存，`roi_refresh_every=0` 表示只生成一次；验证/评估在线生成。                            |
| 任务版本   | WHS`--roi_strategy whole_heart/great_vessel`；LGE `--v2`、`--split_pathology`、`--flat_four_prompt` | 决定 prompt 分组、最终类别以及 ROI 内外合成规则，不是可以任意混搭的独立开关。`split_pathology` 要求 `v2`，flat 关闭 two-pass。         |
| 损失/增强  | `--roi_prompt_loss_reduction`、`--augment/--no_augment`                                                 | 现在 CT/MRI/LGE 共用参数定义。WHS 增强默认开；LGE 默认随`v2` 变化，显式开关优先。                                                        |
|            |                                                                                                             |                                                                                                                                            |

WHS whole-heart 默认 ROI 内七类完整替换；WHS great-vessel 使用空间合成。

## 复现与交接警戒线

- 当前常规画布：CT `512×512`、WHS MRI `320×320`、LGE `256×256`

快速回归：

```bash
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python -m pytest -q
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python run_train_whs_roi.py --help
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python run_eval_lge_roi.py --help
/data4/baihexiang/conda_envs/biomedparse_v2/bin/python scripts/visualize_mechanism_v3.py --help
```

相关历史结果见 `docs/MRI_ROI_STAGE_REPORT_20260921.md`。该报告里的“运行中”状态是报告当日快照，不应当作今天的任务状态。
