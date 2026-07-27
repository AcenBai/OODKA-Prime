# OODKA Experiment Registry

每次正式实验使用一个独立目录，目录名同时作为稳定的实验 ID：

```text
experiments/<method>_<variant>_f<fold>_<epochs>ep_<YYYYMMDD>/
```

同一天同名实验可追加 `_r2`、`_r3`，不要覆盖已有实验目录。

## Git 与实验的关系

每个实验必须在自己的 `README.md` 中记录完整 `source_commit`。

- 一个代码提交可以对应多个实验，例如不同 fold、seed 或超参数。
- 一个实验只能绑定一个代码提交。
- 如果代码发生变化，先提交代码，再启动新实验。
- 仅改变命令行参数或配置时，可以复用同一个代码提交，但必须创建新的实验目录。
- 实验完成后，再提交 README、配置、指标、曲线和校验清单。

因此通常存在两类提交：

1. **代码提交**：实验启动前创建，负责恢复代码。
2. **实验记录提交**：实验结束后创建，负责保存配置、指标和说明。

正式里程碑实验可以再创建 annotated tag。普通调试和 smoke test 不需要打 tag。

## 推荐流程

```bash
# 1. 实验前确认代码状态
git status

# 2. 如果代码有变化，先提交代码
git add <本次实验相关代码>
git commit -m "feat(ot): describe the experiment code change"

# 3. 记下实验代码提交
git rev-parse HEAD

# 4. 运行实验，并创建独立实验目录

# 5. 实验完成后提交小型实验记录
git add experiments/<experiment_id>
git commit -m "exp(ot): archive <experiment_id>"

# 6. 重要结果再打标签
git tag -a <experiment_tag> -m "<short experiment conclusion>"
```

## 每个实验目录的推荐结构

```text
<experiment_id>/
├── README.md
├── .gitignore
├── resolved_config.json
├── metrics/
├── plots/
├── models/
├── logs/
├── analysis/
├── predictions/
└── artifacts.sha256
```

默认进入 Git：

- README 和实验结论
- resolved config
- 汇总及逐病例指标
- 报告用曲线
- 小型 JSON 分析结果
- SHA256 清单

默认不进入普通 Git：

- `.pth`、`.pt`、`.ckpt` 等模型
- 完整训练日志
- NIfTI 预测
- 大型 `.npz` 中间特征
- 大型分析图片

这些大文件仍放在相应实验目录中，并应备份到第二个存储位置。恢复后可在实验目录运行：

```bash
sha256sum -c artifacts.sha256
```

确认归档没有损坏或拿错版本。
