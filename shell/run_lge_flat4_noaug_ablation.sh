#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/lge_flat4_noaug_z1_b18_30ep_20260818}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

"${python_bin}" run_train_lge_roi.py \
  --v2 \
  --flat_four_prompt \
  --no_augment \
  --device cuda:2 \
  --n_epochs 30 \
  --warmup_epochs 10 \
  --batch_size 18 \
  --num_workers 4 \
  --val_every_epochs 5 \
  --test_best_on_improvement \
  --best_test_device cuda:1 \
  --output_dir "${experiment_dir}"

checkpoint="${experiment_dir}/fusion_lge_flat_best.pth"
"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split val \
  --device cuda:2 \
  --batch_size 18 \
  --decision auto \
  --out_dir "${experiment_dir}/eval_val_best"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device cuda:2 \
  --batch_size 18 \
  --decision auto \
  --out_dir "${experiment_dir}/eval_test_best"

"${python_bin}" scripts/plot_lge_roi_results.py \
  --experiment_dir "${experiment_dir}"
