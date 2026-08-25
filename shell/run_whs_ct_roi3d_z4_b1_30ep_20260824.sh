#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/whs_ct_roi3d_z4_b1_promptmean_30ep_20260824}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

exec "${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --n_epochs 30 \
  --warmup_epochs 10 \
  --block_z 4 \
  --batch_size 1 \
  --image_size 512 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.3 \
  --roi_expand 1.25 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}"
