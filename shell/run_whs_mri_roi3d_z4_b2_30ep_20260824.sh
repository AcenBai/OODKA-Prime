#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/whs_mri_roi3d_z4_b2_promptmean_30ep_20260824}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

exec "${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device cuda:1 \
  --n_epochs 30 \
  --warmup_epochs 10 \
  --block_z 4 \
  --batch_size 2 \
  --image_size 320 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.2 \
  --roi_expand 1.4 \
  --roi_fallback full \
  --roi_refresh_every 5 \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}"
