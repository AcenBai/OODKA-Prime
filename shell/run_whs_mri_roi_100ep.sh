#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/whs_mri_roi_z1_b10_100ep_20260821}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

exec "${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device cuda:2 \
  --n_epochs 100 \
  --warmup_epochs 10 \
  --batch_size 10 \
  --image_size 320 \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.3 \
  --roi_expand 1.25 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}"
