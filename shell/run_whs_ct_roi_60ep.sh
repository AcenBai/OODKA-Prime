#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/whs_ct_roi_z1_b4_60ep_20260821}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

exec "${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --n_epochs 60 \
  --warmup_epochs 10 \
  --batch_size 4 \
  --image_size 512 \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.3 \
  --roi_expand 1.25 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}"
