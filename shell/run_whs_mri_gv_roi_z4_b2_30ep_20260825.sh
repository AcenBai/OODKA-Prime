#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:-${repo_dir}/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_20260825}"
test_dir="${experiment_dir}/test_best_predicted"

cd "${repo_dir}"
mkdir -p "${experiment_dir}" "${test_dir}"

"${python_bin}" run_train_whs_roi.py \
  --roi_strategy great_vessel \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device cuda:2 \
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
  --output_dir "${experiment_dir}" \
  2>&1 | tee "${experiment_dir}/launcher.log"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${experiment_dir}/fusion_whs_mri_gv_roi_best.pth" \
  --split test \
  --device cuda:2 \
  --batch_size 4 \
  --roi_source predicted \
  --decision spatial \
  --out_dir "${test_dir}" \
  2>&1 | tee "${test_dir}/eval.log"

touch "${experiment_dir}/PIPELINE_COMPLETE"
