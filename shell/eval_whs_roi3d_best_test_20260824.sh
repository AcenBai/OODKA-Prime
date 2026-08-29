#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
device="${1:-cuda:0}"
batch_size="${2:-4}"

ct_experiment="${repo_dir}/experiments/whs_ct_roi3d_z4_b1_promptmean_30ep_20260824"
ct_output="${ct_experiment}/test_best_epoch020_roi3d_z4_predicted"
mri_experiment="${repo_dir}/experiments/whs_mri_roi3d_z4_b2_promptmean_30ep_20260824"
mri_output="${mri_experiment}/test_best_epoch025_roi3d_z4_predicted"

cd "${repo_dir}"
mkdir -p "${ct_output}" "${mri_output}"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${ct_experiment}/fusion_whs_ct_roi_best.pth" \
  --split test \
  --device "${device}" \
  --batch_size "${batch_size}" \
  --roi_source predicted \
  --decision refinement \
  --out_dir "${ct_output}" \
  2>&1 | tee "${ct_output}/eval.log"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${mri_experiment}/fusion_whs_mri_roi_best.pth" \
  --split test \
  --device "${device}" \
  --batch_size "${batch_size}" \
  --roi_source predicted \
  --decision refinement \
  --out_dir "${mri_output}" \
  2>&1 | tee "${mri_output}/eval.log"
