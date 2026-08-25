#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment="${repo_dir}/experiments/whs_ct_roi3d_z4_b1_promptmean_30ep_20260824"
output="${experiment}/test_best_epoch020_roi3d_z4_predicted"
device="${1:-cuda:0}"
batch_size="${2:-4}"

cd "${repo_dir}"
mkdir -p "${output}"
"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${experiment}/fusion_whs_ct_roi_best.pth" \
  --split test \
  --device "${device}" \
  --batch_size "${batch_size}" \
  --roi_source predicted \
  --decision refinement \
  --out_dir "${output}" \
  2>&1 | tee "${output}/eval.log"
