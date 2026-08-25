#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment="${repo_dir}/experiments/whs_ct_roi3d_z4_b1_promptmean_30ep_20260824"
checkpoint="${experiment}/fusion_whs_ct_roi_best.pth"
device="${1:-cuda:0}"
batch_size="${2:-4}"
full_output="${experiment}/diagnostic_test_full_roi3d_z4"
oracle_output="${experiment}/diagnostic_test_oracle_roi3d_z4"

cd "${repo_dir}"
mkdir -p "${full_output}" "${oracle_output}"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device "${device}" \
  --batch_size "${batch_size}" \
  --roi_source full \
  --decision refinement \
  --out_dir "${full_output}" \
  2>&1 | tee "${full_output}/eval.log"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device "${device}" \
  --batch_size "${batch_size}" \
  --roi_source ground_truth \
  --decision refinement \
  --out_dir "${oracle_output}" \
  2>&1 | tee "${oracle_output}/eval.log"
