#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:?usage: $0 <experiment-dir> <gpu-index>}"
gpu_index="${2:?usage: $0 <experiment-dir> <gpu-index>}"
checkpoint="${experiment_dir}/fusion_whs_mri_roi_best.pth"
out_dir="${experiment_dir}/test_best_oracle_largest_per_class"

cd "${repo_dir}"
test -f "${checkpoint}"
mkdir -p "${out_dir}"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device "cuda:${gpu_index}" \
  --batch_size 1 \
  --roi_source ground_truth \
  --roi_transform checkpoint \
  --decision refinement \
  --postprocess largest_per_class \
  --out_dir "${out_dir}" \
  2>&1 | tee "${out_dir}/eval.log"

touch "${experiment_dir}/ORACLE_LCC_EVAL_COMPLETE"
