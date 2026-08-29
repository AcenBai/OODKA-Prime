#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash shell/manual_eval_mri_gv_roi.sh [physical_gpu_id] [experiment_dir]

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:-2}"
experiment_dir="${2:-${repo_dir}/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_detached_rerun_20260825}"
checkpoint="${experiment_dir}/fusion_whs_mri_gv_roi_best.pth"
output_dir="${experiment_dir}/test_best_predicted_offline"

test -f "${checkpoint}"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${repo_dir}"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device cuda:0 \
  --batch_size 2 \
  --roi_source predicted \
  --decision spatial \
  --out_dir "${output_dir}" \
  2>&1 | tee "${output_dir}/eval.log"

touch "${experiment_dir}/MANUAL_TEST_COMPLETE"
