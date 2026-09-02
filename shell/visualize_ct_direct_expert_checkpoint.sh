#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash shell/visualize_ct_direct_expert_checkpoint.sh \
#     GPU EXPERIMENT_DIR [CASE_ID] [SLICE_INDEX]

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:?physical GPU id is required}"
experiment_dir="${2:?experiment directory is required}"
case_id="${3:-heart_1004}"
slice_index="${4:-79}"
checkpoint="${experiment_dir}/fusion_disentangle_best.pth"
output_root="${experiment_dir}/analysis/mechanism_v3_matched_scale"
shared_scale="${repo_dir}/experiments/20260730_milestone1_OTvisual_CT30epochs/analysis/mechanism_v3/heart_1004_z0079/representation/color_scales.json"

test -f "${checkpoint}"
test -f "${shared_scale}"
mkdir -p "${experiment_dir}/logs" "${output_root}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${experiment_dir}/mpl_cache"

cd "${repo_dir}"
"${python_bin}" scripts/visualize_mechanism_v3.py \
  --checkpoint "${checkpoint}" \
  --case_id "${case_id}" \
  --split val \
  --slice_index "${slice_index}" \
  --selection all_classes \
  --device cuda:0 \
  --block_z 4 \
  --shared_color_scales "${shared_scale}" \
  --output_root "${output_root}" \
  2>&1 | tee "${experiment_dir}/logs/visualize_${case_id}.log"

