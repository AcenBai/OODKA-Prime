#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash shell/manual_eval_visualize_relative_kd_ct.sh [physical_gpu_id] [experiment_dir]

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:-0}"
experiment_dir="${2:-${repo_dir}/experiments/spatial_beta_p07_ct_f0_relativekd_no_roi_noaug_30ep_detached_rerun_20260825}"
checkpoint="${experiment_dir}/fusion_disentangle_best.pth"
output_dir="${experiment_dir}/test_best_adjacent_offline"
visual_dir="${experiment_dir}/analysis/mechanism_v3_matched_scale_offline"
shared_scale="${repo_dir}/experiments/spatial_beta_p07_ct_f0_30ep_20260730/analysis/mechanism_v3/heart_1004_z0079/representation/color_scales.json"

test -f "${checkpoint}"
test -f "${shared_scale}"
mkdir -p "${output_dir}" "${visual_dir}" "${experiment_dir}/logs" "${experiment_dir}/analysis"

# Force Transformers to use the already populated local cache. This avoids the
# hf-mirror.com HEAD request that caused the previous automatic evaluation to fail.
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${experiment_dir}/mpl_cache_manual_eval"

cd "${repo_dir}"

"${python_bin}" run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --split test \
  --block_z 4 \
  --batch_size 1 \
  --image_size 512 \
  --norm_mode ct \
  --pseudo_rgb_mode adjacent \
  --device cuda:0 \
  --distangler_ckpt "${checkpoint}" \
  --out_dir "${output_dir}" \
  2>&1 | tee "${experiment_dir}/logs/manual_eval_test_offline.log"

"${python_bin}" scripts/analyze_beta_router.py \
  --checkpoint "${checkpoint}" \
  --device cuda:0 \
  --output "${experiment_dir}/analysis/beta_router_offline.json" \
  2>&1 | tee "${experiment_dir}/logs/manual_analyze_beta_router_offline.log"

"${python_bin}" scripts/visualize_mechanism_v3.py \
  --checkpoint "${checkpoint}" \
  --case_id heart_1004 \
  --split val \
  --slice_index 79 \
  --selection all_classes \
  --device cuda:0 \
  --block_z 4 \
  --shared_color_scales "${shared_scale}" \
  --output_root "${visual_dir}" \
  2>&1 | tee "${experiment_dir}/logs/manual_visualize_heart_1004_offline.log"

"${python_bin}" scripts/visualize_mechanism_v3.py \
  --checkpoint "${checkpoint}" \
  --case_id heart_1014 \
  --split val \
  --slice_index 191 \
  --selection all_classes \
  --device cuda:0 \
  --block_z 4 \
  --shared_color_scales "${shared_scale}" \
  --output_root "${visual_dir}" \
  2>&1 | tee "${experiment_dir}/logs/manual_visualize_heart_1014_offline.log"

touch "${experiment_dir}/MANUAL_EVAL_VIS_COMPLETE"
