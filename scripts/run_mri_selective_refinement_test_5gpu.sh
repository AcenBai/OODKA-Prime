#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-/data4/baihexiang/conda_envs/biomedparse_v2/bin/python}"
local_ckpt="${LOCAL_CKPT:-/data4/baihexiang/SegMan/spatial_combination/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_detached_rerun_20260825/fusion_whs_mri_gv_roi_best.pth}"
global_pred_dir="${GLOBAL_PRED_DIR:-/data4/baihexiang/SegMan/Distangler3/distangler3_output/distangler_Dataset010_WHS_MRI_OOD_mri/run_20260324_175830/sliding_window_eval/pred_raw}"
out_root="${OUT_ROOT:-experiments/selective_refinement_20260902/test_locked}"
export BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/data4/baihexiang/SegMan/BiomedParse}"
export BIOMEDPARSE_CKPT="${BIOMEDPARSE_CKPT:-${BIOMEDPARSE_DIR}/biomedparse_v2.ckpt}"

case_shards=(
  "heart_5006,heart_5013,heart_5020,heart_5021,heart_5009"
  "heart_5012,heart_5025,heart_5016,heart_5017,heart_5011"
  "heart_5015,heart_5007,heart_5022,heart_5004,heart_5008,heart_5010"
  "heart_5026,heart_5018,heart_5002,heart_5001,heart_5024"
  "heart_5014,heart_5003,heart_5019,heart_5023,heart_5005"
)

pids=()
shard_dirs=()
for gpu in 0 1 2 3 4; do
  shard_dir="${out_root}/shard_gpu${gpu}"
  shard_dirs+=("${shard_dir}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" run_eval_lge_roi.py \
    --checkpoint "${local_ckpt}" \
    --split test \
    --case_ids "${case_shards[${gpu}]}" \
    --device cuda:0 \
    --batch_size 4 \
    --out_dir "${shard_dir}" \
    --selective_global_pred_dir "${global_pred_dir}" \
    --selective_thresholds 0.95 \
    --selective_ambiguity_margins 0.05 \
    --selective_postprocess both \
    --selective_overwrite_scope any \
    --selective_save_predictions &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${python_bin}" scripts/merge_selective_refinement_shards.py \
  --shard_dirs "${shard_dirs[@]}" \
  --out_dir "${out_root}/merged"
