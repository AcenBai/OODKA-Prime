#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
roi_transform="${1:?usage: $0 <resize|pad|letterbox> <gpu-index> [output-dir]}"
gpu_index="${2:?usage: $0 <resize|pad|letterbox> <gpu-index> [output-dir]}"
train_batch_size="${ROI_BATCH_SIZE:-1}"
experiment_dir="${3:-${repo_dir}/experiments/whs_mri_oracle_${roi_transform}_noaug_relative_capacity_s_z4_b${train_batch_size}_30ep_20260920}"
test_dir="${experiment_dir}/test_best_oracle"

case "${roi_transform}" in
  resize|pad|letterbox) ;;
  *)
    echo "roi transform must be resize, pad, or letterbox" >&2
    exit 2
    ;;
esac

cd "${repo_dir}"
mkdir -p "${experiment_dir}" "${test_dir}"

"${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device "cuda:${gpu_index}" \
  --n_epochs 30 \
  --warmup_epochs 0 \
  --block_z 4 \
  --batch_size "${train_batch_size}" \
  --image_size 320 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.2 \
  --roi_expand 1.4 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --roi_train_source ground_truth \
  --roi_transform "${roi_transform}" \
  --no_augment \
  --relative_kd \
  --s_transport_mode capacity_partial \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}" \
  2>&1 | tee "${experiment_dir}/launcher.log"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${experiment_dir}/fusion_whs_mri_roi_best.pth" \
  --split test \
  --device "cuda:${gpu_index}" \
  --batch_size 1 \
  --roi_source ground_truth \
  --roi_transform checkpoint \
  --decision refinement \
  --block_diagnostics_jsonl "${test_dir}/block_diagnostics.jsonl" \
  --out_dir "${test_dir}" \
  2>&1 | tee "${test_dir}/eval.log"

"${python_bin}" scripts/plot_roi_geometry_diagnostics.py \
  --block_diagnostics "${test_dir}/block_diagnostics.jsonl" \
  --output_dir "${test_dir}/geometry" \
  --image_size 320 \
  --source_classes 1,2,3,4,5,6,7 \
  --title "MRI whole-heart Oracle ROI geometry: ${roi_transform}" \
  > "${test_dir}/geometry.log"

touch "${experiment_dir}/PIPELINE_COMPLETE"
