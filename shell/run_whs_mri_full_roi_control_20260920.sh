#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_index="${1:?usage: $0 <gpu-index> [output-dir]}"
experiment_dir="${2:-${repo_dir}/experiments/whs_mri_full_roi_control_noaug_relative_capacity_s_z4_b1_30ep_20260920}"

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

"${python_bin}" run_train_whs_roi.py \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device "cuda:${gpu_index}" \
  --n_epochs 30 \
  --warmup_epochs 0 \
  --block_z 4 \
  --batch_size 1 \
  --image_size 320 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.2 \
  --roi_expand 1.4 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --roi_train_source full \
  --roi_transform resize \
  --no_augment \
  --relative_kd \
  --s_transport_mode capacity_partial \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}" \
  2>&1 | tee "${experiment_dir}/launcher.log"

checkpoint="${experiment_dir}/fusion_whs_mri_roi_best.pth"
test -f "${checkpoint}"

for postprocess in none largest_per_class; do
  out_dir="${experiment_dir}/test_best_full_${postprocess}"
  mkdir -p "${out_dir}"
  "${python_bin}" run_eval_lge_roi.py \
    --checkpoint "${checkpoint}" \
    --split test \
    --device "cuda:${gpu_index}" \
    --batch_size 1 \
    --roi_source full \
    --roi_transform checkpoint \
    --decision refinement \
    --postprocess "${postprocess}" \
    --out_dir "${out_dir}" \
    2>&1 | tee "${out_dir}/eval.log"
done

touch "${experiment_dir}/PIPELINE_COMPLETE"
