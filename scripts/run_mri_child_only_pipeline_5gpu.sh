#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-/data4/baihexiang/conda_envs/biomedparse_v2/bin/python}"
init_checkpoint="${INIT_CHECKPOINT:-/data4/baihexiang/SegMan/spatial_combination/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_detached_rerun_20260825/fusion_whs_mri_gv_roi_best.pth}"
global_val_pred_dir="${GLOBAL_VAL_PRED_DIR:-${repo_root}/experiments/selective_refinement_20260902/global_val/pred_raw}"
global_test_pred_dir="${GLOBAL_TEST_PRED_DIR:-/data4/baihexiang/SegMan/Distangler3/distangler3_output/distangler_Dataset010_WHS_MRI_OOD_mri/run_20260324_175830/sliding_window_eval/pred_raw}"
pipeline_root="${PIPELINE_ROOT:-${repo_root}/experiments/selective_refinement_20260902/child_only_pipeline}"
export BIOMEDPARSE_DIR="${BIOMEDPARSE_DIR:-/data4/baihexiang/SegMan/BiomedParse}"
export BIOMEDPARSE_CKPT="${BIOMEDPARSE_CKPT:-${BIOMEDPARSE_DIR}/biomedparse_v2.ckpt}"

variant_names=(
  "full_t02_e140"
  "center_t02_e140"
  "center_t03_e125"
  "center_t02_e125"
  "full_t03_e125"
)
roi_thresholds=(0.2 0.2 0.3 0.2 0.3)
roi_expands=(1.4 1.4 1.25 1.25 1.25)
roi_fallbacks=(full center center center full)

mkdir -p "${pipeline_root}/train"
pids=()
run_dirs=()
for gpu in 0 1 2 3 4; do
  run_dir="${pipeline_root}/train/${variant_names[${gpu}]}"
  mkdir -p "${run_dir}"
  run_dirs+=("${run_dir}")
  echo "Starting GPU ${gpu}: ${variant_names[${gpu}]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" run_train_whs_roi.py \
    --dataset_name Dataset010_WHS_MRI_OOD \
    --roi_strategy great_vessel \
    --selective_children_only \
    --init_checkpoint "${init_checkpoint}" \
    --device cuda:0 \
    --output_dir "${run_dir}" \
    --n_epochs 10 \
    --warmup_epochs 0 \
    --block_z 4 \
    --batch_size 1 \
    --image_size 320 \
    --num_workers 2 \
    --raw_cache_cases 4 \
    --roi_threshold "${roi_thresholds[${gpu}]}" \
    --roi_expand "${roi_expands[${gpu}]}" \
    --roi_fallback "${roi_fallbacks[${gpu}]}" \
    --roi_refresh_every 5 \
    --pseudo_rgb_mode adjacent \
    --roi_prompt_loss_reduction prompt_mean \
    --lr 5e-5 \
    --val_every_epochs 2 \
    >"${run_dir}/launcher.log" 2>&1 &
  pids+=("$!")
  echo "$!" >"${run_dir}/detached.pid"
done

train_status=0
for index in 0 1 2 3 4; do
  if wait "${pids[${index}]}"; then
    echo "Finished: ${variant_names[${index}]}"
  else
    echo "FAILED: ${variant_names[${index}]}" >&2
    train_status=1
  fi
done
if [[ "${train_status}" -ne 0 ]]; then
  exit "${train_status}"
fi

selection_json="${pipeline_root}/model_selection.json"
selected_checkpoint="$("${python_bin}" scripts/select_child_only_model.py \
  --run_dirs "${run_dirs[@]}" \
  --output "${selection_json}")"
echo "Selected checkpoint: ${selected_checkpoint}"

val_dir="${pipeline_root}/val_sweep"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${selected_checkpoint}" \
  --split val \
  --device cuda:0 \
  --batch_size 4 \
  --out_dir "${val_dir}" \
  --selective_global_pred_dir "${global_val_pred_dir}" \
  --selective_thresholds 0.5,0.65,0.75,0.8,0.85,0.9,0.95,0.98,0.99 \
  --selective_ambiguity_margins 0.0,0.05,0.1 \
  --selective_postprocess both \
  --selective_overwrite_scope both \
  >"${pipeline_root}/val_sweep.log" 2>&1

locked_json="${pipeline_root}/locked_selective_config.json"
read -r locked_threshold locked_margin locked_scope locked_postprocess < <(
  "${python_bin}" scripts/select_selective_config.py \
    --summary "${val_dir}/selective_sweep_summary.json" \
    --output "${locked_json}"
)
echo "Locked from validation: threshold=${locked_threshold}, margin=${locked_margin}, scope=${locked_scope}, postprocess=${locked_postprocess}"

case_shards=(
  "heart_5006,heart_5013,heart_5020,heart_5021,heart_5009"
  "heart_5012,heart_5025,heart_5016,heart_5017,heart_5011"
  "heart_5015,heart_5007,heart_5022,heart_5004,heart_5008,heart_5010"
  "heart_5026,heart_5018,heart_5002,heart_5001,heart_5024"
  "heart_5014,heart_5003,heart_5019,heart_5023,heart_5005"
)
test_root="${pipeline_root}/test_locked"
pids=()
shard_dirs=()
for gpu in 0 1 2 3 4; do
  shard_dir="${test_root}/shard_gpu${gpu}"
  mkdir -p "${shard_dir}"
  shard_dirs+=("${shard_dir}")
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" run_eval_lge_roi.py \
    --checkpoint "${selected_checkpoint}" \
    --split test \
    --case_ids "${case_shards[${gpu}]}" \
    --device cuda:0 \
    --batch_size 4 \
    --out_dir "${shard_dir}" \
    --selective_global_pred_dir "${global_test_pred_dir}" \
    --selective_thresholds "${locked_threshold}" \
    --selective_ambiguity_margins "${locked_margin}" \
    --selective_postprocess both \
    --selective_overwrite_scope "${locked_scope}" \
    --selective_save_predictions \
    >"${shard_dir}/launcher.log" 2>&1 &
  pids+=("$!")
done

test_status=0
for index in 0 1 2 3 4; do
  if ! wait "${pids[${index}]}"; then
    echo "FAILED test shard GPU ${index}" >&2
    test_status=1
  fi
done
if [[ "${test_status}" -ne 0 ]]; then
  exit "${test_status}"
fi

merged_dir="${test_root}/merged"
"${python_bin}" scripts/merge_selective_refinement_shards.py \
  --shard_dirs "${shard_dirs[@]}" \
  --out_dir "${merged_dir}"

"${python_bin}" scripts/plot_selective_refinement_results.py \
  --merged_dir "${merged_dir}" \
  --shard_root "${test_root}" \
  --global_pred_dir "${global_test_pred_dir}" \
  --images_dir nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset010_WHS_MRI_OOD/imagesTs \
  --labels_dir nnUNet/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset010_WHS_MRI_OOD/labelsTs \
  --scope "${locked_scope}" \
  --out_dir "${test_root}/analysis"

touch "${pipeline_root}/PIPELINE_COMPLETE"
echo "Pipeline complete: ${pipeline_root}"
