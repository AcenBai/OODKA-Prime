#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_index="${1:?usage: $0 <gpu-index> [output-dir]}"
experiment_dir="${2:-${repo_dir}/experiments/whs_mri_global_noaug_relative_capacity_s_z4_b1_30ep_20260920}"
seed="${GLOBAL_SEED:-42}"
aligned_dir="/data4/baihexiang/SegMan/Distangler3/distangler3_output/biomedparse_preprocessed_Dataset010_WHS_MRI_OOD"

cd "${repo_dir}"
mkdir -p "${experiment_dir}/logs"

"${python_bin}" run_train.py \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --fold 0 \
  --block_z 4 \
  --batch_size 1 \
  --image_size 320 \
  --norm_mode mri \
  --biomedparse_preproc_dir "${aligned_dir}" \
  --n_epochs 30 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --lr_warmup_epochs 0 \
  --min_lr_ratio 0.05 \
  --seed "${seed}" \
  --w_seg 1.0 \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --relative_kd \
  --s_transport_mode capacity_partial \
  --val_every_epochs 5 \
  --device "cuda:${gpu_index}" \
  --output_dir "${experiment_dir}" \
  2>&1 | tee "${experiment_dir}/logs/train_console.log"

checkpoint="${experiment_dir}/fusion_disentangle_best.pth"
test -f "${checkpoint}"

for postprocess in none largest_per_class; do
  out_dir="${experiment_dir}/test_best_${postprocess}"
  mkdir -p "${out_dir}"
  "${python_bin}" run_eval_oodka.py \
    --dataset_name Dataset010_WHS_MRI_OOD \
    --fold 0 \
    --split test \
    --block_z 4 \
    --batch_size 1 \
    --image_size 320 \
    --norm_mode mri \
    --pseudo_rgb_mode adjacent \
    --postprocess "${postprocess}" \
    --device "cuda:${gpu_index}" \
    --distangler_ckpt "${checkpoint}" \
    --out_dir "${out_dir}" \
    2>&1 | tee "${out_dir}/eval.log"
done

touch "${experiment_dir}/PIPELINE_COMPLETE"
