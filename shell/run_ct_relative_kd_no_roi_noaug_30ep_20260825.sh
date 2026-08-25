#!/usr/bin/env bash
set -euo pipefail

ROOT="/data4/baihexiang/SegMan/spatial_combination"
PYTHON="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
EXP_DIR="${ROOT}/experiments/spatial_beta_p07_ct_f0_relativekd_no_roi_noaug_30ep_20260825"
OLD_SCALE="${ROOT}/experiments/spatial_beta_p07_ct_f0_30ep_20260730/analysis/mechanism_v3/heart_1004_z0079/representation/color_scales.json"

export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR="${EXP_DIR}/mpl_cache"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/analysis"
cd "${ROOT}"

"${PYTHON}" run_train.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --block_z 4 \
  --batch_size 1 \
  --image_size 512 \
  --norm_mode ct \
  --n_epochs 30 \
  --lr 1e-4 \
  --lr_schedule cosine \
  --lr_warmup_epochs 2 \
  --min_lr_ratio 0.1 \
  --seed 42 \
  --num_workers 2 \
  --route_warmup_epochs 5 \
  --w_p_ot 0.1 \
  --w_s_ot 0.1 \
  --p_ot_start_epoch 2 \
  --s_ot_start_epoch 3 \
  --ot_warmup_epochs 5 \
  --ot_max_grid_size 32 \
  --relative_kd \
  --relative_kd_expert_weight 1.0 \
  --val_every_epochs 5 \
  --device cuda:0 \
  --output_dir "${EXP_DIR}" \
  2>&1 | tee "${EXP_DIR}/logs/train_console.log"

BEST_CKPT="${EXP_DIR}/fusion_disentangle_best.pth"
test -f "${BEST_CKPT}"

"${PYTHON}" run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --split test \
  --block_z 4 \
  --batch_size 1 \
  --image_size 512 \
  --norm_mode ct \
  --pseudo_rgb_mode adjacent \
  --device cuda:0 \
  --distangler_ckpt "${BEST_CKPT}" \
  --out_dir "${EXP_DIR}/test_best_adjacent" \
  2>&1 | tee "${EXP_DIR}/logs/eval_test.log"

"${PYTHON}" scripts/analyze_beta_router.py \
  --checkpoint "${BEST_CKPT}" \
  --device cuda:0 \
  --output "${EXP_DIR}/analysis/beta_router.json" \
  2>&1 | tee "${EXP_DIR}/logs/analyze_beta_router.log"

"${PYTHON}" scripts/visualize_mechanism_v3.py \
  --checkpoint "${BEST_CKPT}" \
  --case_id heart_1004 \
  --split val \
  --slice_index 79 \
  --selection all_classes \
  --device cuda:0 \
  --block_z 4 \
  --shared_color_scales "${OLD_SCALE}" \
  --output_root "${EXP_DIR}/analysis/mechanism_v3_matched_scale" \
  2>&1 | tee "${EXP_DIR}/logs/visualize_heart_1004.log"

"${PYTHON}" scripts/visualize_mechanism_v3.py \
  --checkpoint "${BEST_CKPT}" \
  --case_id heart_1014 \
  --split val \
  --slice_index 191 \
  --selection all_classes \
  --device cuda:0 \
  --block_z 4 \
  --shared_color_scales "${OLD_SCALE}" \
  --output_root "${EXP_DIR}/analysis/mechanism_v3_matched_scale" \
  2>&1 | tee "${EXP_DIR}/logs/visualize_heart_1014.log"

touch "${EXP_DIR}/PIPELINE_COMPLETE"
