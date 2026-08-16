#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
n_epochs="${2:-30}"
augment_mode="${3:-on}"
experiment_dir="${1:-${repo_dir}/experiments/lge_roi_v2_z1_b18_${n_epochs}ep_20260817}"

augmentation_args=()
case "${augment_mode}" in
  on) ;;
  off) augmentation_args+=(--no_augment) ;;
  *)
    echo "augment mode must be 'on' or 'off', got: ${augment_mode}" >&2
    exit 2
    ;;
esac

cd "${repo_dir}"
mkdir -p "${experiment_dir}"

"${python_bin}" run_train_lge_roi.py \
  --v2 \
  --device cuda:1 \
  --n_epochs "${n_epochs}" \
  --warmup_epochs 10 \
  --batch_size 18 \
  --num_workers 4 \
  --roi_threshold 0.3 \
  --roi_expand 1.25 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 5 \
  --test_best_on_improvement \
  --best_test_device cuda:2 \
  "${augmentation_args[@]}" \
  --output_dir "${experiment_dir}"

checkpoint="${experiment_dir}/fusion_lge_roi_v2_best.pth"
"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split val \
  --device cuda:1 \
  --batch_size 18 \
  --decision auto \
  --out_dir "${experiment_dir}/eval_val_best"

"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device cuda:1 \
  --batch_size 18 \
  --decision auto \
  --out_dir "${experiment_dir}/eval_test_best"

"${python_bin}" scripts/plot_lge_roi_results.py \
  --experiment_dir "${experiment_dir}"
