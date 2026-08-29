#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash shell/manual_train_eval_ct_gv_roi_lowmem.sh [physical_gpu_id] [block_z] [experiment_dir]
#
# Defaults to block_z=2 because the Z=4 mixed anchor+refinement step reached
# 20.50 GiB before backward and OOMed in a shared 24 GiB GPU environment.

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:-1}"
block_z="${2:-2}"
experiment_dir="${3:-${repo_dir}/experiments/whs_ct_gv_roi_z${block_z}_b1_promptmean_aug_30ep_lowmem_20260827}"
test_dir="${experiment_dir}/test_best_predicted_offline"

if [[ ! "${block_z}" =~ ^[1-9][0-9]*$ ]]; then
  echo "block_z must be a positive integer" >&2
  exit 2
fi

mkdir -p "${experiment_dir}" "${test_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${repo_dir}"

"${python_bin}" run_train_whs_roi.py \
  --roi_strategy great_vessel \
  --dataset_name Dataset009_CT_OOD \
  --device cuda:0 \
  --n_epochs 30 \
  --warmup_epochs 10 \
  --block_z "${block_z}" \
  --batch_size 1 \
  --image_size 512 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 4 \
  --raw_cache_cases 4 \
  --roi_threshold 0.3 \
  --roi_expand 1.25 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 5 \
  --output_dir "${experiment_dir}" \
  2>&1 | tee "${experiment_dir}/launcher.log"

checkpoint="${experiment_dir}/fusion_whs_ct_gv_roi_best.pth"
test -f "${checkpoint}"

# Use batch_size=1 for the independent test as well, prioritizing robustness.
"${python_bin}" run_eval_lge_roi.py \
  --checkpoint "${checkpoint}" \
  --split test \
  --device cuda:0 \
  --batch_size 1 \
  --roi_source predicted \
  --decision spatial \
  --out_dir "${test_dir}" \
  2>&1 | tee "${test_dir}/eval.log"

touch "${experiment_dir}/PIPELINE_COMPLETE"
