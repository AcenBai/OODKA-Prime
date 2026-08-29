#!/usr/bin/env bash
set -euo pipefail

# Train CT GV-ROI from scratch with the requested B=1, Z=4 configuration,
# then evaluate the best checkpoint on the independent test split.
#
# Usage:
#   bash shell/manual_train_eval_ct_gv_roi_z4_b1.sh [physical_gpu_id] [experiment_dir]
#
# Run this only on a genuinely idle 24 GiB GPU. The mixed anchor/refinement
# stage previously used about 20.5 GiB inside the training process.

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
gpu_id="${1:-1}"
experiment_dir="${2:-${repo_dir}/experiments/whs_ct_gv_roi_z4_b1_promptmean_aug_30ep_clean_gpu_20260827}"

exec bash "${repo_dir}/shell/manual_train_eval_ct_gv_roi_lowmem.sh" \
  "${gpu_id}" 4 "${experiment_dir}"
