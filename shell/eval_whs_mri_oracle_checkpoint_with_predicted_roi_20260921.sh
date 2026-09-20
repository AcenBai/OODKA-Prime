#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
experiment_dir="${1:?usage: $0 <oracle-experiment-dir> <gpu-index>}"
gpu_index="${2:?usage: $0 <oracle-experiment-dir> <gpu-index>}"
checkpoint="${experiment_dir}/fusion_whs_mri_roi_best.pth"

cd "${repo_dir}"
test -f "${checkpoint}"

for postprocess in none largest_per_class; do
  out_dir="${experiment_dir}/test_best_predicted_transfer_${postprocess}"
  mkdir -p "${out_dir}"
  diagnostics=()
  if [ "${postprocess}" = none ]; then
    diagnostics=(
      --block_diagnostics_jsonl "${out_dir}/block_diagnostics.jsonl"
    )
  fi
  "${python_bin}" run_eval_lge_roi.py \
    --checkpoint "${checkpoint}" \
    --split test \
    --device "cuda:${gpu_index}" \
    --batch_size 1 \
    --roi_source predicted \
    --roi_transform checkpoint \
    --decision refinement \
    --postprocess "${postprocess}" \
    "${diagnostics[@]}" \
    --out_dir "${out_dir}" \
    2>&1 | tee "${out_dir}/eval.log"
done

touch "${experiment_dir}/PREDICTED_TRANSFER_EVAL_COMPLETE"
