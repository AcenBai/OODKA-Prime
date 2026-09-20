#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
output_dir="${repo_dir}/experiments/mri_predicted_pad_comparison_20260921"
global_dir="${repo_dir}/experiments/whs_mri_global_noaug_relative_capacity_s_z4_b1_30ep_20260920"
global_seed43_dir="${repo_dir}/experiments/whs_mri_global_noaug_relative_capacity_s_z4_b1_seed43_30ep_20260921"

matrix_runs=(
  "predicted_pad_w5|${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_30ep_20260921"
  "predicted_pad_w10|${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w10_30ep_20260921"
  "predicted_pad_w5_expand18|${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_t02_e18_30ep_20260921"
  "predicted_pad_w5_threshold01|${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_t01_e14_30ep_20260921"
)
seed43_dir="${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_seed43_30ep_20260921"

wait_for_pipeline() {
  local experiment_dir="$1"
  while [ ! -f "${experiment_dir}/PIPELINE_COMPLETE" ]; do
    sleep 60
  done
}

compare_runs() {
  local comparison_dir="$1"
  local baseline_dir="$2"
  local baseline_name="$3"
  shift 3
  local runs=("$@")
  local raw_args=()
  local lcc_args=()
  local spec name experiment_dir

  for spec in "${runs[@]}"; do
    name="${spec%%|*}"
    experiment_dir="${spec#*|}"
    raw_args+=(
      --run "${name}=${experiment_dir}/test_best_predicted_none/metrics.csv"
    )
    lcc_args+=(
      --run "${name}_lcc=${experiment_dir}/test_best_predicted_largest_per_class/metrics.csv"
    )
  done

  mkdir -p "${comparison_dir}/raw" "${comparison_dir}/largest_per_class"
  "${python_bin}" "${repo_dir}/scripts/compare_paired_segmentation_runs.py" \
    --baseline "${baseline_dir}/test_best_none/metrics.csv" \
    --baseline-name "${baseline_name}" \
    "${raw_args[@]}" \
    --output-dir "${comparison_dir}/raw"
  "${python_bin}" "${repo_dir}/scripts/compare_paired_segmentation_runs.py" \
    --baseline "${baseline_dir}/test_best_largest_per_class/metrics.csv" \
    --baseline-name "${baseline_name}_lcc" \
    "${lcc_args[@]}" \
    --output-dir "${comparison_dir}/largest_per_class"
}

cd "${repo_dir}"
mkdir -p "${output_dir}"

wait_for_pipeline "${global_dir}"
for spec in "${matrix_runs[@]}"; do
  wait_for_pipeline "${spec#*|}"
done
compare_runs "${output_dir}/matrix" "${global_dir}" global "${matrix_runs[@]}"
touch "${output_dir}/MATRIX_COMPARISON_COMPLETE"

wait_for_pipeline "${seed43_dir}"
wait_for_pipeline "${global_seed43_dir}"
compare_runs \
  "${output_dir}/seed_replication/seed42" \
  "${global_dir}" \
  global_seed42 \
  "predicted_pad_w5_seed42|${repo_dir}/experiments/whs_mri_predicted_pad_noaug_relative_capacity_s_z4_b1_w5_30ep_20260921"
compare_runs \
  "${output_dir}/seed_replication/seed43" \
  "${global_seed43_dir}" \
  global_seed43 \
  "predicted_pad_w5_seed43|${seed43_dir}"
touch "${output_dir}/SEED_REPLICATION_COMPARISON_COMPLETE"
