#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
relative_dir="${repo_dir}/experiments/spatial_beta_p07_ct_f0_relativekd_no_roi_noaug_30ep_detached_rerun_20260825"
ct_roi_dir="${repo_dir}/experiments/whs_ct_gv_roi_z4_b1_promptmean_aug_30ep_detached_rerun_20260825"
mri_roi_dir="${repo_dir}/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_detached_rerun_20260825"

mkdir -p "${relative_dir}" "${ct_roi_dir}" "${mri_roi_dir}"

nohup setsid bash "${repo_dir}/shell/run_ct_relative_kd_no_roi_noaug_30ep_20260825.sh" \
  "${relative_dir}" >"${relative_dir}/detached_launcher.log" 2>&1 </dev/null &
relative_pid=$!
printf '%s\n' "${relative_pid}" >"${relative_dir}/detached.pid"

nohup setsid bash "${repo_dir}/shell/run_whs_ct_gv_roi_z4_b1_30ep_20260825.sh" \
  "${ct_roi_dir}" >"${ct_roi_dir}/detached_launcher.log" 2>&1 </dev/null &
ct_roi_pid=$!
printf '%s\n' "${ct_roi_pid}" >"${ct_roi_dir}/detached.pid"

nohup setsid bash "${repo_dir}/shell/run_whs_mri_gv_roi_z4_b2_30ep_20260825.sh" \
  "${mri_roi_dir}" >"${mri_roi_dir}/detached_launcher.log" 2>&1 </dev/null &
mri_roi_pid=$!
printf '%s\n' "${mri_roi_pid}" >"${mri_roi_dir}/detached.pid"

printf 'relative_kd_gpu0 pid=%s dir=%s\n' "${relative_pid}" "${relative_dir}"
printf 'ct_gv_roi_gpu1 pid=%s dir=%s\n' "${ct_roi_pid}" "${ct_roi_dir}"
printf 'mri_gv_roi_gpu2 pid=%s dir=%s\n' "${mri_roi_pid}" "${mri_roi_dir}"
