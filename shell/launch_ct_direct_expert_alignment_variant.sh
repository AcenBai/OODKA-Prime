#!/usr/bin/env bash
set -euo pipefail

# Launch one complete train -> test -> visualization pipeline independently of
# the current terminal/session.
#
# Usage:
#   bash shell/launch_ct_direct_expert_alignment_variant.sh GPU VARIANT [OUTPUT_DIR]

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
gpu_id="${1:?physical GPU id is required}"
variant="${2:?variant is required}"
batch_size="${BATCH_SIZE:-1}"
run_tag="ct_direct_${variant}_z4_b${batch_size}_30ep_20260903"
experiment_dir="${3:-${repo_dir}/experiments/${run_tag}}"
pid_file="${experiment_dir}/detached.pid"

if [[ -f "${pid_file}" ]]; then
  previous_pid="$(<"${pid_file}")"
  if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Experiment already appears to be running: pid=${previous_pid}" >&2
    exit 3
  fi
fi

mkdir -p "${experiment_dir}/logs"
nohup setsid env \
  BATCH_SIZE="${batch_size}" \
  S_PARTIAL_MASS_FRACTION="${S_PARTIAL_MASS_FRACTION:-0.5}" \
  RELATIVE_KD_RMS_WEIGHT="${RELATIVE_KD_RMS_WEIGHT:-0.5}" \
  bash "${repo_dir}/shell/run_ct_direct_expert_alignment_variant.sh" \
    "${gpu_id}" "${variant}" "${experiment_dir}" \
  >"${experiment_dir}/detached_launcher.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"

printf 'launched variant=%s gpu=%s pid=%s\n' "${variant}" "${gpu_id}" "${pid}"
printf 'experiment=%s\n' "${experiment_dir}"
printf 'log=%s\n' "${experiment_dir}/detached_launcher.log"

