#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
stable_checks=0

cd "${repo_dir}"
while (( stable_checks < 3 )); do
  gpu_line="$(nvidia-smi --id=2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  gpu_memory="${gpu_line%%,*}"
  gpu_utilization="${gpu_line##*,}"
  gpu_memory="${gpu_memory// /}"
  gpu_utilization="${gpu_utilization// /}"
  if (( gpu_memory < 2000 && gpu_utilization < 10 )); then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi
  printf '[%s] GPU2 memory=%sMiB util=%s%% idle_checks=%s/3\n' \
    "$(date '+%F %T')" "${gpu_memory}" "${gpu_utilization}" "${stable_checks}"
  if (( stable_checks < 3 )); then
    sleep 30
  fi
done

exec bash shell/run_whs_mri_roi3d_z4_b2_30ep_20260824.sh
