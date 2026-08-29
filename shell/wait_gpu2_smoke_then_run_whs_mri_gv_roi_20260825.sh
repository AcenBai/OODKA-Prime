#!/usr/bin/env bash
set -euo pipefail

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
smoke_dir="/tmp/whs_mri_gv_roi_smoke_20260825"
wait_log="${repo_dir}/experiments/whs_mri_gv_roi_z4_b2_promptmean_aug_30ep_20260825.wait.log"

cd "${repo_dir}"
stable=0
while (( stable < 3 )); do
  read -r util memory < <(
    nvidia-smi --id=2 \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
  )
  printf '%s gpu=2 util=%s memory=%s stable=%s/3\n' \
    "$(date '+%F %T')" "${util}" "${memory}" "${stable}" | tee -a "${wait_log}"
  if (( util <= 10 && memory <= 5000 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  (( stable >= 3 )) || sleep 20
done

"${python_bin}" run_train_whs_roi.py \
  --roi_strategy great_vessel \
  --dataset_name Dataset010_WHS_MRI_OOD \
  --device cuda:2 \
  --n_epochs 1 \
  --warmup_epochs 0 \
  --block_z 4 \
  --batch_size 2 \
  --image_size 320 \
  --pseudo_rgb_mode adjacent \
  --roi_prompt_loss_reduction prompt_mean \
  --num_workers 0 \
  --raw_cache_cases 1 \
  --roi_threshold 0.2 \
  --roi_expand 1.4 \
  --roi_fallback full \
  --roi_refresh_every 0 \
  --val_every_epochs 1 \
  --train_case_limit 1 \
  --val_case_limit 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --output_dir "${smoke_dir}"

exec bash shell/run_whs_mri_gv_roi_z4_b2_30ep_20260825.sh
