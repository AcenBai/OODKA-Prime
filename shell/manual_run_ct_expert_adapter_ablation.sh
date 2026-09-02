#!/usr/bin/env bash
set -euo pipefail

# Compare the legacy and direct/shared Expert adapters under the same CT setup.
#
# Usage:
#   BATCH_SIZE=2 \
#   bash shell/manual_run_ct_expert_adapter_ablation.sh \
#     [physical_gpu_id] [legacy|direct_shared] [relative_kd_expert_weight] [output_dir]
#
# relative_kd_expert_weight=0 disables bidirectional Relative KD entirely.
# A positive value enables it and sets the student-to-expert multiplier.

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:-0}"
variant="${2:-direct_shared}"
relative_weight="${3:-0}"
batch_size="${BATCH_SIZE:-1}"
batch_tag=""
if [[ "${batch_size}" != "1" ]]; then
  batch_tag="_b${batch_size}"
fi
run_tag="ct_expert_${variant}_relative${relative_weight}${batch_tag}_no_roi_noaug_30ep_20260829"
experiment_dir="${4:-${repo_dir}/experiments/${run_tag}}"
shared_scale="${repo_dir}/experiments/spatial_beta_p07_ct_f0_30ep_20260730/analysis/mechanism_v3/heart_1004_z0079/representation/color_scales.json"
test_dir="${experiment_dir}/test_best_adjacent"
visual_dir="${experiment_dir}/analysis/mechanism_v3_matched_scale"

if [[ "${variant}" != "legacy" && "${variant}" != "direct_shared" ]]; then
  echo "variant must be legacy or direct_shared" >&2
  exit 2
fi
if ! [[ "${relative_weight}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "relative_kd_expert_weight must be a non-negative number" >&2
  exit 2
fi
if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

test -f "${shared_scale}"
mkdir -p "${experiment_dir}/logs" "${experiment_dir}/analysis" \
  "${test_dir}" "${visual_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${experiment_dir}/mpl_cache"

relative_args=()
if [[ "${relative_weight}" != "0" && "${relative_weight}" != "0.0" ]]; then
  relative_args+=(
    --relative_kd
    --relative_kd_expert_weight "${relative_weight}"
  )
fi

cd "${repo_dir}"

"${python_bin}" run_train.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --block_z 4 \
  --batch_size "${batch_size}" \
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
  --expert_adapter_variant "${variant}" \
  --val_every_epochs 5 \
  --device cuda:0 \
  --output_dir "${experiment_dir}" \
  "${relative_args[@]}" \
  2>&1 | tee "${experiment_dir}/logs/train_console.log"

checkpoint="${experiment_dir}/fusion_disentangle_best.pth"
test -f "${checkpoint}"

"${python_bin}" run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --split test \
  --block_z 4 \
  --batch_size 1 \
  --image_size 512 \
  --norm_mode ct \
  --pseudo_rgb_mode adjacent \
  --device cuda:0 \
  --distangler_ckpt "${checkpoint}" \
  --out_dir "${test_dir}" \
  2>&1 | tee "${experiment_dir}/logs/eval_test.log"

for case_spec in "heart_1004:79" "heart_1014:191"; do
  case_id="${case_spec%%:*}"
  slice_index="${case_spec##*:}"
  "${python_bin}" scripts/visualize_mechanism_v3.py \
    --checkpoint "${checkpoint}" \
    --case_id "${case_id}" \
    --split val \
    --slice_index "${slice_index}" \
    --selection all_classes \
    --device cuda:0 \
    --block_z 4 \
    --shared_color_scales "${shared_scale}" \
    --output_root "${visual_dir}" \
    2>&1 | tee "${experiment_dir}/logs/visualize_${case_id}.log"
done

touch "${experiment_dir}/PIPELINE_COMPLETE"
