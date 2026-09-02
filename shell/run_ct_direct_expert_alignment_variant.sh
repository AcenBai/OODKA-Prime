#!/usr/bin/env bash
set -euo pipefail

# Foreground worker for one CT-OOD Direct-Expert alignment ablation.
# Prefer launch_ct_direct_expert_alignment_variant.sh for detached execution.
#
# Usage:
#   bash shell/run_ct_direct_expert_alignment_variant.sh GPU VARIANT [OUTPUT_DIR]
#
# VARIANT:
#   no_relative
#   relative_no_expert_ortho
#   relative_rms
#   relative_capacity_s
#   relative_p_only

repo_dir="/data4/baihexiang/SegMan/spatial_combination"
python_bin="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"
gpu_id="${1:?physical GPU id is required}"
variant="${2:?variant is required}"
batch_size="${BATCH_SIZE:-1}"
partial_mass="${S_PARTIAL_MASS_FRACTION:-0.5}"
rms_weight="${RELATIVE_KD_RMS_WEIGHT:-0.5}"
run_tag="ct_direct_${variant}_z4_b${batch_size}_30ep_20260903"
experiment_dir="${3:-${repo_dir}/experiments/${run_tag}}"
shared_scale="${repo_dir}/experiments/20260730_milestone1_OTvisual_CT30epochs/analysis/mechanism_v3/heart_1004_z0079/representation/color_scales.json"
test_dir="${experiment_dir}/test_best_adjacent"
visual_dir="${experiment_dir}/analysis/mechanism_v3_matched_scale"

if ! [[ "${batch_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

variant_args=()
case "${variant}" in
  no_relative)
    ;;
  relative_no_expert_ortho)
    variant_args+=(
      --relative_kd
      --relative_kd_expert_weight 1.0
      --relative_kd_branches both
      --expert_ortho_weight 0.0
    )
    ;;
  relative_rms)
    variant_args+=(
      --relative_kd
      --relative_kd_expert_weight 1.0
      --relative_kd_branches both
      --relative_kd_rms_weight "${rms_weight}"
    )
    ;;
  relative_capacity_s)
    variant_args+=(
      --relative_kd
      --relative_kd_expert_weight 1.0
      --relative_kd_branches both
      --s_transport_mode capacity_partial
      --s_partial_mass_fraction "${partial_mass}"
    )
    ;;
  relative_p_only)
    variant_args+=(
      --relative_kd
      --relative_kd_expert_weight 1.0
      --relative_kd_branches p
    )
    ;;
  *)
    echo "Unknown variant: ${variant}" >&2
    exit 2
    ;;
esac

test -x "${python_bin}"
test -f "${shared_scale}"
if [[ -e "${experiment_dir}/PIPELINE_COMPLETE" ]]; then
  echo "Refusing to overwrite completed experiment: ${experiment_dir}" >&2
  exit 3
fi
mkdir -p "${experiment_dir}/logs" "${test_dir}" "${visual_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export HF_HOME="/data3/baihexiang/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${experiment_dir}/mpl_cache"

cd "${repo_dir}"
git rev-parse HEAD >"${experiment_dir}/source_commit.txt"
printf '%s\n' "${variant}" >"${experiment_dir}/variant.txt"

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
  --expert_adapter_variant direct_shared \
  --val_every_epochs 5 \
  --device cuda:0 \
  --output_dir "${experiment_dir}" \
  "${variant_args[@]}" \
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

