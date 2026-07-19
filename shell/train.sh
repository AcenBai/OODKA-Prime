#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-Dataset009_CT_OOD}"
DEVICE="${DEVICE:-cuda:0}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BLOCK_Z="${BLOCK_Z:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NORM_MODE="${NORM_MODE:-ct}"

python run_train.py \
  --dataset_name "${DATASET}" \
  --device "${DEVICE}" \
  --n_epochs "${N_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --block_z "${BLOCK_Z}" \
  --image_size "${IMAGE_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --norm_mode "${NORM_MODE}"

echo "[Done] Training complete."
