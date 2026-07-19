#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-Dataset009_CT_OOD}"
DEVICE="${DEVICE:-cuda:0}"
BLOCK_Z="${BLOCK_Z:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
NORM_MODE="${NORM_MODE:-ct}"
CKPT="${CKPT:?Please set CKPT=/path/to/fusion_disentangle_best.pth}"

python run_eval_oodka.py \
  --dataset_name "${DATASET}" \
  --device "${DEVICE}" \
  --block_z "${BLOCK_Z}" \
  --batch_size "${BATCH_SIZE}" \
  --image_size "${IMAGE_SIZE}" \
  --norm_mode "${NORM_MODE}" \
  --distangler_ckpt "${CKPT}"

echo "[Done] OODKA evaluation complete."
