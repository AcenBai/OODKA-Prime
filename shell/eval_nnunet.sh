#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-Dataset009_CT_OOD}"
DEVICE="${DEVICE:-cuda:0}"

python run_eval_nnunet.py \
  --dataset_name "${DATASET}" \
  --device "${DEVICE}"

echo "[Done] nnUNet evaluation complete."
