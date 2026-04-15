#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-Dataset009_CT_OOD}"
NORM_MODE="${NORM_MODE:-ct}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

python run_preprocess.py \
  --dataset_name "${DATASET}" \
  --norm_mode "${NORM_MODE}" \
  --num_processes "${NUM_PROCESSES}" \
  --include_test

echo "[Done] Preprocessing complete for ${DATASET}."
