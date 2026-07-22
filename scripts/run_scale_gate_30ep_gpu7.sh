#!/usr/bin/env bash
set -euo pipefail

ROOT="/data4/baihexiang/SegMan/OODKA"
PYTHON="/data4/baihexiang/conda_envs/biomedparse_v2/bin/python"

RUN_TAG="${RUN_TAG:-scale_gate_fold0_30ep_$(date +%Y%m%d_%H%M%S)}"
EXP_DIR="${ROOT}/outputs/oodka_ot_experiments/${RUN_TAG}"
ANALYSIS_DIR="${EXP_DIR}/analysis"

# 只让程序看到物理7号GPU；程序内部对应cuda:0
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${EXP_DIR}/mpl_cache"

mkdir -p "${EXP_DIR}/logs"
mkdir -p "${ANALYSIS_DIR}"
cd "${ROOT}"

echo "Experiment: ${RUN_TAG}"
echo "Output:     ${EXP_DIR}"
nvidia-smi -i 7

# ============================================================
# 1. 30轮训练
# ============================================================

"${PYTHON}" run_train.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --block_z 6 \
  --batch_size 1 \
  --n_epochs 30 \
  --device cuda:0 \
  --output_dir "${EXP_DIR}" \
  --route_warmup_epochs 5 \
  --p_ot_start_epoch 2 \
  --s_ot_start_epoch 3 \
  --ot_warmup_epochs 5 \
  --val_every_epochs 1 \
  2>&1 | tee "${EXP_DIR}/logs/train_console.log"

FULL_CKPT="${EXP_DIR}/fusion_disentangle_best.pth"
STUDENT_CKPT="${EXP_DIR}/student_deploy_best.pth"

test -f "${FULL_CKPT}"
test -f "${STUDENT_CKPT}"

# ============================================================
# 2. 整卷validation评估
# ============================================================

"${PYTHON}" run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --split val \
  --block_z 6 \
  --batch_size 1 \
  --device cuda:0 \
  --distangler_ckpt "${STUDENT_CKPT}" \
  --out_dir "${ANALYSIS_DIR}/val_full_volume" \
  2>&1 | tee "${EXP_DIR}/logs/eval_val.log"

# ============================================================
# 3. 整卷test评估
# ============================================================

"${PYTHON}" run_eval_oodka.py \
  --dataset_name Dataset009_CT_OOD \
  --fold 0 \
  --split test \
  --block_z 6 \
  --batch_size 1 \
  --device cuda:0 \
  --distangler_ckpt "${STUDENT_CKPT}" \
  --out_dir "${ANALYSIS_DIR}/test_full_volume" \
  2>&1 | tee "${EXP_DIR}/logs/eval_test.log"

# ============================================================
# 4. 每个prompt、每一层的Beta gate分析
# ============================================================

"${PYTHON}" scripts/analyze_beta_router.py \
  --checkpoint "${STUDENT_CKPT}" \
  --device cuda:0 \
  --output "${ANALYSIS_DIR}/beta_router.json" \
  2>&1 | tee "${EXP_DIR}/logs/analyze_beta_router.log"

# ============================================================
# 5. P/S特征热图与定量指标
# heart_1004可以替换成其他val case
# ============================================================

"${PYTHON}" scripts/analyze_ps_features.py \
  --checkpoint "${STUDENT_CKPT}" \
  --case_id heart_1004 \
  --split val \
  --block_z 6 \
  --device cuda:0 \
  --output_dir "${ANALYSIS_DIR}/ps_features" \
  2>&1 | tee "${EXP_DIR}/logs/analyze_ps_features.log"

# ============================================================
# 6. UOT拒绝机制压力测试
# 需要包含nnUNet adapter与OT模块的完整checkpoint
# ============================================================

"${PYTHON}" scripts/stress_test_uot.py \
  --checkpoint "${FULL_CKPT}" \
  --case_id heart_1004 \
  --block_z 6 \
  --device cuda:0 \
  --output "${ANALYSIS_DIR}/uot_stress.json" \
  2>&1 | tee "${EXP_DIR}/logs/stress_test_uot.log"

echo
echo "Experiment complete."
echo "Experiment directory: ${EXP_DIR}"
echo "Val summary:  ${ANALYSIS_DIR}/val_full_volume/summary.json"
echo "Test summary: ${ANALYSIS_DIR}/test_full_volume/summary.json"
echo "Gate report:  ${ANALYSIS_DIR}/beta_router.json"
echo "P/S maps:     ${ANALYSIS_DIR}/ps_features"
echo "UOT report:   ${ANALYSIS_DIR}/uot_stress.json"