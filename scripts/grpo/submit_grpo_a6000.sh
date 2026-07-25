#!/usr/bin/env bash
# Submit GRPO collect/train on gamma A6000.
#
# 首轮框架选择：自定义 HF+PEFT GRPO（FlashAttention-2），不用 verl / ms-swift。
#
# Usage:
#   bash scripts/grpo/submit_grpo_a6000.sh smoke
#   LIVE=1 bash scripts/grpo/submit_grpo_a6000.sh all
#   STAGE=gpu_train GROUPS_ALREADY=1 bash scripts/grpo/submit_grpo_a6000.sh gpu_train
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
STAGE_ARG="${1:-smoke}"
STAGE="${STAGE:-${STAGE_ARG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/grpo_a6000_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-default}"
GRES="${GRES:-gpu:rtxa6000:1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
WALLTIME="${WALLTIME:-04:00:00}"
NODELIST="${NODELIST:-}"
LIVE="${LIVE:-0}"
MODE="${MODE:-l2_repair}"
K="${K:-4}"
LIMIT="${LIMIT:-8}"
MAX_GROUPS="${MAX_GROUPS:-8}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
ALLOW_SDPA_FALLBACK="${ALLOW_SDPA_FALLBACK:-0}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

node_args=()
if [[ -n "${NODELIST}" ]]; then
  node_args+=(--nodelist="${NODELIST}")
fi

JOB_NAME="vs-grpo-${STAGE}"
echo "Submitting ${JOB_NAME} -> ${OUTPUT_ROOT}"

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${WALLTIME}" \
  --output="${LOG_ROOT}/grpo-%j.out" \
  --error="${LOG_ROOT}/grpo-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},STAGE=${STAGE},OUTPUT_ROOT=${OUTPUT_ROOT},LIVE=${LIVE},MODE=${MODE},K=${K},LIMIT=${LIMIT},MAX_GROUPS=${MAX_GROUPS},INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN},ALLOW_SDPA_FALLBACK=${ALLOW_SDPA_FALLBACK}" \
  "${node_args[@]}" \
  "${REPO_ROOT}/scripts/grpo/run_grpo_worker.sh"
