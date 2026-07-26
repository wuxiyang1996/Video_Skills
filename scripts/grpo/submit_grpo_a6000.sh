#!/usr/bin/env bash
# Submit GRPO collect/train on gamma A6000.
#
# 首轮框架：自定义 HF+PEFT GRPO（FlashAttention-2）。
# 默认算力预算：8×A6000（huge-long / gamma-huge-long）；smoke 仍可 1 卡。
#
# Usage:
#   bash scripts/grpo/submit_grpo_a6000.sh smoke          # 1×A6000, QoS=default
#   PROFILE=8gpu LIVE=1 bash scripts/grpo/submit_grpo_a6000.sh all
#   NUM_GPUS=8 QOS=huge-long bash scripts/grpo/submit_grpo_a6000.sh all
#   STAGE=gpu_train GROUPS_ALREADY=1 bash scripts/grpo/submit_grpo_a6000.sh gpu_train
#
# Fan-out（多作业各 1 卡，占满 ~8 卡预算）:
#   for i in $(seq 0 5); do
#     SHARD_ID=$i SHARD_COUNT=6 LIVE=1 LIMIT=32 \
#       bash scripts/grpo/submit_grpo_a6000.sh live_collect
#   done
#   # 另开 1–2 卡做 gpu_train / 重跑
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
STAGE_ARG="${1:-smoke}"
STAGE="${STAGE:-${STAGE_ARG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/grpo_a6000_${STAGE}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
PROFILE="${PROFILE:-}"   # smoke|1gpu|8gpu|"" (manual NUM_GPUS/QOS)

# Budget profiles (gamma A6000 = 4/node → 8 GPUs ≈ 2 nodes)
case "${PROFILE}" in
  smoke|1gpu)
    QOS="${QOS:-default}"
    GRES="${GRES:-gpu:rtxa6000:1}"
    CPUS="${CPUS:-4}"
    MEM="${MEM:-32G}"
    NODES="${NODES:-1}"
    ;;
  8gpu)
    # Prefer huge-long (8 GPU / 256G); override with QOS=gamma-huge-long if needed.
    QOS="${QOS:-huge-long}"
    GRES="${GRES:-gpu:rtxa6000:8}"
    CPUS="${CPUS:-32}"
    MEM="${MEM:-256G}"
    NODES="${NODES:-2}"
    ;;
  "")
    if [[ "${STAGE}" == "smoke" ]]; then
      QOS="${QOS:-default}"
      GRES="${GRES:-gpu:rtxa6000:1}"
      CPUS="${CPUS:-4}"
      MEM="${MEM:-32G}"
      NODES="${NODES:-1}"
    else
      # Formal GRPO budget: 8×A6000
      QOS="${QOS:-huge-long}"
      GRES="${GRES:-gpu:rtxa6000:${NUM_GPUS:-8}}"
      CPUS="${CPUS:-32}"
      MEM="${MEM:-256G}"
      NODES="${NODES:-2}"
    fi
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE} (use smoke|1gpu|8gpu or leave empty)" >&2
    exit 2
    ;;
esac

WALLTIME="${WALLTIME:-04:00:00}"
NODELIST="${NODELIST:-}"
LIVE="${LIVE:-0}"
MODE="${MODE:-l2_repair}"
K="${K:-4}"
LIMIT="${LIMIT:-8}"
MAX_GROUPS="${MAX_GROUPS:-8}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
ALLOW_SDPA_FALLBACK="${ALLOW_SDPA_FALLBACK:-0}"
SHARD_ID="${SHARD_ID:-}"
SHARD_COUNT="${SHARD_COUNT:-}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

node_args=()
if [[ -n "${NODELIST}" ]]; then
  node_args+=(--nodelist="${NODELIST}")
fi
if [[ -n "${NODES}" ]]; then
  node_args+=(--nodes="${NODES}")
fi

JOB_NAME="vs-grpo-${STAGE}"
if [[ -n "${SHARD_ID}" ]]; then
  JOB_NAME="${JOB_NAME}-s${SHARD_ID}"
fi
echo "Submitting ${JOB_NAME} -> ${OUTPUT_ROOT}"
echo "  qos=${QOS} gres=${GRES} cpus=${CPUS} mem=${MEM} nodes=${NODES:-auto}"

sbatch --parsable \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --gres="${GRES}" --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${WALLTIME}" \
  --output="${LOG_ROOT}/grpo-%j.out" \
  --error="${LOG_ROOT}/grpo-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},STAGE=${STAGE},OUTPUT_ROOT=${OUTPUT_ROOT},LIVE=${LIVE},MODE=${MODE},K=${K},LIMIT=${LIMIT},MAX_GROUPS=${MAX_GROUPS},INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN},ALLOW_SDPA_FALLBACK=${ALLOW_SDPA_FALLBACK},CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo},GRPO_PYTHON=${GRPO_PYTHON:-},SHARD_ID=${SHARD_ID},SHARD_COUNT=${SHARD_COUNT}" \
  "${node_args[@]}" \
  "${REPO_ROOT}/scripts/grpo/run_grpo_worker.sh"
