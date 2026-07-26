#!/usr/bin/env bash
# Fan-out live collect across N×1-GPU jobs, then merge + optional gpu_train.
#
# Usage:
#   bash scripts/grpo/submit_grpo_shard_fanout.sh
#   SHARD_COUNT=4 LIMIT=8 K=4 bash scripts/grpo/submit_grpo_shard_fanout.sh
#   TRAIN_AFTER=1 MAX_GROUPS=32 bash scripts/grpo/submit_grpo_shard_fanout.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
SHARD_COUNT="${SHARD_COUNT:-4}"
LIMIT="${LIMIT:-8}"          # per-shard example cap → total ≈ SHARD_COUNT * LIMIT
K="${K:-4}"
TRAIN_AFTER="${TRAIN_AFTER:-1}"
MAX_GROUPS="${MAX_GROUPS:-0}"
FORCE_EXPLORE="${FORCE_EXPLORE:-1}"
EXPLORE_TOP_K="${EXPLORE_TOP_K:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/grpo_a6000_shard_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}/slurm_logs"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "Submitting ${SHARD_COUNT} shards × LIMIT=${LIMIT} K=${K} (≈$((SHARD_COUNT * LIMIT)) examples)"

job_ids=()
for i in $(seq 0 $((SHARD_COUNT - 1))); do
  jid=$(
    PROFILE=1gpu LIVE=1 STAGE=live_collect \
      SHARD_ID=$i SHARD_COUNT="${SHARD_COUNT}" \
      LIMIT="${LIMIT}" K="${K}" \
      FORCE_EXPLORE="${FORCE_EXPLORE}" EXPLORE_TOP_K="${EXPLORE_TOP_K}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash "${REPO_ROOT}/scripts/grpo/submit_grpo_a6000.sh" live_collect | tail -n1
  )
  job_ids+=("${jid}")
  echo "  shard ${i} -> job ${jid}"
done

dep=$(IFS=:; echo "${job_ids[*]}")
echo "Merge dependency: afterok:${dep}"

MERGE_JOB=$(sbatch --parsable \
  --job-name="vs-grpo-merge" \
  --partition="${PARTITION:-gamma}" --account="${ACCOUNT:-gamma}" --qos="${QOS:-default}" \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --dependency="afterok:${dep}" \
  --output="${OUTPUT_ROOT}/slurm_logs/merge-%j.out" \
  --error="${OUTPUT_ROOT}/slurm_logs/merge-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},STAGE=merge_collect,OUTPUT_ROOT=${OUTPUT_ROOT},LIVE=0,MODE=${MODE:-l2_repair},K=${K},LIMIT=${LIMIT},MAX_GROUPS=${MAX_GROUPS},INSTALL_FLASH_ATTN=0,ALLOW_SDPA_FALLBACK=0,CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}" \
  "${REPO_ROOT}/scripts/grpo/run_grpo_worker.sh")
echo "merge -> ${MERGE_JOB}"

if [[ "${TRAIN_AFTER}" == "1" ]]; then
  TRAIN_JOB=$(sbatch --parsable \
    --job-name="vs-grpo-train" \
    --partition="${PARTITION:-gamma}" --account="${ACCOUNT:-gamma}" --qos="${QOS:-default}" \
    --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
    --dependency="afterok:${MERGE_JOB}" \
    --output="${OUTPUT_ROOT}/slurm_logs/train-%j.out" \
    --error="${OUTPUT_ROOT}/slurm_logs/train-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},STAGE=gpu_train,OUTPUT_ROOT=${OUTPUT_ROOT},LIVE=0,MODE=${MODE:-l2_repair},K=${K},LIMIT=${LIMIT},MAX_GROUPS=${MAX_GROUPS},INSTALL_FLASH_ATTN=0,ALLOW_SDPA_FALLBACK=0,CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}" \
    "${REPO_ROOT}/scripts/grpo/run_grpo_worker.sh")
  echo "train -> ${TRAIN_JOB}"
fi

echo "Done submitting fan-out under ${OUTPUT_ROOT}"
