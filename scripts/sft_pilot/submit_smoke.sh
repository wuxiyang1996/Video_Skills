#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
LOG_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710/slurm_logs"
mkdir -p "${LOG_ROOT}"

sbatch \
  --parsable \
  --job-name=vs-qwen-smoke \
  --partition=scavenger \
  --account=scavenger \
  --qos=scavenger \
  --nodelist=cml32 \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:45:00 \
  --output="${LOG_ROOT}/smoke-%j.out" \
  --error="${LOG_ROOT}/smoke-%j.err" \
  --export=ALL,DATASET=video_holmes,START_INDEX=0,LIMIT=1,SMOKE=1,PILOT_TAG=pilot_20260710_smoke \
  "${REPO_ROOT}/scripts/sft_pilot/run_local_qwen_worker.sh"
