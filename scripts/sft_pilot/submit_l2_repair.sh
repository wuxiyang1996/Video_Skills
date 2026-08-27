#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
REPAIR_TAG="${REPAIR_TAG:-l2_repair_20260713_free}"
LOG_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/${REPAIR_TAG}/slurm_logs"
mkdir -p "${LOG_ROOT}"

WALLTIME="${WALLTIME:-12:00:00}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-default}"
GRES="${GRES:-gpu:l40s:1}"
NODELIST="${NODELIST:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-9}"
EXAMPLE_OFFSET="${EXAMPLE_OFFSET:-0}"
GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-120b:free}"
DATASETS="${DATASETS:-cg_bench}"

node_args=()
if [[ -n "${NODELIST}" ]]; then node_args+=(--nodelist="${NODELIST}"); fi

sbatch --parsable \
  --job-name="vs-l2-repair" \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --gres="${GRES}" --cpus-per-task=4 --mem=32G --time="${WALLTIME}" \
  --output="${LOG_ROOT}/l2-repair-%j.out" \
  --error="${LOG_ROOT}/l2-repair-%j.err" \
  --export="ALL,REPAIR_TAG=${REPAIR_TAG},MAX_EXAMPLES=${MAX_EXAMPLES},EXAMPLE_OFFSET=${EXAMPLE_OFFSET},GPTOSS_MODEL=${GPTOSS_MODEL}" \
  "${node_args[@]}" \
  "${REPO_ROOT}/scripts/sft_pilot/run_l2_repair_worker.sh"
