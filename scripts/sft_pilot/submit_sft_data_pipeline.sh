#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PIPELINE_TAG="${PIPELINE_TAG:-auto_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/${PIPELINE_TAG}/slurm_logs"
mkdir -p "${LOG_ROOT}"

WALLTIME="${WALLTIME:-12:00:00}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-default}"

sbatch --parsable \
  --job-name="vs-sft-pipeline" \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --cpus-per-task=2 --mem=16G --time="${WALLTIME}" \
  --output="${LOG_ROOT}/pipeline-%j.out" \
  --error="${LOG_ROOT}/pipeline-%j.err" \
  --export="ALL,PIPELINE_TAG=${PIPELINE_TAG},TOTAL_REPAIR_EXAMPLES=${TOTAL_REPAIR_EXAMPLES:-100},LANES=${LANES:-20},EXAMPLES_PER_LANE=${EXAMPLES_PER_LANE:-5},GPTOSS_MODEL=${GPTOSS_MODEL:-openai/gpt-oss-120b}" \
  "${REPO_ROOT}/scripts/sft_pilot/run_sft_data_pipeline.sh"
