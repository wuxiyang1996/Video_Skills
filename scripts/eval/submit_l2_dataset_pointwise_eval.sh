#!/usr/bin/env bash
# Submit dataset-specific L2 pointwise/evidence evaluation.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
ADAPTER="${ADAPTER:?set ADAPTER}"
DEV_JSONL="${DEV_JSONL:?set DEV_JSONL}"
OUTPUT="${OUTPUT:?set OUTPUT}"
LOG_ROOT="${LOG_ROOT:-$(dirname "${OUTPUT}")/slurm_logs}"
mkdir -p "${LOG_ROOT}" "$(dirname "${OUTPUT}")"

sbatch_args=(
  --parsable
  --job-name="${JOB_NAME:-vs-l2-dev-eval}"
  --partition="${PARTITION:-scavenger}"
  --account="${ACCOUNT:-scavenger}"
  --qos="${QOS:-scavenger}"
  --gres="${GRES:-gpu:l40s:1}"
  --cpus-per-task="${CPUS:-6}"
  --mem="${MEM:-48G}"
  --time="${WALLTIME:-02:00:00}"
  --output="${LOG_ROOT}/eval-%j.out"
  --error="${LOG_ROOT}/eval-%j.err"
  --export="ALL,REPO_ROOT=${REPO_ROOT},ADAPTER=${ADAPTER},DEV_JSONL=${DEV_JSONL},OUTPUT=${OUTPUT},TOP_K=${TOP_K:-2},BATCH_SIZE=${BATCH_SIZE:-8},BOUNDARY_ANCHOR_INDEX0=${BOUNDARY_ANCHOR_INDEX0:-0},SCORING_MODE=${SCORING_MODE:-},CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
)
if [[ -n "${DEPENDENCY:-}" ]]; then
  sbatch_args+=(--dependency="${DEPENDENCY}")
fi
jobid=$(sbatch "${sbatch_args[@]}" "${REPO_ROOT}/scripts/eval/run_l2_dataset_pointwise_eval.sh")
echo "l2-dataset-eval -> ${jobid}"
