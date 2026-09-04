#!/usr/bin/env bash
# Submit one-GPU L2 pointwise OPD training (L40S/A6000 48GB minimum).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_pointwise_opd_$(date +%Y%m%d_%H%M%S)}"
DISTILL="${DISTILL:?set DISTILL to opd_distill.jsonl}"
mkdir -p "${OUTPUT_ROOT}/slurm_logs"

jobid=$(sbatch --parsable \
  --job-name="vs-l2-opd" \
  --partition="${PARTITION:-gamma}" --account="${ACCOUNT:-gamma}" --qos="${QOS:-default}" \
  --gres="${GRES:-gpu:l40s:1}" --cpus-per-task="${CPUS:-8}" --mem="${MEM:-64G}" \
  --time="${WALLTIME:-04:00:00}" \
  --output="${OUTPUT_ROOT}/slurm_logs/train-%j.out" \
  --error="${OUTPUT_ROOT}/slurm_logs/train-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},DISTILL=${DISTILL},SFT_ADAPTER=${SFT_ADAPTER:-},CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo},EPOCHS=${EPOCHS:-1},GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4},LEARNING_RATE=${LEARNING_RATE:-1e-5},WARMUP_RATIO=${WARMUP_RATIO:-0.05},SEED=${SEED:-42},DATASET_BALANCED_LOSS=${DATASET_BALANCED_LOSS:-1}" \
  "${REPO_ROOT}/scripts/opd/run_l2_pointwise_opd_train.sh")
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "l2-pointwise-opd -> ${jobid}"
