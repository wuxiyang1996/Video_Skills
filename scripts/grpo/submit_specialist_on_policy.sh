#!/usr/bin/env bash
# Submit L2 on-policy GRPO from the frozen SFT adapter.
# Repair is opt-in for smoke diagnostics only; its low-data formal path is OPD.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
SPECIALISTS="${SPECIALISTS:-l2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_grpo_on_policy_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${OUTPUT_ROOT}/slurm_logs"
mkdir -p "${LOG_ROOT}"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
for specialist in ${SPECIALISTS}; do
  jobid=$(sbatch --parsable \
    --job-name="vs-${specialist}-grpo" \
    --partition="${PARTITION:-gamma}" --account="${ACCOUNT:-gamma}" --qos="${QOS:-default}" \
    --gres="${GRES:-gpu:rtxa6000:1}" --cpus-per-task="${CPUS:-4}" --mem="${MEM:-32G}" \
    --time="${WALLTIME:-08:00:00}" \
    --output="${LOG_ROOT}/${specialist}-%j.out" \
    --error="${LOG_ROOT}/${specialist}-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},SPECIALIST=${specialist},OUTPUT_ROOT=${OUTPUT_ROOT},MAX_GROUPS=${MAX_GROUPS:-32},K=${K:-4},PPO_EPOCHS=${PPO_EPOCHS:-2},MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384},EVAL_SAMPLES=${EVAL_SAMPLES:-16},CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}" \
    "${REPO_ROOT}/scripts/grpo/run_specialist_on_policy.sh")
  echo "${specialist} -> ${jobid}"
done
