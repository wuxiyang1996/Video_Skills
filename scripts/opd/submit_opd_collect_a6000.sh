#!/usr/bin/env bash
# Submit real-teacher OPD collect (CPU/light; optional 1×A6000 for colocated tools).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/opd_real_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"
LIMIT="${LIMIT:-16}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-default}"
GRES="${GRES:-}"   # leave empty for CPU; set gpu:rtxa6000:1 if desired
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
WALLTIME="${WALLTIME:-04:00:00}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
echo "Submitting OPD collect -> ${OUTPUT_ROOT}"

gres_args=()
if [[ -n "${GRES}" ]]; then
  gres_args+=(--gres="${GRES}")
fi

sbatch --parsable \
  --job-name="vs-opd-collect" \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  "${gres_args[@]}" \
  --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${WALLTIME}" \
  --output="${LOG_ROOT}/opd-%j.out" \
  --error="${LOG_ROOT}/opd-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},LIMIT=${LIMIT},CONDA_ENV=${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}" \
  "${REPO_ROOT}/scripts/opd/run_opd_collect_worker.sh"
