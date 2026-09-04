#!/usr/bin/env bash
# Compare fixed-contract OPD terminal baseline against the matching SFT baseline.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
SFT_REPORT="${SFT_REPORT:?set SFT_REPORT}"
OPD_REPORT="${OPD_REPORT:?set OPD_REPORT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
status=0
for dataset in cg_bench video_holmes; do
  "${PYTHON}" scripts/eval/gate_l2_terminal.py \
    --sft "${SFT_REPORT}" \
    --opd "${OPD_REPORT}" \
    --dataset "${dataset}" \
    --output "${OUTPUT_ROOT}/gate_${dataset}.json" || status=1
done
exit "${status}"
