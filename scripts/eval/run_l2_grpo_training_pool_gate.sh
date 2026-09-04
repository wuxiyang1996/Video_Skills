#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
TRAINING_REPORT="${TRAINING_REPORT:?set TRAINING_REPORT}"
OUTPUT="${OUTPUT:?set OUTPUT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
"${CONDA_ENV}/bin/python" scripts/eval/audit_l2_grpo_training_pool.py \
  --training-report "${TRAINING_REPORT}" \
  --output "${OUTPUT}"
