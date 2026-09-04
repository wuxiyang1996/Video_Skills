#!/usr/bin/env bash
# Aggregate completed per-seed reports. Runs afterany so failures remain visible.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
SEED_SPECS="${SEED_SPECS:?set SEED_SPECS}"
OUTPUT="${OUTPUT:?set OUTPUT}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${OUTPUT}")"
IFS='^' read -r -a specs <<< "${SEED_SPECS}"
args=()
for spec in "${specs[@]}"; do
  [[ -n "${spec}" ]] && args+=(--seed "${spec}")
done
"${PYTHON}" scripts/eval/aggregate_l2_grpo_seeds.py \
  "${args[@]}" \
  --output "${OUTPUT}"
