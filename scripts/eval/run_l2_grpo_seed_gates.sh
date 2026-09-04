#!/usr/bin/env bash
# Run all paper-facing dev gates for one completed L2 GRPO seed.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
ADAPTER="${ADAPTER:?set ADAPTER}"
CG_REPORT="${CG_REPORT:?set CG_REPORT}"
VH_REPORT="${VH_REPORT:?set VH_REPORT}"
TERMINAL_REPORT="${TERMINAL_REPORT:?set TERMINAL_REPORT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SFT_CG_REPORT="${SFT_CG_REPORT:?set SFT_CG_REPORT}"
SFT_VH_REPORT="${SFT_VH_REPORT:?set SFT_VH_REPORT}"
TERMINAL_BASELINE_REPORT="${TERMINAL_BASELINE_REPORT:?set TERMINAL_BASELINE_REPORT}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

status=0
"${PYTHON}" scripts/eval/select_l2_opd_checkpoint.py \
  --sft-cg-report "${SFT_CG_REPORT}" \
  --sft-vh-report "${SFT_VH_REPORT}" \
  --candidate "grpo_seed|1.0|${ADAPTER}|${CG_REPORT}|${VH_REPORT}" \
  --output "${OUTPUT_ROOT}/pointwise_preservation_gate.json" || status=1
"${PYTHON}" scripts/eval/gate_l2_terminal.py \
  --sft "${TERMINAL_BASELINE_REPORT}" \
  --opd "${TERMINAL_REPORT}" \
  --dataset cg_bench \
  --output "${OUTPUT_ROOT}/gate_cg_bench.json" || status=1
"${PYTHON}" scripts/eval/gate_l2_terminal.py \
  --sft "${TERMINAL_BASELINE_REPORT}" \
  --opd "${TERMINAL_REPORT}" \
  --dataset video_holmes \
  --output "${OUTPUT_ROOT}/gate_video_holmes.json" || status=1
exit "${status}"
