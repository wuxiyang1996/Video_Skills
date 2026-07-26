#!/usr/bin/env bash
# Real-teacher OPD collect on opd_pool (API-heavy; 0–1 GPU).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
if [[ -x "${CONDA_ENV}/bin/python" ]]; then
  PYTHON="${CONDA_ENV}/bin/python"
else
  PYTHON="${REPO_ROOT}/.venv-qwen35-serve/bin/python"
fi

SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
FROZEN_L1_GLOB="${FROZEN_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json}"
MOTIF_BANK="${MOTIF_BANK:-${REPO_ROOT}/motif/output/pilot_online_motif_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/opd_real_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-16}"
TEACHER_MODEL="${TEACHER_MODEL:-openai/gpt-4.1-mini}"
TEACHER_MODE="${TEACHER_MODE:-auto}"   # soft|ranking|auto
# gpt-4.1-mini is more reliable for strict JSON rankings on OpenRouter;
# deepseek/deepseek-v4-pro remains available via RANKING_MODEL override.
RANKING_MODEL="${RANKING_MODEL:-openai/gpt-4.1-mini}"
RANKING_METHOD="${RANKING_METHOD:-borda}"
PLANNER_MODEL="${PLANNER_MODEL:-openai/gpt-oss-120b}"
KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
hostname

"${PYTHON}" -m trainer.collect_opd_real_teacher \
  --frozen-l1-glob "${FROZEN_L1_GLOB}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --motif-bank "${MOTIF_BANK}" \
  --output-dir "${OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  --teacher-model "${TEACHER_MODEL}" \
  --teacher-mode "${TEACHER_MODE}" \
  --ranking-model "${RANKING_MODEL}" \
  --ranking-method "${RANKING_METHOD}" \
  --planner-model "${PLANNER_MODEL}" \
  --keys-py "${KEYS_PY}"

"${PYTHON}" "${REPO_ROOT}/scripts/opd/gate_teacher_calibration.py" \
  --summary "${OUTPUT_ROOT}/collect_summary.json" \
  --out "${OUTPUT_ROOT}/calibration_gate.json" || true

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
