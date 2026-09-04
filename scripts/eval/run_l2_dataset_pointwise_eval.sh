#!/usr/bin/env bash
# Evaluate one L2 adapter on a frozen dataset-specific pointwise dev file.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
ADAPTER="${ADAPTER:?set ADAPTER}"
DEV_JSONL="${DEV_JSONL:?set DEV_JSONL}"
OUTPUT="${OUTPUT:?set OUTPUT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}"
args=(
  --adapter "${ADAPTER}"
  --dev-jsonl "${DEV_JSONL}"
  --output "${OUTPUT}"
  --batch-size "${BATCH_SIZE:-8}"
  --top-k "${TOP_K:-2}"
)
if [[ "${BOUNDARY_ANCHOR_INDEX0:-0}" == "1" ]]; then
  args+=(--boundary-anchor-index0)
fi
if [[ -n "${SCORING_MODE:-}" ]]; then
  args+=(--scoring-mode "${SCORING_MODE}")
fi
"${PYTHON}" -m dataset_clip_wrapper.training.evaluate_l2_pointwise_adapter "${args[@]}"
