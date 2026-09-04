#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
ADAPTER="${ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v9_expanded_20260831/l2/pilot/adapter}"
DEV_JSONL="${DEV_JSONL:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v8_pointwise_expanded_20260831/five_lora/l2/dev_label_independent.jsonl}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v9_expanded_20260831/l2/dev_label_independent/report.json}"

export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT}")"
cd "${REPO_ROOT}"
args=(
  --adapter "${ADAPTER}"
  --dev-jsonl "${DEV_JSONL}"
  --output "${OUTPUT}"
  --batch-size 8
)
if [[ "${BOUNDARY_ANCHOR_INDEX0:-0}" == "1" ]]; then
  args+=(--boundary-anchor-index0)
fi
exec .venv-qwen35-serve/bin/python -m dataset_clip_wrapper.training.evaluate_l2_pointwise_adapter "${args[@]}"
