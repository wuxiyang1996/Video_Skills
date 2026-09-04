#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
NUM_SHARDS="${NUM_SHARDS:-4}"
V5_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5_expanded_20260831/five_lora/l2"
DEV_JSONL="${DEV_JSONL:-${V5_ROOT}/train.jsonl}"
ROLLOUTS="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/l2_expansion_20260831/finalized/deduplicated_rollouts.jsonl"
OUTPUT_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_expanded_20260831/candidate_retrieval"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT="${OUTPUT:-${OUTPUT_ROOT}/train_fine8s_shard${SHARD_INDEX}.json}"

export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
exec .venv-qwen35-serve/bin/python -m dataset_clip_wrapper.training.evaluate_l2_visual_candidate_retrieval \
  --rollouts "${ROLLOUTS}" \
  --dev-jsonl "${DEV_JSONL}" \
  --model Qwen/Qwen3-VL-Embedding-2B \
  --output "${OUTPUT}" \
  --batch-size "${BATCH_SIZE}" --num-frames 4 --max-side 448 \
  --fine-window-sec 8 --fine-stride-sec 4 \
  --num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX}"
