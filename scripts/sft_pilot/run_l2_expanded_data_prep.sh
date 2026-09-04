#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
TAG="${TAG:-l2_expansion_20260831}"
EXPANSION_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${TAG}"
FINAL_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/${TAG}/finalized"
V5_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5_expanded_20260831/five_lora/l2"

cd "${REPO_ROOT}"
python scripts/sft_pilot/merge_rollout_sources.py \
  --inputs dataset_clip_wrapper/output/sft_cold_start/controller_expansion_20260721/finalized/deduplicated_rollouts.jsonl \
  --input-roots "${EXPANSION_ROOT}" \
  --output "${FINAL_ROOT}/deduplicated_rollouts.jsonl" \
  --report "${FINAL_ROOT}/merge_report.json"

python -m dataset_clip_wrapper.training.l2_oracle_retrieval_v5 \
  --rollouts "${FINAL_ROOT}/deduplicated_rollouts.jsonl" \
  --split-manifest dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json \
  --cg-bench /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cgbench.json \
  --frozen-dev-jsonl dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5/five_lora/l2/dev.jsonl \
  --output-dir "${V5_ROOT}"
