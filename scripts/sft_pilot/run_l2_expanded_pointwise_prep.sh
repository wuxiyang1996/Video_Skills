#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
V5_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5_expanded_20260831/five_lora/l2"
CANDIDATE_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_expanded_20260831/candidate_retrieval"
V7_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v7_expanded_20260831/five_lora/l2"
V8_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v8_pointwise_expanded_20260831/five_lora/l2"
V9_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v9_mixed_expanded_20260831/five_lora/l2"
ROLLOUTS="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/l2_expansion_20260831/finalized/deduplicated_rollouts.jsonl"
DEV_REPORT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_oracle_v5_20260830_011126/l2/candidate_retrieval_eval/qwen3_vl_embedding_2b_dev_fine8s_report.json"
OLD_TRAIN_REPORT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_oracle_v5_20260830_011126/l2/candidate_retrieval_eval/qwen3_vl_embedding_2b_train_fine8s_report.json"

cd "${REPO_ROOT}"
python -m dataset_clip_wrapper.training.merge_l2_visual_candidate_reports \
  --inputs "${OLD_TRAIN_REPORT}" "${CANDIDATE_ROOT}"/train_fine8s_shard{0,1,2,3}.json \
  --output "${CANDIDATE_ROOT}/train_fine8s_report.json"

python -m dataset_clip_wrapper.training.l2_candidate_reranker_v7 \
  --train-jsonl "${V5_ROOT}/train.jsonl" \
  --dev-jsonl "${V5_ROOT}/dev.jsonl" \
  --train-report "${CANDIDATE_ROOT}/train_fine8s_report.json" \
  --dev-report "${DEV_REPORT}" \
  --rollouts "${ROLLOUTS}" \
  --output-dir "${V7_ROOT}"

python -m dataset_clip_wrapper.training.l2_pointwise_reranker_v8 \
  --train-jsonl "${V7_ROOT}/train.jsonl" \
  --dev-jsonl "${V5_ROOT}/dev.jsonl" \
  --label-independent-dev-candidate-report "${DEV_REPORT}" \
  --output-dir "${V8_ROOT}"

python -m dataset_clip_wrapper.training.build_l2_mixed_v9 \
  --selection-root "${V7_ROOT}" \
  --pointwise-root "${V8_ROOT}" \
  --output-dir "${V9_ROOT}"
