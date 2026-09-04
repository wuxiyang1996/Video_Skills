#!/usr/bin/env bash
# Materialize frozen pointwise heldout rows only after the canonical pre-test gate passes.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
RELEASE_GATE="${RELEASE_GATE:-${PAPER_ROOT}/paper_pretest_release_gate.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PAPER_ROOT}/heldout_pointwise_v1}"
VH_HELDOUT_L1_GLOB="${VH_HELDOUT_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_vh_heldout_l1_v3/video_holmes/test/**/04_l1_example.json}"
VH_HELDOUT_EXPECTED_VIDEOS="${VH_HELDOUT_EXPECTED_VIDEOS:-270}"
VH_HELDOUT_CLIP_MODEL="${VH_HELDOUT_CLIP_MODEL:-Qwen/Qwen3.5-9B}"
VH_HELDOUT_SAMPLED_FRAMES="${VH_HELDOUT_SAMPLED_FRAMES:-4}"
VH_HELDOUT_ANCHOR_REPASS_FRAMES="${VH_HELDOUT_ANCHOR_REPASS_FRAMES:-6}"
VH_HELDOUT_MAX_TOKENS="${VH_HELDOUT_MAX_TOKENS:-1600}"

[[ -f "${RELEASE_GATE}" ]] || { echo "missing pre-test release gate: ${RELEASE_GATE}" >&2; exit 2; }
[[ "$(jq -r '.passed // false' "${RELEASE_GATE}")" == "true" ]] || {
  echo "pre-test release gate did not pass: ${RELEASE_GATE}" >&2
  exit 3
}
[[ -f "${SPLIT_MANIFEST}" ]] || { echo "missing split manifest: ${SPLIT_MANIFEST}" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}/cg_bench" "${OUTPUT_ROOT}/video_holmes"
cd "${REPO_ROOT}"

"${PYTHON}" scripts/eval/audit_l2_video_holmes_heldout_l1.py \
  --split-manifest "${SPLIT_MANIFEST}" \
  --frozen-l1-glob "${VH_HELDOUT_L1_GLOB}" \
  --expected-count "${VH_HELDOUT_EXPECTED_VIDEOS}" \
  --expected-clip-model "${VH_HELDOUT_CLIP_MODEL}" \
  --expected-sampled-frames "${VH_HELDOUT_SAMPLED_FRAMES}" \
  --expected-anchor-repass-frames "${VH_HELDOUT_ANCHOR_REPASS_FRAMES}" \
  --expected-max-tokens "${VH_HELDOUT_MAX_TOKENS}" \
  --output "${OUTPUT_ROOT}/video_holmes/l1_coverage_audit.json"

common_args=(
  --frozen-l1-glob 'dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json'
  --frozen-l1-glob 'dataset_clip_wrapper/output/sft_auto_20260713_full_retrieval/**/04_l1_example.json'
  --frozen-l1-glob 'dataset_clip_wrapper/output/l2_expansion_20260831/**/04_l1_example.json'
  --frozen-l1-glob "${VH_HELDOUT_L1_GLOB}"
  --split-manifest "${SPLIT_MANIFEST}"
  --dataset-root "${DATASET_ROOT}"
  --split-role heldout_test
)

"${PYTHON}" trainer/build_l2_dataset_dev_eval.py \
  "${common_args[@]}" \
  --datasets cg_bench \
  --output-jsonl "${OUTPUT_ROOT}/cg_bench/pointwise.jsonl" \
  --output-report "${OUTPUT_ROOT}/cg_bench/build_report.json"

"${PYTHON}" trainer/build_l2_dataset_dev_eval.py \
  "${common_args[@]}" \
  --datasets video_holmes \
  --output-jsonl "${OUTPUT_ROOT}/video_holmes/pointwise.jsonl" \
  --output-report "${OUTPUT_ROOT}/video_holmes/build_report.json"
