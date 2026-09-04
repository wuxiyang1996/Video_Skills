#!/usr/bin/env bash
# Build evaluator-labeled, train-only CG/VH pointwise OPD rows.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
FROZEN_L1_GLOBS="${FROZEN_L1_GLOBS:-${FROZEN_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json}|${REPO_ROOT}/dataset_clip_wrapper/output/sft_auto_20260713_full_retrieval/**/04_l1_example.json|${REPO_ROOT}/dataset_clip_wrapper/output/l2_expansion_20260831/**/04_l1_example.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}"
args=(
  --split-manifest "${SPLIT_MANIFEST}"
  --dataset-root "${DATASET_ROOT}"
  --datasets "${DATASETS:-cg_bench,video_holmes}"
  --output-jsonl "${OUTPUT_ROOT}/opd_distill.jsonl"
  --output-report "${OUTPUT_ROOT}/build_report.json"
  --positives-per-example "${POSITIVES_PER_EXAMPLE:-3}"
  --negatives-per-example "${NEGATIVES_PER_EXAMPLE:-3}"
  --min-video-holmes-score "${MIN_VIDEO_HOLMES_SCORE:-0.50}"
  --max-video-holmes-negative-score "${MAX_VIDEO_HOLMES_NEGATIVE_SCORE:-0.05}"
)
IFS='|' read -r -a frozen_l1_patterns <<< "${FROZEN_L1_GLOBS}"
for pattern in "${frozen_l1_patterns[@]}"; do
  [[ -n "${pattern}" ]] && args+=(--frozen-l1-glob "${pattern}")
done
if [[ -n "${LIMIT_PER_DATASET:-}" ]]; then
  args+=(--limit-per-dataset "${LIMIT_PER_DATASET}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "${LIMIT}")
fi
cd "${REPO_ROOT}"
"${PYTHON}" -m trainer.build_l2_dataset_opd "${args[@]}"
"${PYTHON}" -m trainer.train_opd_kl \
  --distill "${OUTPUT_ROOT}/opd_distill.jsonl" \
  --output "${OUTPUT_ROOT}/opd_schema_smoke.json"
