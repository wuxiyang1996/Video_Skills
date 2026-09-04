#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
HELDOUT_ROOT="${HELDOUT_ROOT:-${PAPER_ROOT}/heldout_pointwise_v1}"
RESULTS="${RESULTS:-${HELDOUT_ROOT}/results}"
THREE_SEED_AGGREGATE="${THREE_SEED_AGGREGATE:-${PAPER_ROOT}/grpo_main_v8alpha075_three_seed_aggregate.json}"
OUTPUT="${OUTPUT:-${HELDOUT_ROOT}/three_seed_aggregate.json}"
cd "${REPO_ROOT}"

args=()
for model in sft opd_alpha075 grpo_seed42 grpo_seed43 grpo_seed44; do
  args+=(--model "${model}|${RESULTS}/${model}/cg_bench/eval_report.json|${RESULTS}/${model}/video_holmes/eval_report.json")
done

"${PYTHON}" scripts/eval/aggregate_l2_paper_heldout.py \
  "${args[@]}" \
  --sft-reference "${PAPER_ROOT}/dev_eval_results_clean_v7_frozen_prompt/sft_cg14/report.json" \
  --opd-selection "${PAPER_ROOT}/opd_v8_dev/terminal_qualified_checkpoint_selection_repairfinal_v1.json" \
  --three-seed-aggregate "${THREE_SEED_AGGREGATE}" \
  --output "${OUTPUT}"
