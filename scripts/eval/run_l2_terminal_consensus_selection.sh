#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PAPER_ROOT}/terminal_consensus_v1}"
cd "${REPO_ROOT}"

args=()
for seed in 42 43 44; do
  samples="${PAPER_ROOT}/grpo_main_v8alpha075_relv2_balanced200_k8_seed${seed}/terminal_samples.jsonl"
  args+=(--seed "${seed}|${samples}")
done

"${PYTHON}" scripts/eval/select_l2_terminal_consensus_groups.py \
  "${args[@]}" \
  --samples-per-group 8 \
  --target-per-dataset 50 \
  --min-predicted-trainable-rate 0.25 \
  --allowlist "${OUTPUT_ROOT}/exact_balanced100.tsv" \
  --report "${OUTPUT_ROOT}/selection_report.json"
