#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
REPLAY_ROOT="${REPLAY_ROOT:-${PAPER_ROOT}/terminal_replay_typedplan_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PAPER_ROOT}/terminal_consensus_typedplan_v2}"
cd "${REPO_ROOT}"

args=()
for seed in 42 43 44; do
  args+=(--seed "${seed}|${REPLAY_ROOT}/seed${seed}_terminal_samples.corrected.jsonl")
done

"${PYTHON}" scripts/eval/select_l2_terminal_consensus_groups.py \
  "${args[@]}" \
  --samples-per-group 8 \
  --target-per-dataset 50 \
  --min-predicted-trainable-rate 0.25 \
  --source-provenance "${REPLAY_ROOT}/replay_report.json" \
  --allowlist "${OUTPUT_ROOT}/exact_balanced100.tsv" \
  --report "${OUTPUT_ROOT}/selection_report.json"
