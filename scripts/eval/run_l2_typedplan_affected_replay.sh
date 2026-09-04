#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PAPER_ROOT}/terminal_replay_typedplan_v2}"
CACHE_DIR="${CACHE_DIR:-${PAPER_ROOT}/executor_cache_grpo_main_v8_relv2_typedplan_v2}"
cd "${REPO_ROOT}"

args=()
for seed in 42 43 44; do
  args+=(--seed "${seed}|${PAPER_ROOT}/grpo_main_v8alpha075_relv2_balanced200_k8_seed${seed}/terminal_samples.jsonl")
done

"${PYTHON}" scripts/eval/replay_l2_typedplan_affected_rollouts.py \
  "${args[@]}" \
  --frozen-l1-glob 'dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json' \
  --frozen-l1-glob 'dataset_clip_wrapper/output/sft_auto_20260713_full_retrieval/**/04_l1_example.json' \
  --frozen-l1-glob 'dataset_clip_wrapper/output/l2_expansion_20260831/**/04_l1_example.json' \
  --split-manifest "${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json" \
  --dataset-root "${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}" \
  --keys-py "${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}" \
  --planner-model openai/gpt-oss-120b \
  --skill-model openai/gpt-oss-120b \
  --cache-dir "${CACHE_DIR}" \
  --output-root "${OUTPUT_ROOT}"
