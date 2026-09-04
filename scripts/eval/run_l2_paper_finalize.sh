#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
HELDOUT_ROOT="${HELDOUT_ROOT:-${PAPER_ROOT}/heldout_pointwise_consensus_repairfinal_v1}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PAPER_ROOT}/paper_artifacts_consensus_repairfinal_v1}"
cd "${REPO_ROOT}"

"${PYTHON}" scripts/eval/finalize_l2_paper_artifacts.py \
  --split-audit "${PAPER_ROOT}/gates_v2_frozen_prompt/split_manifest_video_exclusive_audit.json" \
  --reward-separation "${PAPER_ROOT}/gates_v2_frozen_prompt/terminal_reward_separation_repairfinal_v1.json" \
  --reward-normalization "${PAPER_ROOT}/reward_normalization_audit_consensus_repairfinal_v1.json" \
  --opd-selection "${PAPER_ROOT}/opd_v8_dev/terminal_qualified_checkpoint_selection_repairfinal_v1.json" \
  --grpo-aggregate "${PAPER_ROOT}/grpo_consensus_repairfinal_v1_v8alpha075_three_seed_aggregate.json" \
  --pretest-gate "${PAPER_ROOT}/grpo_consensus_repairfinal_v1_v8alpha075_paper_pretest_release_gate.json" \
  --vh-l1-audit "${HELDOUT_ROOT}/video_holmes/l1_coverage_audit.json" \
  --heldout-aggregate "${HELDOUT_ROOT}/three_seed_aggregate.json" \
  --output-dir "${OUTPUT_DIR}"
