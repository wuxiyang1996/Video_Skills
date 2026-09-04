#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
MINING_REPORT="${MINING_REPORT:-${PAPER_ROOT}/retrieval_mining_opd_v8alpha075_relv2_balanced420_k8_r3_seed42/mining_report.json}"
THREE_SEED_AGGREGATE="${THREE_SEED_AGGREGATE:-${PAPER_ROOT}/grpo_main_v8alpha075_three_seed_aggregate.json}"
OUTPUT="${OUTPUT:-${PAPER_ROOT}/paper_pretest_release_gate.json}"
REWARD_SEPARATION="${REWARD_SEPARATION:-${PAPER_ROOT}/gates_v2_frozen_prompt/terminal_reward_separation_repairfinal_v1.json}"
OPD_TERMINAL_SELECTION="${OPD_TERMINAL_SELECTION:-${PAPER_ROOT}/opd_v8_dev/terminal_qualified_checkpoint_selection_repairfinal_v1.json}"
PYTHON="${PYTHON:-python}"
cd "${REPO_ROOT}"

"${PYTHON}" scripts/eval/audit_l2_paper_pretest.py \
  --split-audit "${PAPER_ROOT}/gates_v2_frozen_prompt/split_manifest_video_exclusive_audit.json" \
  --reward-separation "${REWARD_SEPARATION}" \
  --opd-selection "${PAPER_ROOT}/opd_v8_dev/checkpoint_selection.json" \
  --opd-terminal-selection "${OPD_TERMINAL_SELECTION}" \
  --mining "${MINING_REPORT}" \
  --pilot-pointwise-gate "${PAPER_ROOT}/grpo_pilot_v8alpha075_dev/pointwise_preservation_gate.json" \
  --pilot-cg-gate "${PAPER_ROOT}/terminal_dev_grpo_pilot_v8alpha075_core10x8_pt09_relv2_seed42/gate_cg_bench.json" \
  --pilot-vh-gate "${PAPER_ROOT}/terminal_dev_grpo_pilot_v8alpha075_core10x8_pt09_relv2_seed42/gate_video_holmes.json" \
  --three-seed-aggregate "${THREE_SEED_AGGREGATE}" \
  --output "${OUTPUT}"
