#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
DEV_REPORT="${DEV_REPORT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v9_expanded_20260831/l2/dev_label_independent/report.json}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v9_expanded_20260831/l2/pilot/adapter}"
GATE_ROOT="${GATE_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_terminal_gate_v9_20260831}"
MIN_RECALL="${MIN_RECALL:-0.60}"
MIN_HIT="${MIN_HIT:-0.60}"
MIN_TERMINAL="${MIN_TERMINAL:-0.10}"

mkdir -p "${GATE_ROOT}"
recall="$(jq -r '.metrics.pointwise_top2.mean_recall' "${DEV_REPORT}")"
hit="$(jq -r '.metrics.pointwise_top2.hit_rate' "${DEV_REPORT}")"
if ! awk -v value="${recall}" -v threshold="${MIN_RECALL}" 'BEGIN { exit !(value >= threshold) }' \
  || ! awk -v value="${hit}" -v threshold="${MIN_HIT}" 'BEGIN { exit !(value >= threshold) }'; then
  jq -n --argjson recall "${recall}" --argjson hit "${hit}" \
    --argjson min_recall "${MIN_RECALL}" --argjson min_hit "${MIN_HIT}" \
    '{stage:"dev_retrieval",passed:false,mean_recall:$recall,hit_rate:$hit,min_recall:$min_recall,min_hit:$min_hit}' \
    > "${GATE_ROOT}/release_decision.json"
  exit 0
fi

export REPO_ROOT SFT_ADAPTER
export OUTPUT_ROOT="${GATE_ROOT}/terminal_eval"
export EVAL_ONLY=1 MAX_GROUPS="${TERMINAL_EVAL_GROUPS:-10}" K="${TERMINAL_EVAL_K:-2}"
export PLANNER_MODEL="${PLANNER_MODEL:-openai/gpt-oss-120b:free}"
export SKILL_MODEL="${SKILL_MODEL:-qwen/qwen3.5-9b}"
export ALLOW_SDPA_FALLBACK=1
"${REPO_ROOT}/scripts/grpo/run_l2_terminal_on_policy.sh"

terminal_rate="$(jq -r '.terminal_success_rate' "${OUTPUT_ROOT}/terminal_grpo_report.json")"
if ! awk -v value="${terminal_rate}" -v threshold="${MIN_TERMINAL}" 'BEGIN { exit !(value >= threshold) }'; then
  jq -n --argjson recall "${recall}" --argjson hit "${hit}" \
    --argjson terminal_rate "${terminal_rate}" --argjson min_terminal "${MIN_TERMINAL}" \
    '{stage:"terminal_executor",passed:false,mean_recall:$recall,hit_rate:$hit,terminal_success_rate:$terminal_rate,min_terminal:$min_terminal}' \
    > "${GATE_ROOT}/release_decision.json"
  exit 0
fi

jq -n --argjson recall "${recall}" --argjson hit "${hit}" \
  --argjson terminal_rate "${terminal_rate}" \
  '{stage:"grpo_release",passed:true,mean_recall:$recall,hit_rate:$hit,terminal_success_rate:$terminal_rate}' \
  > "${GATE_ROOT}/release_decision.json"

unset EVAL_ONLY
export OUTPUT_ROOT="${GATE_ROOT}/grpo"
export MAX_GROUPS="${GRPO_MAX_GROUPS:-10}" K="${GRPO_K:-4}" PPO_EPOCHS="${PPO_EPOCHS:-1}"
export WALLTIME="${GRPO_WALLTIME:-08:00:00}"
"${REPO_ROOT}/scripts/grpo/submit_l2_terminal_on_policy.sh" | tee "${GATE_ROOT}/grpo_submit.txt"
