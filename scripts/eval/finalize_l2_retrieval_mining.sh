#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
MINING_ROOT="${MINING_ROOT:?set MINING_ROOT}"
OUTPUT_REPORT="${OUTPUT_REPORT:?set OUTPUT_REPORT}"
OUTPUT_ALLOWLIST="${OUTPUT_ALLOWLIST:?set OUTPUT_ALLOWLIST}"
MIN_ELIGIBLE_PER_DATASET="${MIN_ELIGIBLE_PER_DATASET:-50}"
MIN_ELIGIBLE_GROUP_RATE="${MIN_ELIGIBLE_GROUP_RATE:-0.25}"
PYTHON="${PYTHON:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python}"

cd "${REPO_ROOT}"
sample_log_args=()
if [[ -s "${MINING_ROOT}/terminal_samples.jsonl" ]]; then
  sample_log_args+=(--sample-log "${MINING_ROOT}/terminal_samples.jsonl")
else
  shopt -s nullglob
  slurm_sample_logs=("${MINING_ROOT}"/slurm_logs/train-*.out)
  shopt -u nullglob
  if (( ${#slurm_sample_logs[@]} == 0 )); then
    echo "no sample-level terminal log found under ${MINING_ROOT}" >&2
    exit 26
  fi
  for sample_log in "${slurm_sample_logs[@]}"; do
    sample_log_args+=(--sample-log "${sample_log}")
  done
fi
"${PYTHON}" scripts/eval/aggregate_l2_retrieval_samples.py \
  "${sample_log_args[@]}" \
  --run-report "${MINING_ROOT}/terminal_grpo_report.json" \
  --exact-group-allowlist \
  --balanced-datasets \
  --max-groups-per-dataset 100 \
  --allowlist "${OUTPUT_ALLOWLIST}" \
  --report "${OUTPUT_REPORT}"

# Fail closed on the exact balanced allowlist contract.  Eligibility alone is
# insufficient: an older submitted copy of this script once emitted 91/100
# groups even though both datasets passed the mining-rate gate.
if ! jq -e --argjson minimum "${MIN_ELIGIBLE_PER_DATASET}" '
  .allowlist_selection as $selection
  | ($selection.groups_by_dataset.cg_bench // 0) as $cg
  | ($selection.groups_by_dataset.video_holmes // 0) as $vh
  | $selection.balanced_datasets == true
    and $selection.ordering_contract == "dataset-round-robin-v1"
    and $cg == $vh
    and $cg >= $minimum
    and $selection.balanced_target_per_dataset == $cg
    and $selection.groups == ($cg + $vh)
' "${OUTPUT_REPORT}" >/dev/null; then
  echo "mining allowlist does not satisfy the balanced dataset contract" >&2
  jq '.allowlist_selection' "${OUTPUT_REPORT}" >&2
  exit 28
fi

selected_groups="$(jq -r '.allowlist_selection.groups // 0' "${OUTPUT_REPORT}")"
allowlist_rows="$(wc -l < "${OUTPUT_ALLOWLIST}")"
if (( allowlist_rows != selected_groups )); then
  echo "allowlist has ${allowlist_rows} rows; report declares ${selected_groups}" >&2
  exit 29
fi

for dataset in cg_bench video_holmes; do
  eligible="$(jq -r --arg dataset "${dataset}" '.dataset_metrics[$dataset].groups_eligible // 0' "${OUTPUT_REPORT}")"
  if (( eligible < MIN_ELIGIBLE_PER_DATASET )); then
    echo "${dataset} has ${eligible} eligible groups; need ${MIN_ELIGIBLE_PER_DATASET}" >&2
    exit 24
  fi
  rate="$(jq -r --arg dataset "${dataset}" '.dataset_metrics[$dataset].eligible_group_rate // 0' "${OUTPUT_REPORT}")"
  "${PYTHON}" -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' \
    "${rate}" "${MIN_ELIGIBLE_GROUP_RATE}" || {
      echo "${dataset} eligible group rate ${rate}; need ${MIN_ELIGIBLE_GROUP_RATE}" >&2
      exit 27
    }
done

jq '{groups_seen,groups_eligible,dataset_metrics,controller_action_contract,sampling_protocol,relationship_support_contract}' "${OUTPUT_REPORT}"
