#!/usr/bin/env bash
# Launch the fail-closed consensus GRPO -> dev -> pretest -> heldout paper chain.
# Submit this wrapper with an afterok dependency on terminal consensus selection.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-qwen35-serve/bin/python}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-high}"
GRES="${GRES:-gpu:l40s:1}"
RUN_PREFIX="${RUN_PREFIX:-grpo_consensus_repairfinal_v1_v8alpha075_balanced100_k8}"
EVAL_TAG="${EVAL_TAG:-grpo_consensus_repairfinal_v1_v8alpha075}"
TERMINAL_EVAL_TAG="${TERMINAL_EVAL_TAG:-terminal_dev_grpo_consensus_repairfinal_v1_v8alpha075}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:-${PAPER_ROOT}/terminal_consensus_typedplan_v2}"
ALLOWLIST="${ALLOWLIST:-${CONSENSUS_ROOT}/exact_balanced100.tsv}"
SELECTION_REPORT="${SELECTION_REPORT:-${CONSENSUS_ROOT}/selection_report.json}"
AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT:-${PAPER_ROOT}/${EVAL_TAG}_three_seed_aggregate.json}"
PRETEST_OUTPUT="${PRETEST_OUTPUT:-${PAPER_ROOT}/${EVAL_TAG}_paper_pretest_release_gate.json}"
PRETEST_REWARD_SEPARATION="${PRETEST_REWARD_SEPARATION:-${PAPER_ROOT}/gates_v2_frozen_prompt/terminal_reward_separation_repairfinal_v1.json}"
PRETEST_OPD_TERMINAL_SELECTION="${PRETEST_OPD_TERMINAL_SELECTION:-${PAPER_ROOT}/opd_v8_dev/terminal_qualified_checkpoint_selection_repairfinal_v1.json}"
HELDOUT_ROOT="${HELDOUT_ROOT:-${PAPER_ROOT}/heldout_pointwise_consensus_repairfinal_v1}"
LOG_ROOT="${LOG_ROOT:-${PAPER_ROOT}/dependency_logs/consensus_chain}"
TERMINAL_BASELINE_ROOT="${TERMINAL_BASELINE_ROOT:-${PAPER_ROOT}/terminal_dev_opd_v8alpha075_repairfinal_v1_core10x8_pt09_relv2_seed42}"
TERMINAL_BASELINE_REPORT="${TERMINAL_BASELINE_REPORT:-${TERMINAL_BASELINE_ROOT}/terminal_grpo_report.json}"
TERMINAL_BASELINE_GATES="${TERMINAL_BASELINE_GATES:-${TERMINAL_BASELINE_ROOT}/gate_cg_bench.json|${TERMINAL_BASELINE_ROOT}/gate_video_holmes.json}"
TRAIN_EXECUTOR_CACHE_DIR="${TRAIN_EXECUTOR_CACHE_DIR:-${PAPER_ROOT}/executor_cache_grpo_main_v8_relv2_typedplan_v2}"
DEV_EXECUTOR_CACHE_DIR="${DEV_EXECUTOR_CACHE_DIR:-${PAPER_ROOT}/executor_cache_terminal_core10x8_dataset_routed_typedplan_v2}"
RELEASE_GATE_FILES="${RELEASE_GATE_FILES:-${TERMINAL_BASELINE_GATES}|${PAPER_ROOT}/grpo_pilot_v8alpha075_dev/pointwise_preservation_gate.json}"
# Recovery path for a launcher that already submitted the three training jobs.
# This lets us rebuild only the downstream evaluation chain without duplicating
# GPU training, while still proving that the running jobs use the canonical
# consensus rows byte-for-byte.
EXISTING_SEED_JOBS="${EXISTING_SEED_JOBS:-}"
EXISTING_TRAIN_ALLOWLIST="${EXISTING_TRAIN_ALLOWLIST:-}"

mkdir -p "${LOG_ROOT}" "${HELDOUT_ROOT}"
cd "${REPO_ROOT}"

[[ -f "${ALLOWLIST}" ]] || { echo "missing consensus allowlist: ${ALLOWLIST}" >&2; exit 2; }
[[ -f "${SELECTION_REPORT}" ]] || { echo "missing consensus report: ${SELECTION_REPORT}" >&2; exit 2; }
[[ "$(jq -r '.passed // false' "${SELECTION_REPORT}")" == "true" ]] || {
  echo "consensus selection gate did not pass: ${SELECTION_REPORT}" >&2
  exit 3
}
[[ "$(jq -r '.checks.source_provenance_train_only // false' "${SELECTION_REPORT}")" == "true" ]] || {
  echo "consensus selection is missing passed train-only source provenance: ${SELECTION_REPORT}" >&2
  exit 3
}
IFS='|' read -r -a terminal_baseline_gates <<< "${TERMINAL_BASELINE_GATES}"
[[ -f "${TERMINAL_BASELINE_REPORT}" ]] || {
  echo "missing typed-plan terminal baseline: ${TERMINAL_BASELINE_REPORT}" >&2
  exit 2
}
for gate in "${terminal_baseline_gates[@]}"; do
  [[ -f "${gate}" && "$(jq -r '.passed // false' "${gate}")" == "true" ]] || {
    echo "typed-plan terminal baseline gate missing or failed: ${gate}" >&2
    exit 3
  }
done
[[ "$(wc -l < "${ALLOWLIST}")" -eq 100 ]] || {
  echo "consensus allowlist must contain exactly 100 rows" >&2
  exit 3
}
reported_allowlist_sha="$(jq -r '.allowlist_artifact.sha256 // empty' "${SELECTION_REPORT}")"
actual_allowlist_sha="$(sha256sum "${ALLOWLIST}" | awk '{print $1}')"
[[ -n "${reported_allowlist_sha}" && "${reported_allowlist_sha}" == "${actual_allowlist_sha}" ]] || {
  echo "consensus allowlist hash does not match selection report" >&2
  exit 3
}
if [[ -n "${EXISTING_SEED_JOBS}" ]]; then
  [[ -f "${EXISTING_TRAIN_ALLOWLIST}" ]] || {
    echo "EXISTING_TRAIN_ALLOWLIST is required when reusing training jobs" >&2
    exit 4
  }
  cmp -s "${ALLOWLIST}" "${EXISTING_TRAIN_ALLOWLIST}" || {
    echo "existing training allowlist differs from canonical consensus allowlist" >&2
    exit 4
  }
  seed_jobs="${EXISTING_SEED_JOBS}"
  for seed_job in ${seed_jobs//|/ }; do
    seed="${seed_job%%:*}"
    job="${seed_job#*:}"
    job_record="$(scontrol show job -dd -o "${job}" 2>/dev/null || true)"
    [[ -n "${job_record}" && "${job_record}" == *"OUTPUT_ROOT=${PAPER_ROOT}/${RUN_PREFIX}_seed${seed}"* ]] || {
      echo "existing seed job does not match expected training root: ${seed_job}" >&2
      exit 4
    }
    [[ "${job_record}" == *"EXECUTOR_CACHE_DIR=${TRAIN_EXECUTOR_CACHE_DIR}"* ]] || {
      echo "existing seed job does not use canonical typed-plan executor cache: ${seed_job}" >&2
      exit 4
    }
  done
  echo "reusing training jobs: ${seed_jobs}"
else
  for seed in 42 43 44; do
    [[ ! -e "${PAPER_ROOT}/${RUN_PREFIX}_seed${seed}/terminal_grpo_report.json" ]] || {
      echo "refusing to overwrite completed consensus run for seed ${seed}" >&2
      exit 4
    }
  done

  training_output="$(env REPO_ROOT="${REPO_ROOT}" PAPER_ROOT="${PAPER_ROOT}" \
    CONDA_ENV="${CONDA_ENV}" RUN_PREFIX="${RUN_PREFIX}" EXAMPLE_ID_ALLOWLIST="${ALLOWLIST}" \
    EXECUTOR_CACHE_DIR="${TRAIN_EXECUTOR_CACHE_DIR}" RELEASE_GATE_FILES="${RELEASE_GATE_FILES}" \
    PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" \
    WALLTIME="${TRAIN_WALLTIME:-16:00:00}" \
    "${REPO_ROOT}/scripts/grpo/submit_l2_paper_three_seed_training.sh")"
  printf '%s\n' "${training_output}"
  seed_jobs="$(printf '%s\n' "${training_output}" | sed -n 's/^SEED_JOBS=//p' | tail -n 1)"
fi
[[ "${seed_jobs}" =~ ^42:[0-9]+\|43:[0-9]+\|44:[0-9]+$ ]] || {
  echo "could not parse three training jobs" >&2
  exit 5
}

eval_output="$(env REPO_ROOT="${REPO_ROOT}" PAPER_ROOT="${PAPER_ROOT}" \
  CONDA_ENV="${CONDA_ENV}" SEED_JOBS="${seed_jobs}" TRAIN_RUN_PREFIX="${RUN_PREFIX}" \
  EVAL_RUN_TAG="${EVAL_TAG}" TERMINAL_EVAL_TAG="${TERMINAL_EVAL_TAG}" \
  AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT}" PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" \
  QOS="${QOS}" GRES="${GRES}" TERMINAL_BASELINE_ROOT="${TERMINAL_BASELINE_ROOT}" \
  TERMINAL_BASELINE_REPORT="${TERMINAL_BASELINE_REPORT}" TERMINAL_BASELINE_GATES="${TERMINAL_BASELINE_GATES}" \
  EXECUTOR_CACHE_DIR="${DEV_EXECUTOR_CACHE_DIR}" \
  "${REPO_ROOT}/scripts/eval/submit_l2_grpo_three_seed_paper_eval.sh")"
printf '%s\n' "${eval_output}"
aggregate_job="$(printf '%s\n' "${eval_output}" | sed -n 's/^aggregate: \([0-9][0-9]*\) ->.*$/\1/p' | tail -n 1)"
[[ "${aggregate_job}" =~ ^[0-9]+$ ]] || { echo "could not parse aggregate job" >&2; exit 5; }

pretest_job="$(sbatch --parsable --job-name=l2-consensus-pretest \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --cpus-per-task=2 --mem=8G --time=00:10:00 --dependency="afterok:${aggregate_job}" \
  --output="${LOG_ROOT}/pretest-%j.out" --error="${LOG_ROOT}/pretest-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},PAPER_ROOT=${PAPER_ROOT},PYTHON=${PYTHON},MINING_REPORT=${SELECTION_REPORT},THREE_SEED_AGGREGATE=${AGGREGATE_OUTPUT},REWARD_SEPARATION=${PRETEST_REWARD_SEPARATION},OPD_TERMINAL_SELECTION=${PRETEST_OPD_TERMINAL_SELECTION},OUTPUT=${PRETEST_OUTPUT}" \
  "${REPO_ROOT}/scripts/eval/run_l2_paper_pretest_release.sh")"

build_job="$(sbatch --parsable --job-name=l2-consensus-heldout-build \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --cpus-per-task=4 --mem=32G --time=01:00:00 --dependency="afterok:${pretest_job}" \
  --output="${LOG_ROOT}/heldout-build-%j.out" --error="${LOG_ROOT}/heldout-build-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},PAPER_ROOT=${PAPER_ROOT},PYTHON=${PYTHON},RELEASE_GATE=${PRETEST_OUTPUT},OUTPUT_ROOT=${HELDOUT_ROOT}" \
  "${REPO_ROOT}/scripts/eval/build_l2_paper_heldout_pointwise.sh")"

heldout_output="$(env REPO_ROOT="${REPO_ROOT}" PAPER_ROOT="${PAPER_ROOT}" BUILD_JOB="${build_job}" \
  GRPO_RUN_PREFIX="${RUN_PREFIX}" HELDOUT_ROOT="${HELDOUT_ROOT}" PARTITION="${PARTITION}" \
  ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" \
  "${REPO_ROOT}/scripts/eval/submit_l2_paper_heldout_pointwise.sh")"
printf '%s\n' "${heldout_output}"
mapfile -t heldout_jobs < <(printf '%s\n' "${heldout_output}" | sed -n 's/.* -> \([0-9][0-9]*\)$/\1/p')
[[ "${#heldout_jobs[@]}" -eq 10 ]] || { echo "expected 10 heldout jobs" >&2; exit 5; }
heldout_dependency="afterok:$(IFS=:; echo "${heldout_jobs[*]}")"

heldout_aggregate_job="$(sbatch --parsable --job-name=l2-consensus-heldout-aggregate \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --cpus-per-task=2 --mem=8G --time=00:10:00 --dependency="${heldout_dependency}" \
  --output="${LOG_ROOT}/heldout-aggregate-%j.out" \
  --error="${LOG_ROOT}/heldout-aggregate-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},PAPER_ROOT=${PAPER_ROOT},PYTHON=${PYTHON},HELDOUT_ROOT=${HELDOUT_ROOT},THREE_SEED_AGGREGATE=${AGGREGATE_OUTPUT}" \
  "${REPO_ROOT}/scripts/eval/run_l2_paper_heldout_aggregate.sh")"

echo "consensus chain: training=${seed_jobs} aggregate=${aggregate_job} pretest=${pretest_job} build=${build_job} heldout_aggregate=${heldout_aggregate_job}"
