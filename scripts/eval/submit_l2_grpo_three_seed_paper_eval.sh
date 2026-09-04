#!/usr/bin/env bash
# Submit unattended per-seed dev evaluation, gates, and final 3-seed aggregate.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
SEED_JOBS="${SEED_JOBS:?set SEED_JOBS as SEED:JOB|SEED:JOB|SEED:JOB}"
PARTITION="${PARTITION:-scavenger}"
ACCOUNT="${ACCOUNT:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:rtxa5000:1}"
TRAIN_RUN_PREFIX="${TRAIN_RUN_PREFIX:-grpo_main_v8alpha075_relv2_balanced200_k8}"
EVAL_RUN_TAG="${EVAL_RUN_TAG:-grpo_main_v8alpha075}"
TERMINAL_EVAL_TAG="${TERMINAL_EVAL_TAG:-terminal_dev_grpo_main_v8alpha075}"
AGGREGATE_OUTPUT="${AGGREGATE_OUTPUT:-${PAPER_ROOT}/${EVAL_RUN_TAG}_three_seed_aggregate.json}"

for leaf in "${TRAIN_RUN_PREFIX}" "${EVAL_RUN_TAG}" "${TERMINAL_EVAL_TAG}"; do
  [[ "${leaf}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "run prefix/tag must be a single safe path component: ${leaf}" >&2
    exit 2
  }
done

CG_DEV_JSONL="${CG_DEV_JSONL:-${PAPER_ROOT}/dev_eval_cg_bench_core14_frozen_prompt_v2/dev_pointwise.jsonl}"
VH_DEV_JSONL="${VH_DEV_JSONL:-${PAPER_ROOT}/dev_eval_video_holmes_full_v2_placeholder_clean/dev_pointwise.jsonl}"
DEV_ALLOWLIST="${DEV_ALLOWLIST:-${PAPER_ROOT}/dev_core_cg14_vh21_v1.txt}"
SFT_CG_REPORT="${SFT_CG_REPORT:-${PAPER_ROOT}/dev_eval_results_clean_v7_frozen_prompt/sft_cg14/report.json}"
SFT_VH_REPORT="${SFT_VH_REPORT:-${PAPER_ROOT}/pointwise_relv2_sft_vh21_top4/eval_report.json}"
TERMINAL_BASELINE_ROOT="${TERMINAL_BASELINE_ROOT:-${PAPER_ROOT}/terminal_dev_opd_v8alpha075_repairfinal_v1_core10x8_pt09_relv2_seed42}"
TERMINAL_BASELINE_REPORT="${TERMINAL_BASELINE_REPORT:-${TERMINAL_BASELINE_ROOT}/terminal_grpo_report.json}"
TERMINAL_BASELINE_GATES="${TERMINAL_BASELINE_GATES:-${TERMINAL_BASELINE_ROOT}/gate_cg_bench.json|${TERMINAL_BASELINE_ROOT}/gate_video_holmes.json}"
EXECUTOR_CACHE_DIR="${EXECUTOR_CACHE_DIR:-${PAPER_ROOT}/executor_cache_terminal_core10x8_dataset_routed_typedplan_v2}"

for required in "${CG_DEV_JSONL}" "${VH_DEV_JSONL}" "${DEV_ALLOWLIST}" \
  "${SFT_CG_REPORT}" "${SFT_VH_REPORT}" "${TERMINAL_BASELINE_REPORT}"; do
  [[ -f "${required}" ]] || { echo "missing frozen evaluation input: ${required}" >&2; exit 2; }
done
IFS='|' read -r -a baseline_gates <<< "${TERMINAL_BASELINE_GATES}"
for gate in "${baseline_gates[@]}"; do
  [[ -f "${gate}" ]] || { echo "missing baseline gate: ${gate}" >&2; exit 2; }
  [[ "$(jq -r '.passed // false' "${gate}")" == "true" ]] || {
    echo "baseline gate did not pass: ${gate}" >&2
    exit 3
  }
done

IFS='|' read -r -a seed_jobs <<< "${SEED_JOBS}"
[[ "${#seed_jobs[@]}" -eq 3 ]] || { echo "exactly three SEED_JOBS are required" >&2; exit 4; }
declare -A seen_seeds=()
gate_jobs=()
aggregate_specs=()
mkdir -p "${PAPER_ROOT}/dependency_logs"

extract_job_id() {
  local output="$1"
  local job_id="${output##* -> }"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "could not parse job id from: ${output}" >&2; return 1; }
  printf '%s' "${job_id}"
}

for seed_job in "${seed_jobs[@]}"; do
  seed="${seed_job%%:*}"
  train_job="${seed_job#*:}"
  [[ "${seed}" =~ ^[0-9]+$ && "${train_job}" =~ ^[0-9]+$ ]] || {
    echo "invalid seed job spec: ${seed_job}" >&2
    exit 4
  }
  [[ -z "${seen_seeds[${seed}]:-}" ]] || { echo "duplicate seed: ${seed}" >&2; exit 4; }
  seen_seeds["${seed}"]=1

  train_root="${PAPER_ROOT}/${TRAIN_RUN_PREFIX}_seed${seed}"
  adapter="${train_root}/adapter"
  eval_root="${PAPER_ROOT}/${EVAL_RUN_TAG}_seed${seed}_dev"
  cg_root="${eval_root}/cg14"
  vh_root="${eval_root}/vh21"
  terminal_root="${PAPER_ROOT}/${TERMINAL_EVAL_TAG}_seed${seed}_core10x8_pt09_relv2_evalseed42"
  gate_root="${eval_root}/gates"
  training_gate_root="${eval_root}/training_pool_gate"
  mkdir -p "${cg_root}/slurm_logs" "${vh_root}/slurm_logs" \
    "${gate_root}/slurm_logs" "${training_gate_root}/slurm_logs"

  training_gate_job="$(sbatch --parsable --job-name="grpo-s${seed}-train-gate" \
    --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
    --cpus-per-task=2 --mem=8G --time=00:10:00 --dependency="afterok:${train_job}" \
    --output="${training_gate_root}/slurm_logs/gate-%j.out" \
    --error="${training_gate_root}/slurm_logs/gate-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},TRAINING_REPORT=${train_root}/terminal_grpo_report.json,OUTPUT=${training_gate_root}/gate.json" \
    "${REPO_ROOT}/scripts/eval/run_l2_grpo_training_pool_gate.sh")"

  cg_submit="$(env REPO_ROOT="${REPO_ROOT}" ADAPTER="${adapter}" DEV_JSONL="${CG_DEV_JSONL}" \
    OUTPUT="${cg_root}/eval_report.json" LOG_ROOT="${cg_root}/slurm_logs" TOP_K=2 BATCH_SIZE=8 \
    BOUNDARY_ANCHOR_INDEX0=1 CONDA_ENV="${CONDA_ENV}" JOB_NAME="grpo-s${seed}-cg-dev" \
    PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" WALLTIME=00:30:00 \
    DEPENDENCY="afterok:${training_gate_job}" "${REPO_ROOT}/scripts/eval/submit_l2_dataset_pointwise_eval.sh")"
  cg_job="$(extract_job_id "${cg_submit}")"

  vh_submit="$(env REPO_ROOT="${REPO_ROOT}" ADAPTER="${adapter}" DEV_JSONL="${VH_DEV_JSONL}" \
    OUTPUT="${vh_root}/eval_report.json" LOG_ROOT="${vh_root}/slurm_logs" TOP_K=4 BATCH_SIZE=8 \
    BOUNDARY_ANCHOR_INDEX0=0 CONDA_ENV="${CONDA_ENV}" JOB_NAME="grpo-s${seed}-vh-dev" \
    PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" WALLTIME=00:30:00 \
    DEPENDENCY="afterok:${training_gate_job}" "${REPO_ROOT}/scripts/eval/submit_l2_dataset_pointwise_eval.sh")"
  vh_job="$(extract_job_id "${vh_submit}")"

  terminal_submit="$(env REPO_ROOT="${REPO_ROOT}" OUTPUT_ROOT="${terminal_root}" \
    DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}" SPLIT_ROLE=dev_tune \
    MAX_GROUPS=10 REPEATS_PER_EXAMPLE=1 REPEAT_START_INDEX=0 K=8 CG_TOPK=2 VIDEO_HOLMES_TOPK=4 \
    TEMPERATURE=0.9 POINTWISE_TEMPERATURE=0.9 POINTWISE_TRAIN_BATCH_SIZE=1 MAX_NEW_TOKENS=384 \
    GENERATION_TIMEOUT_S=90 ROLLOUT_TIMEOUT_S=240 PLANNER_MODEL=openai/gpt-oss-120b \
    SKILL_MODEL=openai/gpt-oss-120b PLANNER_TIMEOUT_S=180 SKILL_TIMEOUT_S=90 \
    EXECUTOR_CACHE_DIR="${EXECUTOR_CACHE_DIR}" SEED=42 CONDA_ENV="${CONDA_ENV}" \
    KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}" SFT_ADAPTER="${adapter}" \
    BOUNDARY_ANCHOR_INDEX0=0 ALLOW_SDPA_FALLBACK=0 EVAL_ONLY=1 RETRIEVAL_ONLY=0 \
    PROCESS_REWARD_WARMUP=0 TERMINAL_ON_PROCESS_HIT=1 POINTWISE_ACTION_POLICY=1 \
    POINTWISE_ACTION_DATASETS=video_holmes EXAMPLE_ID_ALLOWLIST="${DEV_ALLOWLIST}" \
    PRESERVE_ALLOWLIST_ORDER=0 DATASET_BALANCED_SAMPLING=1 REQUIRE_PROCESS_SUPERVISION=1 \
    MIN_CATALOG_SIZE=1 RELEASE_GATE_FILES="${TERMINAL_BASELINE_GATES}" \
    PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" CPUS=8 MEM=64G \
    WALLTIME=02:00:00 DEPENDENCY="afterok:${training_gate_job}" \
    "${REPO_ROOT}/scripts/grpo/submit_l2_terminal_on_policy.sh")"
  terminal_job="$(extract_job_id "${terminal_submit}")"

  gate_job="$(sbatch --parsable --job-name="grpo-s${seed}-dev-gates" \
    --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
    --cpus-per-task=2 --mem=8G --time=00:10:00 \
    --dependency="afterok:${cg_job}:${vh_job}:${terminal_job}" \
    --output="${gate_root}/slurm_logs/gate-%j.out" --error="${gate_root}/slurm_logs/gate-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},ADAPTER=${adapter},CG_REPORT=${cg_root}/eval_report.json,VH_REPORT=${vh_root}/eval_report.json,TERMINAL_REPORT=${terminal_root}/terminal_grpo_report.json,OUTPUT_ROOT=${gate_root},SFT_CG_REPORT=${SFT_CG_REPORT},SFT_VH_REPORT=${SFT_VH_REPORT},TERMINAL_BASELINE_REPORT=${TERMINAL_BASELINE_REPORT}" \
    "${REPO_ROOT}/scripts/eval/run_l2_grpo_seed_gates.sh")"
  gate_jobs+=("${gate_job}")
  aggregate_specs+=("${seed}|${train_root}/terminal_grpo_report.json|${cg_root}/eval_report.json|${vh_root}/eval_report.json|${terminal_root}/terminal_grpo_report.json|${gate_root}/pointwise_preservation_gate.json|${gate_root}/gate_cg_bench.json|${gate_root}/gate_video_holmes.json")
  echo "seed ${seed}: train=${train_job} train_gate=${training_gate_job} cg=${cg_job} vh=${vh_job} terminal=${terminal_job} gates=${gate_job}"
done

gate_dependency="afterany:$(IFS=:; echo "${gate_jobs[*]}")"
aggregate_seed_specs="$(IFS=^; echo "${aggregate_specs[*]}")"
aggregate_output="${AGGREGATE_OUTPUT}"
aggregate_job="$(sbatch --parsable --job-name=grpo-3seed-aggregate-final \
  --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
  --cpus-per-task=2 --mem=8G --time=00:10:00 --dependency="${gate_dependency}" \
  --output="${PAPER_ROOT}/dependency_logs/aggregate-final-%j.out" \
  --error="${PAPER_ROOT}/dependency_logs/aggregate-final-%j.err" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},SEED_SPECS=${aggregate_seed_specs},OUTPUT=${aggregate_output}" \
  "${REPO_ROOT}/scripts/eval/run_l2_grpo_three_seed_aggregate.sh")"
echo "aggregate: ${aggregate_job} -> ${aggregate_output}"
