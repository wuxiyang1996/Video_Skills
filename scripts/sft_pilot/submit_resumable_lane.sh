#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PILOT_TAG="${PILOT_TAG:-pilot_20260710_free}"
LOG_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${PILOT_TAG}/slurm_logs"
mkdir -p "${LOG_ROOT}"

DATASET="${DATASET:?set DATASET, e.g. cg_bench or video_holmes}"
STARTS="${STARTS:?set STARTS, e.g. '0 50'}"
LIMIT="${LIMIT:-25}"
REPEATS="${REPEATS:-8}"
ATTEMPT_START="${ATTEMPT_START:-1}"
WALLTIME="${WALLTIME:-12:00:00}"
PARTITION="${PARTITION:-gamma}"
ACCOUNT="${ACCOUNT:-gamma}"
QOS="${QOS:-default}"
GRES="${GRES:-gpu:l40s:1}"
NODELIST="${NODELIST:-}"
INITIAL_DEPENDENCY="${INITIAL_DEPENDENCY:-}"
CLIP_WORKERS="${CLIP_WORKERS:-1}"
GRAPH_WORKERS="${GRAPH_WORKERS:-2}"
GRAPH_MODEL="${GRAPH_MODEL:-openai/gpt-oss-120b:free}"

submit_one() {
  local start="$1" attempt="$2" dependency="$3"
  local dep_args=()
  local node_args=()
  if [[ -n "${dependency}" ]]; then dep_args+=(--dependency="afterany:${dependency}"); fi
  if [[ -n "${NODELIST}" ]]; then node_args+=(--nodelist="${NODELIST}"); fi
  sbatch --parsable \
    --job-name="vs-resume-${DATASET}-${start}-r${attempt}" \
    --partition="${PARTITION}" --account="${ACCOUNT}" --qos="${QOS}" \
    --gres="${GRES}" --cpus-per-task=4 --mem=32G --time="${WALLTIME}" \
    "${dep_args[@]}" \
    --output="${LOG_ROOT}/${DATASET}-${start}-r${attempt}-%j.out" \
    --error="${LOG_ROOT}/${DATASET}-${start}-r${attempt}-%j.err" \
    --export="ALL,DATASET=${DATASET},START_INDEX=${start},LIMIT=${LIMIT},SMOKE=0,PILOT_TAG=${PILOT_TAG},CLIP_WORKERS=${CLIP_WORKERS},GRAPH_WORKERS=${GRAPH_WORKERS},GRAPH_MODEL=${GRAPH_MODEL}" \
    "${node_args[@]}" \
    "${REPO_ROOT}/scripts/sft_pilot/run_local_qwen_worker.sh"
}

previous="${INITIAL_DEPENDENCY}"
for start in ${STARTS}; do
  attempt_end=$((ATTEMPT_START + REPEATS - 1))
  for attempt in $(seq "${ATTEMPT_START}" "${attempt_end}"); do
    job_id="$(submit_one "${start}" "${attempt}" "${previous}")"
    printf '%s dataset=%s start=%s attempt=%s dependency=%s walltime=%s graph_model=%s\n' \
      "${job_id}" "${DATASET}" "${start}" "${attempt}" "${previous:-none}" "${WALLTIME}" "${GRAPH_MODEL}"
    previous="${job_id}"
  done
done
