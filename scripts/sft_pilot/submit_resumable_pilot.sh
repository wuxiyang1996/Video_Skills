#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PILOT_TAG="${PILOT_TAG:-pilot_20260710_free}"
LIMIT="${LIMIT:-25}"
REPEATS="${REPEATS:-8}"
WALLTIME="${WALLTIME:-12:00:00}"
GRAPH_MODEL="${GRAPH_MODEL:-openai/gpt-oss-120b:free}"

submit_lane() {
  local dataset="$1" starts="$2" partition="$3" account="$4" qos="$5" gres="$6" nodelist="$7" initial_dependency="$8"
  DATASET="${dataset}" STARTS="${starts}" LIMIT="${LIMIT}" REPEATS="${REPEATS}" WALLTIME="${WALLTIME}" \
    PARTITION="${partition}" ACCOUNT="${account}" QOS="${qos}" GRES="${gres}" NODELIST="${nodelist}" \
    INITIAL_DEPENDENCY="${initial_dependency}" PILOT_TAG="${PILOT_TAG}" GRAPH_MODEL="${GRAPH_MODEL}" \
    REPO_ROOT="${REPO_ROOT}" \
    "${REPO_ROOT}/scripts/sft_pilot/submit_resumable_lane.sh"
}

# Four lanes keep parallelism bounded while allowing each shard to resume across
# the cluster's 12h walltime. Set CURRENT_*_JOB to chain after already-running
# jobs; leave empty for a fresh launch.
submit_lane cg_bench "0 50" scavenger scavenger scavenger gpu:a100:1 "${CG_A100_NODELIST:-cml32}" "${CURRENT_CG0_JOB:-}"
submit_lane cg_bench "25 75" gamma gamma default gpu:l40s:1 "" "${CURRENT_CG25_JOB:-}"
submit_lane video_holmes "0 50" gamma gamma default gpu:l40s:1 "" "${CURRENT_VH0_JOB:-}"
submit_lane video_holmes "25 75" gamma gamma default gpu:l40s:1 "" "${CURRENT_VH25_JOB:-}"
