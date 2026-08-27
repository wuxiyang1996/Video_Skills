#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
LOG_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710/slurm_logs"
WALLTIME="${WALLTIME:-12:00:00}"
mkdir -p "${LOG_ROOT}"

submit_worker() {
  local dataset="$1" start="$2" limit="$3" partition="$4" account="$5" qos="$6" gres="$7" nodelist="${8:-}"
  local node_args=()
  if [[ -n "${nodelist}" ]]; then node_args+=(--nodelist="${nodelist}"); fi
  sbatch --parsable \
    --job-name="vs-${dataset}-${start}" \
    --partition="${partition}" --account="${account}" --qos="${qos}" \
    --gres="${gres}" --cpus-per-task=4 --mem=32G --time="${WALLTIME}" \
    --output="${LOG_ROOT}/${dataset}-${start}-%j.out" \
    --error="${LOG_ROOT}/${dataset}-${start}-%j.err" \
    --export="ALL,DATASET=${dataset},START_INDEX=${start},LIMIT=${limit},SMOKE=0,PILOT_TAG=pilot_20260710_free,CLIP_WORKERS=1,GRAPH_WORKERS=2,GRAPH_MODEL=openai/gpt-oss-120b:free" \
    "${node_args[@]}" \
    "${REPO_ROOT}/scripts/sft_pilot/run_local_qwen_worker.sh"
}

submit_worker cg_bench 0 25 scavenger scavenger scavenger gpu:a100:1 cml32
submit_worker video_holmes 0 25 scavenger scavenger scavenger gpu:a100:1 cml32
submit_worker cg_bench 25 25 gamma gamma default gpu:l40s:1
submit_worker video_holmes 25 25 gamma gamma default gpu:l40s:1
