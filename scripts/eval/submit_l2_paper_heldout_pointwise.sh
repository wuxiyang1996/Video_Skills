#!/usr/bin/env bash
# Submit the frozen SFT/OPD/three-seed GRPO heldout pointwise matrix.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
BUILD_JOB="${BUILD_JOB:-}"
PARTITION="${PARTITION:-scavenger}"
ACCOUNT="${ACCOUNT:-scavenger}"
QOS="${QOS:-scavenger}"
GRES="${GRES:-gpu:rtxa5000:1}"
WALLTIME="${WALLTIME:-1-00:00:00}"
GRPO_RUN_PREFIX="${GRPO_RUN_PREFIX:-grpo_main_v8alpha075_relv2_balanced200_k8}"
HELDOUT_ROOT="${HELDOUT_ROOT:-${PAPER_ROOT}/heldout_pointwise_v1}"
# Reports are written under OUTPUT_ROOT, which defaults to HELDOUT_ROOT.  Point it
# elsewhere to re-score a frozen heldout set without overwriting its reports.
OUTPUT_ROOT="${OUTPUT_ROOT:-${HELDOUT_ROOT}}"

[[ "${GRPO_RUN_PREFIX}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "GRPO_RUN_PREFIX must be a single safe path component: ${GRPO_RUN_PREFIX}" >&2
  exit 2
}

if [[ -n "${BUILD_JOB}" ]]; then
  [[ "${BUILD_JOB}" =~ ^[0-9]+$ ]] || { echo "invalid BUILD_JOB: ${BUILD_JOB}" >&2; exit 2; }
  dependency="afterok:${BUILD_JOB}"
else
  # Re-scoring an already-materialised heldout set needs no build dependency.
  dependency=""
fi

models=(
  "sft|${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v11_teacher8b_20260831/l2/pilot/adapter"
  "opd_alpha075|${PAPER_ROOT}/opd_interp_v8_relv2_grid/alpha075/adapter"
  "grpo_seed42|${PAPER_ROOT}/${GRPO_RUN_PREFIX}_seed42/adapter"
  "grpo_seed43|${PAPER_ROOT}/${GRPO_RUN_PREFIX}_seed43/adapter"
  "grpo_seed44|${PAPER_ROOT}/${GRPO_RUN_PREFIX}_seed44/adapter"
)

for model_spec in "${models[@]}"; do
  model_name="${model_spec%%|*}"
  adapter="${model_spec#*|}"
  for dataset in cg_bench video_holmes; do
    if [[ "${dataset}" == "cg_bench" ]]; then
      top_k=2
      boundary_anchor=1
    else
      top_k=4
      boundary_anchor=0
    fi
    output_root="${OUTPUT_ROOT}/results/${model_name}/${dataset}"
    submit_output="$(env \
      REPO_ROOT="${REPO_ROOT}" \
      ADAPTER="${adapter}" \
      DEV_JSONL="${HELDOUT_ROOT}/${dataset}/pointwise.jsonl" \
      OUTPUT="${output_root}/eval_report.json" \
      LOG_ROOT="${output_root}/slurm_logs" \
      TOP_K="${top_k}" \
      BOUNDARY_ANCHOR_INDEX0="${boundary_anchor}" \
      SCORING_MODE="${SCORING_MODE:-}" \
      BATCH_SIZE="${BATCH_SIZE:-8}" \
      JOB_NAME="heldout-${model_name}-${dataset}" \
      PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" \
      CPUS="${CPUS:-6}" MEM="${MEM:-48G}" WALLTIME="${WALLTIME}" \
      DEPENDENCY="${dependency}" \
      "${REPO_ROOT}/scripts/eval/submit_l2_dataset_pointwise_eval.sh")"
    echo "${model_name} ${dataset}: ${submit_output}"
  done
done
