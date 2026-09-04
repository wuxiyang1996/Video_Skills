#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PYTHON="${PYTHON:-/fs/gamma-projects/vlm-robot/conda/envs/swift/bin/python}"

TRAIN_SOURCE="${TRAIN_SOURCE:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5_expanded_20260831/five_lora/l2/train.jsonl}"
DEV_SOURCE="${DEV_SOURCE:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v5_expanded_20260831/five_lora/l2/dev.jsonl}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_expanded_20260831/candidate_retrieval/train_fine8s_report.json}"
DEV_CANDIDATES="${DEV_CANDIDATES:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_oracle_v5_20260830_011126/l2/candidate_retrieval_eval/qwen3_vl_embedding_2b_dev_fine8s_report.json}"
VISUAL_TEACHER_TRAIN="${VISUAL_TEACHER_TRAIN:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_visual_teacher_8b_fine8s_top32_train_v10_20260831/report.json}"
VISUAL_TEACHER_DEV="${VISUAL_TEACHER_DEV:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v8_retry_20260831/l2/visual_reranker_8b_fine8s_top32/report.json}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v11_teacher8b_20260831/five_lora/l2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v11_teacher8b_20260831/l2}"
GATE_ROOT="${GATE_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_terminal_gate_v11_teacher8b_20260831}"
INIT_ADAPTER="${INIT_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v10_pool_20260831/l2/pilot/adapter}"
TEACHER_HARD_NEGATIVES="${TEACHER_HARD_NEGATIVES:-8}"

for path in \
  "${TRAIN_SOURCE}" "${DEV_SOURCE}" "${TRAIN_CANDIDATES}" "${DEV_CANDIDATES}" \
  "${VISUAL_TEACHER_TRAIN}" "${VISUAL_TEACHER_DEV}" "${INIT_ADAPTER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}" "${GATE_ROOT}"

"${PYTHON}" -m dataset_clip_wrapper.training.l2_pointwise_reranker_v8 \
  --train-jsonl "${TRAIN_SOURCE}" \
  --dev-jsonl "${DEV_SOURCE}" \
  --output-dir "${DATA_ROOT}" \
  --label-independent-train-candidate-report "${TRAIN_CANDIDATES}" \
  --label-independent-dev-candidate-report "${DEV_CANDIDATES}" \
  --visual-teacher-train-report "${VISUAL_TEACHER_TRAIN}" \
  --visual-teacher-dev-report "${VISUAL_TEACHER_DEV}" \
  --teacher-hard-negatives "${TEACHER_HARD_NEGATIVES}"

echo "Built v11 data: ${DATA_ROOT}"
jq '{train:.train, dev_label_independent:.dev_label_independent}' "${DATA_ROOT}/report.json"

sft_job="$(
  sbatch --parsable \
    --job-name="vs-l2-v11-teach" \
    --partition=gamma \
    --account=gamma \
    --gres=gpu:l40s:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=12:00:00 \
    --chdir="${REPO_ROOT}" \
    --output="${OUTPUT_ROOT}/slurm-%j.out" \
    --error="${OUTPUT_ROOT}/slurm-%j.err" \
    --export="ALL,STAGE=pilot,SPECIALIST=l2,DATA_ROOT=${DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},REPO_ROOT=${REPO_ROOT},INIT_ADAPTER=${INIT_ADAPTER},EPOCHS=${EPOCHS:-3},GRAD_ACCUM=${GRAD_ACCUM:-8},LEARNING_RATE=${LEARNING_RATE:-1e-5},GEN_EXAMPLES=${GEN_EXAMPLES:-8}" \
    "${REPO_ROOT}/scripts/sft_pilot/run_lora_sft.sh"
)"

eval_job="$(
  sbatch --parsable \
    --job-name="vs-l2-v11-eval" \
    --partition=gamma \
    --account=gamma \
    --gres=gpu:l40s:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --dependency="afterok:${sft_job}" \
    --chdir="${REPO_ROOT}" \
    --output="${OUTPUT_ROOT}/eval-%j.out" \
    --error="${OUTPUT_ROOT}/eval-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},ADAPTER=${OUTPUT_ROOT}/pilot/adapter,DEV_JSONL=${DATA_ROOT}/dev_label_independent.jsonl,OUTPUT=${OUTPUT_ROOT}/dev_label_independent/report.json" \
    "${REPO_ROOT}/scripts/sft_pilot/run_l2_expanded_pointwise_eval.sh"
)"

release_job="$(
  sbatch --parsable \
    --job-name="vs-l2-v11-release" \
    --partition=gamma \
    --account=gamma \
    --gres=gpu:l40s:1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=09:00:00 \
    --dependency="afterok:${eval_job}" \
    --chdir="${REPO_ROOT}" \
    --output="${GATE_ROOT}/slurm-%j.out" \
    --error="${GATE_ROOT}/slurm-%j.err" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},DEV_REPORT=${OUTPUT_ROOT}/dev_label_independent/report.json,SFT_ADAPTER=${OUTPUT_ROOT}/pilot/adapter,GATE_ROOT=${GATE_ROOT}" \
    "${REPO_ROOT}/scripts/grpo/run_l2_release_gate.sh"
)"

echo "submitted v11 chain"
echo "sft_job=${sft_job}"
echo "eval_job=${eval_job}"
echo "release_job=${release_job}"
