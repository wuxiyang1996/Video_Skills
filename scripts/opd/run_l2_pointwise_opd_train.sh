#!/usr/bin/env bash
# Train the L2 pointwise adapter on a frozen CG/VH OPD distillation file.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
DISTILL="${DISTILL:?set DISTILL to opd_distill.jsonl}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/l2_pointwise_v11_teacher8b_20260831/l2/pilot/adapter}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
args=(
  --adapter "${SFT_ADAPTER}"
  --distill "${DISTILL}"
  --output-dir "${OUTPUT_ROOT}"
  --epochs "${EPOCHS:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --warmup-ratio "${WARMUP_RATIO:-0.05}"
  --seed "${SEED:-42}"
)
if [[ "${DATASET_BALANCED_LOSS:-0}" == "1" ]]; then
  args+=(--dataset-balanced-loss)
fi
"${PYTHON}" -m trainer.train_l2_pointwise_opd "${args[@]}"
