#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
STAGE="${STAGE:?set STAGE=smoke or pilot}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/sft_recovery_20260720_full/sft_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/qwen35_9b_lora_20260721}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

common=(
  --model Qwen/Qwen3.5-9B
  --train-jsonl "${DATA_ROOT}/train_sft.jsonl"
  --dev-jsonl "${DATA_ROOT}/dev_sft.jsonl"
  --stage "${STAGE}"
  --max-length 16384
  --lora-rank 16
  --lora-alpha 32
  --lora-dropout 0.05
  --seed 42
)

cd "${REPO_ROOT}"
if [[ "${STAGE}" == "smoke" ]]; then
  exec "${VENV_ROOT}/bin/python" -m dataset_clip_wrapper.training.train_lora_sft \
    "${common[@]}" \
    --output-dir "${OUTPUT_ROOT}/smoke" \
    --max-train-samples 20 \
    --max-steps 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-4 \
    --save-steps 4 \
    --generation-examples 4 \
    --min-json-rate 0.5
elif [[ "${STAGE}" == "pilot" ]]; then
  exec "${VENV_ROOT}/bin/python" -m dataset_clip_wrapper.training.train_lora_sft \
    "${common[@]}" \
    --output-dir "${OUTPUT_ROOT}/pilot" \
    --epochs 3 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --save-steps 10 \
    --generation-examples 6 \
    --min-json-rate 0.5
else
  echo "Unknown STAGE=${STAGE}" >&2
  exit 2
fi
