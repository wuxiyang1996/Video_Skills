#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
STAGE="${STAGE:?set STAGE=smoke|pilot|base_baseline}"

# Per-specialist layout (preferred): DATA_ROOT/<train|dev>.jsonl
# Legacy joint layout: DATA_ROOT/train_sft.jsonl + dev_sft.jsonl
SPECIALIST="${SPECIALIST:-}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v4/five_lora/${SPECIALIST:-l1}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/qwen35_9b_lora}"

if [[ -f "${DATA_ROOT}/train.jsonl" ]]; then
  TRAIN_JSONL="${DATA_ROOT}/train.jsonl"
  DEV_JSONL="${DATA_ROOT}/dev.jsonl"
elif [[ -f "${DATA_ROOT}/train_sft.jsonl" ]]; then
  TRAIN_JSONL="${DATA_ROOT}/train_sft.jsonl"
  DEV_JSONL="${DATA_ROOT}/dev_sft.jsonl"
else
  echo "No train.jsonl or train_sft.jsonl under DATA_ROOT=${DATA_ROOT}" >&2
  exit 2
fi

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

hostname
nvidia-smi || true
echo "STAGE=${STAGE} SPECIALIST=${SPECIALIST:-joint}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

common=(
  --model Qwen/Qwen3.5-9B
  --train-jsonl "${TRAIN_JSONL}"
  --dev-jsonl "${DEV_JSONL}"
  --stage "${STAGE}"
  --max-length 16384
  --lora-rank 16
  --lora-alpha 32
  --lora-dropout 0.05
  --seed 42
)

cd "${REPO_ROOT}"
# L1/L2 tool JSON can exceed 384 tokens; keep headroom for complete objects.
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-2048}"

if [[ "${STAGE}" == "smoke" ]]; then
  exec "${VENV_ROOT}/bin/python" -m dataset_clip_wrapper.training.train_lora_sft \
    "${common[@]}" \
    --output-dir "${OUTPUT_ROOT}/smoke" \
    --max-train-samples 20 \
    --max-eval-samples 16 \
    --max-steps 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-4 \
    --save-steps 4 \
    --generation-examples 4 \
    --generation-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
    --min-json-rate 0.0 \
    --min-action-rate 0.0
elif [[ "${STAGE}" == "pilot" ]]; then
  # Pilot = gate warm-up, not full production SFT.
  # L1 has ~7.6k rows; full 3-epoch is ~16h on 1xL40S. Default to a capped 1-epoch
  # representative subset (~1h). Set L1_FULL=1 for full-data substrate (default 1 epoch).
  GEN_EXAMPLES="${GEN_EXAMPLES:-16}"
  if [[ "${SPECIALIST}" == "l1" && "${L1_FULL:-0}" != "1" ]]; then
    EPOCHS="${EPOCHS:-1}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1536}"
    MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
  elif [[ "${SPECIALIST}" == "l1" && "${L1_FULL:-0}" == "1" ]]; then
    # Full-data substrate warm-up: one pass by default (set EPOCHS=3 for heavier SFT).
    EPOCHS="${EPOCHS:-1}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
    MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
  else
    EPOCHS="${EPOCHS:-3}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
    MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
  fi
  GRAD_ACCUM="${GRAD_ACCUM:-4}"
  echo "PILOT_CFG specialist=${SPECIALIST:-joint} epochs=${EPOCHS} max_train_samples=${MAX_TRAIN_SAMPLES} max_eval_samples=${MAX_EVAL_SAMPLES} grad_accum=${GRAD_ACCUM} gen_examples=${GEN_EXAMPLES} l1_full=${L1_FULL:-0}"
  # Frequent checkpoints help scavenger A100/H100 requeue.
  exec "${VENV_ROOT}/bin/python" -m dataset_clip_wrapper.training.train_lora_sft \
    "${common[@]}" \
    --output-dir "${OUTPUT_ROOT}/pilot" \
    --epochs "${EPOCHS}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --gradient-accumulation-steps "${GRAD_ACCUM}" \
    --learning-rate 1e-4 \
    --save-steps 10 \
    --generation-examples "${GEN_EXAMPLES}" \
    --generation-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
    --min-json-rate 0.5
elif [[ "${STAGE}" == "base_baseline" ]]; then
  exec "${VENV_ROOT}/bin/python" -m dataset_clip_wrapper.training.train_lora_sft \
    "${common[@]}" \
    --output-dir "${OUTPUT_ROOT}" \
    --generation-examples 8 \
    --generation-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
    --min-json-rate 0.0 \
    --min-action-rate 0.0
else
  echo "Unknown STAGE=${STAGE}" >&2
  exit 2
fi
