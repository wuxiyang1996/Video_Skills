#!/usr/bin/env bash
# Run one local SFT -> on-policy action-GRPO specialist continuation.
# Formal default is L2. Repair is supported only for bounded diagnostics because
# the low-data specialists use OPD before any verified-reward GRPO experiment.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
SPECIALIST="${SPECIALIST:?set SPECIALIST=l2 or repair}"
SFT_ROOT="${SFT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/specialist_sft_v4/five_lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${GRPO_PYTHON:-${CONDA_ENV}/bin/python}"

case "${SPECIALIST}" in
  l2|repair) ;;
  *) echo "SPECIALIST must be l2 or repair" >&2; exit 2 ;;
esac

ADAPTER="${ADAPTER:-${SFT_ROOT}/pilot/${SPECIALIST}/pilot/adapter}"
TRAIN_JSONL="${TRAIN_JSONL:-${PACKAGE_ROOT}/${SPECIALIST}/train.jsonl}"
DEV_JSONL="${DEV_JSONL:-${PACKAGE_ROOT}/${SPECIALIST}/dev.jsonl}"

for path in "${PYTHON}" "${ADAPTER}" "${TRAIN_JSONL}" "${DEV_JSONL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required path: ${path}" >&2
    exit 2
  fi
done

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${OUTPUT_ROOT}/${SPECIALIST}"
cd "${REPO_ROOT}"

args=(
  --specialist "${SPECIALIST}"
  --adapter "${ADAPTER}"
  --train-jsonl "${TRAIN_JSONL}"
  --dev-jsonl "${DEV_JSONL}"
  --output-dir "${OUTPUT_ROOT}/${SPECIALIST}"
  --max-groups "${MAX_GROUPS:-32}"
  --k "${K:-4}"
  --ppo-epochs "${PPO_EPOCHS:-2}"
  --max-new-tokens "${MAX_NEW_TOKENS:-384}"
  --temperature "${TEMPERATURE:-0.8}"
  --learning-rate "${LEARNING_RATE:-2e-6}"
  --eval-samples "${EVAL_SAMPLES:-16}"
)
if [[ "${ALLOW_SDPA_FALLBACK:-0}" == "1" ]]; then
  args+=(--allow-sdpa-fallback)
fi
"${PYTHON}" -m trainer.grpo.train_specialist_on_policy "${args[@]}"
