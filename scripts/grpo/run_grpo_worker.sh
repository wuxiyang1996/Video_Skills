#!/usr/bin/env bash
# A6000 worker: optional flash-attn install → collect → GPU GRPO train.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
PYTHON="${VENV_ROOT}/bin/python"

STAGE="${STAGE:-smoke}"   # smoke|live_collect|gpu_train|all
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
FROZEN_L1_GLOB="${FROZEN_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json}"
MOTIF_BANK="${MOTIF_BANK:-${REPO_ROOT}/motif/output/pilot_online_motif_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/grpo_a6000_$(date +%Y%m%d)}"
L2_ADAPTER="${L2_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/l2/pilot/adapter}"
REPAIR_ADAPTER="${REPAIR_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/repair/pilot/adapter}"
L1_ADAPTER="${L1_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot_l1_full/l1/pilot/adapter}"
MODE="${MODE:-l2_repair}"
K="${K:-4}"
LIMIT="${LIMIT:-8}"
MAX_GROUPS="${MAX_GROUPS:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
ALLOW_SDPA_FALLBACK="${ALLOW_SDPA_FALLBACK:-0}"
LIVE="${LIVE:-0}"
PLANNER_MODEL="${PLANNER_MODEL:-openai/gpt-oss-120b}"
KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
hostname
nvidia-smi || true

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing ${PYTHON}" >&2
  exit 2
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  bash "${REPO_ROOT}/scripts/grpo/install_flash_attn.sh"
fi

ATTN_ARGS=()
if [[ "${ALLOW_SDPA_FALLBACK}" != "1" ]]; then
  "${PYTHON}" - <<'PY'
from trainer.grpo.attn_utils import resolve_attn_implementation
print(resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False))
PY
fi

COLLECT_DIR="${OUTPUT_ROOT}/collect"
TRAIN_DIR="${OUTPUT_ROOT}/train_${MODE}"
mkdir -p "${COLLECT_DIR}" "${TRAIN_DIR}"

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "all" ]]; then
  "${PYTHON}" -m trainer.grpo.collect_rollouts \
    --frozen-l1-glob "${FROZEN_L1_GLOB}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${COLLECT_DIR}" \
    --motif-bank "${MOTIF_BANK}" \
    --k "${K}" --limit "${LIMIT}" \
    --mode "${MODE}" \
    --smoke-mock-rollout
fi

if [[ "${STAGE}" == "live_collect" || ( "${STAGE}" == "all" && "${LIVE}" == "1" ) ]]; then
  "${PYTHON}" -m trainer.grpo.collect_rollouts \
    --frozen-l1-glob "${FROZEN_L1_GLOB}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${COLLECT_DIR}" \
    --motif-bank "${MOTIF_BANK}" \
    --k "${K}" --limit "${LIMIT}" \
    --mode "${MODE}" \
    --live \
    --planner-model "${PLANNER_MODEL}" \
    --keys-py "${KEYS_PY}" \
    --judge-mock
fi

GROUPS_PATH="${COLLECT_DIR}/grpo_groups.jsonl"
if [[ ! -f "${GROUPS_PATH}" ]]; then
  echo "missing groups: ${GROUPS_PATH}" >&2
  exit 2
fi

if [[ "${STAGE}" == "gpu_train" || "${STAGE}" == "all" || "${STAGE}" == "smoke" ]]; then
  GPU_FLAGS=(--gpu)
  if [[ "${ALLOW_SDPA_FALLBACK}" == "1" ]]; then
    GPU_FLAGS+=(--allow-sdpa-fallback)
  fi
  L2_STABLE_FLAGS=()
  if [[ "${MODE}" == "joint_l1" ]]; then
    L2_STABLE_FLAGS+=(--l2-stable)
  fi
  "${PYTHON}" -m trainer.grpo.train_verified \
    --groups "${GROUPS_PATH}" \
    --output-dir "${TRAIN_DIR}" \
    --mode "${MODE}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --base-model Qwen/Qwen3.5-9B \
    --l2-adapter "${L2_ADAPTER}" \
    --repair-adapter "${REPAIR_ADAPTER}" \
    --l1-adapter "${L1_ADAPTER}" \
    --max-groups "${MAX_GROUPS}" \
    "${GPU_FLAGS[@]}" \
    "${L2_STABLE_FLAGS[@]}"
fi

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
