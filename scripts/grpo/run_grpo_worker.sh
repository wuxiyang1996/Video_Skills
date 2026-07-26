#!/usr/bin/env bash
# A6000 worker: optional flash-attn install → collect → GPU GRPO train.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
# Prefer dedicated conda env; fall back to legacy serve venv.
if [[ -n "${GRPO_PYTHON:-}" && -x "${GRPO_PYTHON}" ]]; then
  PYTHON="${GRPO_PYTHON}"
elif [[ -x "${CONDA_ENV}/bin/python" ]]; then
  PYTHON="${CONDA_ENV}/bin/python"
else
  PYTHON="${REPO_ROOT}/.venv-qwen35-serve/bin/python"
fi
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"

STAGE="${STAGE:-smoke}"   # smoke|live_collect|gpu_train|all
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
FROZEN_L1_GLOB="${FROZEN_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json}"
MOTIF_BANK="${MOTIF_BANK:-${REPO_ROOT}/motif/output/pilot_online_motif_bank.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/grpo_a6000_$(date +%Y%m%d)}"
L2_ADAPTER="${L2_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/l2/pilot/adapter}"
REPAIR_ADAPTER="${REPAIR_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/repair/pilot/adapter}"
L1_ADAPTER="${L1_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/l1/pilot/adapter}"
MODE="${MODE:-l2_repair}"
K="${K:-4}"
LIMIT="${LIMIT:-8}"
MAX_GROUPS="${MAX_GROUPS:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
ALLOW_SDPA_FALLBACK="${ALLOW_SDPA_FALLBACK:-0}"
LIVE="${LIVE:-0}"
PLANNER_MODEL="${PLANNER_MODEL:-openai/gpt-oss-120b}"
SKILL_MODEL="${SKILL_MODEL:-qwen/qwen3.5-9b}"
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.7}"
WITH_SKILL_EXECUTOR="${WITH_SKILL_EXECUTOR:-1}"
ROTATE_MOTIFS="${ROTATE_MOTIFS:-1}"
FORCE_EXPLORE="${FORCE_EXPLORE:-1}"
EXPLORE_TOP_K="${EXPLORE_TOP_K:-3}"
DROP_DIRTY_SAMPLES="${DROP_DIRTY_SAMPLES:-1}"
SHARD_ID="${SHARD_ID:-}"
SHARD_COUNT="${SHARD_COUNT:-}"
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
  # Install FA2 into whichever python this worker selected.
  VENV_ROOT="$(dirname "$(dirname "${PYTHON}")")" \
    bash "${REPO_ROOT}/scripts/grpo/install_flash_attn.sh"
fi

ATTN_ARGS=()
if [[ "${ALLOW_SDPA_FALLBACK}" != "1" ]]; then
  "${PYTHON}" - <<'PY'
from trainer.grpo.attn_utils import resolve_attn_implementation
print(resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False))
PY
fi

# Shard collect writes under collect/shard_${ID}; single-job under collect/.
if [[ -n "${SHARD_ID}" ]]; then
  COLLECT_DIR="${OUTPUT_ROOT}/collect/shard_${SHARD_ID}"
else
  COLLECT_DIR="${OUTPUT_ROOT}/collect"
fi
TRAIN_DIR="${OUTPUT_ROOT}/train_${MODE}"
mkdir -p "${COLLECT_DIR}" "${TRAIN_DIR}"

# Collect: live takes precedence over mock when LIVE=1 on stage=all.
if [[ "${STAGE}" == "live_collect" || ( "${STAGE}" == "all" && "${LIVE}" == "1" ) ]]; then
  SKILL_FLAGS=()
  if [[ "${WITH_SKILL_EXECUTOR}" == "1" ]]; then
    SKILL_FLAGS+=(--with-skill-executor)
  else
    SKILL_FLAGS+=(--no-with-skill-executor)
  fi
  if [[ "${ROTATE_MOTIFS}" == "1" ]]; then
    SKILL_FLAGS+=(--rotate-motifs)
  else
    SKILL_FLAGS+=(--no-rotate-motifs)
  fi
  if [[ "${FORCE_EXPLORE}" == "1" ]]; then
    SKILL_FLAGS+=(--force-explore)
  else
    SKILL_FLAGS+=(--no-force-explore)
  fi
  if [[ "${DROP_DIRTY_SAMPLES}" == "1" ]]; then
    SKILL_FLAGS+=(--drop-dirty-samples)
  else
    SKILL_FLAGS+=(--no-drop-dirty-samples)
  fi
  SHARD_FLAGS=()
  if [[ -n "${SHARD_ID}" && -n "${SHARD_COUNT}" ]]; then
    SHARD_FLAGS+=(--shard-id "${SHARD_ID}" --shard-count "${SHARD_COUNT}")
  fi
  "${PYTHON}" -m trainer.grpo.collect_rollouts \
    --frozen-l1-glob "${FROZEN_L1_GLOB}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${COLLECT_DIR}" \
    --motif-bank "${MOTIF_BANK}" \
    --k "${K}" --limit "${LIMIT}" \
    --mode "${MODE}" \
    --live \
    --planner-model "${PLANNER_MODEL}" \
    --skill-model "${SKILL_MODEL}" \
    --skill-temperature "${SKILL_TEMPERATURE}" \
    --explore-top-k "${EXPLORE_TOP_K}" \
    --keys-py "${KEYS_PY}" \
    --judge-mock \
    "${SKILL_FLAGS[@]}" \
    "${SHARD_FLAGS[@]}"
elif [[ "${STAGE}" == "smoke" || "${STAGE}" == "all" ]]; then
  "${PYTHON}" -m trainer.grpo.collect_rollouts \
    --frozen-l1-glob "${FROZEN_L1_GLOB}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${COLLECT_DIR}" \
    --motif-bank "${MOTIF_BANK}" \
    --k "${K}" --limit "${LIMIT}" \
    --mode "${MODE}" \
    --smoke-mock-rollout
elif [[ "${STAGE}" == "merge_collect" ]]; then
  "${PYTHON}" -m trainer.grpo.merge_shard_groups \
    --shard-root "${OUTPUT_ROOT}/collect" \
    --output "${OUTPUT_ROOT}/collect/grpo_groups.jsonl" \
    --summary-out "${OUTPUT_ROOT}/collect/collect_summary.json"
fi

GROUPS_PATH="${OUTPUT_ROOT}/collect/grpo_groups.jsonl"
if [[ "${STAGE}" == "live_collect" && -n "${SHARD_ID}" ]]; then
  # Per-shard job: train is a separate merge+gpu_train stage.
  GROUPS_PATH="${COLLECT_DIR}/grpo_groups.jsonl"
fi
if [[ ! -f "${GROUPS_PATH}" && "${STAGE}" != "merge_collect" ]]; then
  # For non-shard all/smoke, groups live under COLLECT_DIR.
  if [[ -f "${COLLECT_DIR}/grpo_groups.jsonl" ]]; then
    GROUPS_PATH="${COLLECT_DIR}/grpo_groups.jsonl"
  else
    echo "missing groups: ${GROUPS_PATH}" >&2
    exit 2
  fi
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
  # Prefer merged groups when present (shard fan-out).
  if [[ -f "${OUTPUT_ROOT}/collect/grpo_groups.jsonl" ]]; then
    GROUPS_PATH="${OUTPUT_ROOT}/collect/grpo_groups.jsonl"
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

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
