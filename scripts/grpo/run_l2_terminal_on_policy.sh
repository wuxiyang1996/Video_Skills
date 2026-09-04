#!/usr/bin/env bash
# Local L2 retrieval policy + fixed execution environment + terminal reward.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
PYTHON="${GRPO_PYTHON:-${CONDA_ENV}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_training/five_lora_pipeline_20260725/pilot/l2/pilot/adapter}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/split_manifest_v1.json}"
FROZEN_L1_GLOBS="${FROZEN_L1_GLOBS:-${FROZEN_L1_GLOB:-${REPO_ROOT}/dataset_clip_wrapper/output/pilot_20260710_free/**/04_l1_example.json}|${REPO_ROOT}/dataset_clip_wrapper/output/sft_auto_20260713_full_retrieval/**/04_l1_example.json|${REPO_ROOT}/dataset_clip_wrapper/output/l2_expansion_20260831/**/04_l1_example.json}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Qwen3.5 alternates rollout inference and differentiable pointwise scoring in
# one process.  Expandable CUDA segments prevent allocator fragmentation at
# that phase boundary without changing model arithmetic.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

# Optional release preflight for unattended paper runs.  A missing/failed gate or
# undersized mined pool stops before loading the model or consuming a GPU run.
if [[ -n "${RELEASE_GATE_FILES:-}" ]]; then
  IFS='|' read -r -a release_gates <<< "${RELEASE_GATE_FILES}"
  for gate in "${release_gates[@]}"; do
    [[ -f "${gate}" ]] || { echo "missing release gate: ${gate}" >&2; exit 20; }
    [[ "$(jq -r '.passed // false' "${gate}")" == "true" ]] || {
      echo "release gate did not pass: ${gate}" >&2
      exit 21
    }
  done
fi
if [[ -n "${MIN_ELIGIBLE_GROUPS_PER_DATASET:-}" ]]; then
  [[ -f "${MINING_REPORT:-}" ]] || {
    echo "missing mining report: ${MINING_REPORT:-unset}" >&2
    exit 22
  }
  mining_contract="select-coarse-clips-exact-v1"
  if [[ "${POINTWISE_ACTION_POLICY:-0}" == "1" ]]; then
    mining_contract="dataset-routed-cg-set-vh-pointwise-v1"
  fi
  mining_validation_args=(
    --report "${MINING_REPORT}"
    --source-adapter "${SFT_ADAPTER}"
    --controller-action-contract "${mining_contract}"
    --relationship-support-contract "structured-concept-overlap-v2"
    --generation-temperature "${TEMPERATURE:-0.9}"
    --pointwise-datasets "${POINTWISE_ACTION_DATASETS:-}"
    --min-eligible-per-dataset "${MIN_ELIGIBLE_GROUPS_PER_DATASET}"
    --min-eligible-group-rate "${MIN_TRAINABLE_GROUP_RATE:-0.25}"
  )
  if [[ -n "${POINTWISE_TEMPERATURE:-}" ]]; then
    mining_validation_args+=(--pointwise-temperature "${POINTWISE_TEMPERATURE}")
  elif [[ "${POINTWISE_ACTION_POLICY:-0}" == "1" ]]; then
    mining_validation_args+=(--pointwise-temperature "${TEMPERATURE:-0.9}")
  fi
  if [[ -n "${DATASET_ADAPTERS:-}" ]]; then
    IFS='|' read -r -a mining_adapter_routes <<< "${DATASET_ADAPTERS}"
    for route in "${mining_adapter_routes[@]}"; do
      [[ -n "${route}" ]] && mining_validation_args+=(--dataset-adapter "${route}")
    done
  fi
  "${PYTHON}" scripts/eval/validate_l2_mining_report.py "${mining_validation_args[@]}"
fi

args=(
  --adapter "${SFT_ADAPTER}"
  --split-manifest "${SPLIT_MANIFEST}"
  --split-role "${SPLIT_ROLE:-grpo_pool}"
  --output-dir "${OUTPUT_ROOT}"
  --dataset-root "${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}"
  --keys-py "${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"
  --max-groups "${MAX_GROUPS:-1}"
  --repeats-per-example "${REPEATS_PER_EXAMPLE:-1}"
  --repeat-start-index "${REPEAT_START_INDEX:-0}"
  --k "${K:-2}"
  --cg-topk "${CG_TOPK:-2}"
  --video-holmes-topk "${VIDEO_HOLMES_TOPK:-4}"
  --ppo-epochs "${PPO_EPOCHS:-1}"
  --learning-rate "${LEARNING_RATE:-2e-6}"
  --kl-coef "${KL_COEF:-0.05}"
  --clip-eps "${CLIP_EPS:-0.2}"
  --temperature "${TEMPERATURE:-0.9}"
  --max-new-tokens "${MAX_NEW_TOKENS:-384}"
  --generation-timeout-s "${GENERATION_TIMEOUT_S:-90}"
  --rollout-timeout-s "${ROLLOUT_TIMEOUT_S:-240}"
  --planner-model "${PLANNER_MODEL:-openai/gpt-oss-120b}"
  --skill-model "${SKILL_MODEL:-qwen/qwen3.5-9b}"
  --planner-timeout-s "${PLANNER_TIMEOUT_S:-180}"
  --skill-timeout-s "${SKILL_TIMEOUT_S:-90}"
  --min-catalog-size "${MIN_CATALOG_SIZE:-1}"
  --seed "${SEED:-42}"
  --checkpoint-every-groups "${CHECKPOINT_EVERY_GROUPS:-10}"
)
if [[ -n "${DATASET_ADAPTERS:-}" ]]; then
  IFS='|' read -r -a dataset_adapter_routes <<< "${DATASET_ADAPTERS}"
  for route in "${dataset_adapter_routes[@]}"; do
    [[ -n "${route}" ]] && args+=(--dataset-adapter "${route}")
  done
fi
if [[ -n "${POINTWISE_TEMPERATURE:-}" ]]; then
  args+=(--pointwise-temperature "${POINTWISE_TEMPERATURE}")
fi
if [[ -n "${POINTWISE_TRAIN_BATCH_SIZE:-}" ]]; then
  args+=(--pointwise-train-batch-size "${POINTWISE_TRAIN_BATCH_SIZE}")
fi
if [[ -n "${EXECUTOR_CACHE_DIR:-}" ]]; then
  args+=(--executor-cache-dir "${EXECUTOR_CACHE_DIR}")
fi
IFS='|' read -r -a frozen_l1_patterns <<< "${FROZEN_L1_GLOBS}"
for pattern in "${frozen_l1_patterns[@]}"; do
  [[ -n "${pattern}" ]] && args+=(--frozen-l1-glob "${pattern}")
done
if [[ -n "${DATASETS:-}" ]]; then
  args+=(--datasets "${DATASETS}")
fi
if [[ -n "${EXAMPLE_ID_ALLOWLIST:-}" ]]; then
  args+=(--example-id-allowlist "${EXAMPLE_ID_ALLOWLIST}")
fi
if [[ "${PRESERVE_ALLOWLIST_ORDER:-0}" == "1" ]]; then
  args+=(--preserve-allowlist-order)
fi
if [[ "${DATASET_BALANCED_SAMPLING:-0}" == "1" ]]; then
  args+=(--dataset-balanced-sampling)
fi
if [[ "${REQUIRE_PROCESS_SUPERVISION:-0}" == "1" ]]; then
  args+=(--require-process-supervision)
fi
if [[ -n "${MAX_CATALOG_SIZE:-}" ]]; then
  args+=(--max-catalog-size "${MAX_CATALOG_SIZE}")
fi
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  args+=(--eval-only)
fi
if [[ "${RETRIEVAL_ONLY:-0}" == "1" ]]; then
  args+=(--retrieval-only)
fi
if [[ "${PROCESS_REWARD_WARMUP:-0}" == "1" ]]; then
  args+=(--process-reward-warmup)
fi
if [[ "${TERMINAL_ON_PROCESS_HIT:-0}" == "1" ]]; then
  args+=(--terminal-on-process-hit)
fi
if [[ "${POINTWISE_ACTION_POLICY:-0}" == "1" ]]; then
  args+=(--pointwise-action-policy --pointwise-action-datasets "${POINTWISE_ACTION_DATASETS:-video_holmes}")
fi
if [[ "${ALLOW_SDPA_FALLBACK:-0}" == "1" ]]; then
  args+=(--allow-sdpa-fallback)
fi
if [[ "${BOUNDARY_ANCHOR_INDEX0:-0}" == "1" ]]; then
  args+=(--boundary-anchor-index0)
fi
"${PYTHON}" -m trainer.grpo.train_l2_terminal_on_policy "${args[@]}"
