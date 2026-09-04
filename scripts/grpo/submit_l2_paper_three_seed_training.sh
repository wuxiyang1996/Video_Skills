#!/usr/bin/env bash
# Submit the frozen CG-Bench + Video-Holmes paper GRPO protocol for seeds 42/43/44.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PAPER_ROOT="${PAPER_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/l2_paper_cg_vh_20260901}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}"
CONDA_ENV="${CONDA_ENV:-/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo}"
KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"
SEEDS="${SEEDS:-42 43 44}"
RUN_PREFIX="${RUN_PREFIX:-grpo_main_v8alpha075_relv2_balanced200_k8}"
GRES="${GRES:-gpu:l40s:1}"
WALLTIME="${WALLTIME:-16:00:00}"
PARTITION="${PARTITION:-scavenger}"
ACCOUNT="${ACCOUNT:-scavenger}"
QOS="${QOS:-scavenger}"
DRY_RUN="${DRY_RUN:-0}"
EXECUTOR_CACHE_DIR="${EXECUTOR_CACHE_DIR:-${PAPER_ROOT}/executor_cache_grpo_main_v8_relv2_typedplan_v2}"

SFT_ADAPTER="${SFT_ADAPTER:-${PAPER_ROOT}/opd_interp_v8_relv2_grid/alpha075/adapter}"
MINING_ROOT="${MINING_ROOT:-${PAPER_ROOT}/retrieval_mining_opd_v8alpha075_relv2_balanced420_k8_r3_seed42}"
EXAMPLE_ID_ALLOWLIST="${EXAMPLE_ID_ALLOWLIST:-${MINING_ROOT}/eligible_exact_groups.tsv}"
MINING_REPORT="${MINING_REPORT:-${MINING_ROOT}/mining_report.json}"
RELEASE_GATE_FILES="${RELEASE_GATE_FILES:-${PAPER_ROOT}/terminal_dev_opd_v8alpha075_core10x8_pt09_relv2_seed42/gate_cg_bench.json|${PAPER_ROOT}/terminal_dev_opd_v8alpha075_core10x8_pt09_relv2_seed42/gate_video_holmes.json|${PAPER_ROOT}/grpo_pilot_v8alpha075_dev/pointwise_preservation_gate.json|${PAPER_ROOT}/terminal_dev_grpo_pilot_v8alpha075_core10x8_pt09_relv2_seed42/gate_cg_bench.json|${PAPER_ROOT}/terminal_dev_grpo_pilot_v8alpha075_core10x8_pt09_relv2_seed42/gate_video_holmes.json}"

for required in "${SFT_ADAPTER}/adapter_config.json" "${EXAMPLE_ID_ALLOWLIST}" "${MINING_REPORT}"; do
  [[ -f "${required}" ]] || { echo "missing frozen training input: ${required}" >&2; exit 2; }
done

# The longest frozen CG catalog repeatedly OOMed at group 54 on a 24 GB A5000.
# A formal run must not silently fall back to that resource class.
if [[ "${ALLOW_LOW_MEMORY_GPU:-0}" != "1" && "${GRES}" == *"rtxa5000"* ]]; then
  echo "formal L2 GRPO requires a >24 GB GPU; rtxa5000 is known to OOM" >&2
  exit 3
fi

submit_script="${REPO_ROOT}/scripts/grpo/submit_l2_terminal_on_policy.sh"
[[ -x "${submit_script}" ]] || { echo "missing executable submitter: ${submit_script}" >&2; exit 2; }

seed_specs=()
for seed in ${SEEDS}; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "invalid seed: ${seed}" >&2; exit 2; }
  output_root="${PAPER_ROOT}/${RUN_PREFIX}_seed${seed}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "seed=${seed} output=${output_root} gres=${GRES} walltime=${WALLTIME} allowlist=${EXAMPLE_ID_ALLOWLIST}"
    seed_specs+=("${seed}:DRY_RUN")
    continue
  fi
  submit_output="$(env \
    REPO_ROOT="${REPO_ROOT}" OUTPUT_ROOT="${output_root}" DATASET_ROOT="${DATASET_ROOT}" \
    SPLIT_ROLE=grpo_pool MAX_GROUPS=200 REPEATS_PER_EXAMPLE=1 REPEAT_START_INDEX=0 K=8 \
    CG_TOPK=2 VIDEO_HOLMES_TOPK=4 PPO_EPOCHS=1 LEARNING_RATE=5e-7 KL_COEF=0.05 \
    CLIP_EPS=0.2 TEMPERATURE=0.9 POINTWISE_TEMPERATURE=0.9 POINTWISE_TRAIN_BATCH_SIZE=1 \
    CHECKPOINT_EVERY_GROUPS=10 MAX_NEW_TOKENS=384 GENERATION_TIMEOUT_S=90 \
    ROLLOUT_TIMEOUT_S=240 PLANNER_MODEL=openai/gpt-oss-120b SKILL_MODEL=openai/gpt-oss-120b \
    PLANNER_TIMEOUT_S=180 SKILL_TIMEOUT_S=90 \
    EXECUTOR_CACHE_DIR="${EXECUTOR_CACHE_DIR}" \
    SEED="${seed}" CONDA_ENV="${CONDA_ENV}" KEYS_PY="${KEYS_PY}" SFT_ADAPTER="${SFT_ADAPTER}" \
    BOUNDARY_ANCHOR_INDEX0=0 ALLOW_SDPA_FALLBACK=0 EVAL_ONLY=0 RETRIEVAL_ONLY=0 \
    PROCESS_REWARD_WARMUP=0 TERMINAL_ON_PROCESS_HIT=1 POINTWISE_ACTION_POLICY=1 \
    POINTWISE_ACTION_DATASETS=video_holmes EXAMPLE_ID_ALLOWLIST="${EXAMPLE_ID_ALLOWLIST}" \
    PRESERVE_ALLOWLIST_ORDER=1 DATASET_BALANCED_SAMPLING=1 REQUIRE_PROCESS_SUPERVISION=1 \
    MIN_CATALOG_SIZE=1 RELEASE_GATE_FILES="${RELEASE_GATE_FILES}" MINING_REPORT="${MINING_REPORT}" \
    MIN_ELIGIBLE_GROUPS_PER_DATASET=50 MIN_TRAINABLE_GROUP_RATE=0.25 \
    PARTITION="${PARTITION}" ACCOUNT="${ACCOUNT}" QOS="${QOS}" GRES="${GRES}" \
    CPUS=8 MEM=64G WALLTIME="${WALLTIME}" "${submit_script}")"
  job_id="${submit_output##* -> }"
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "could not parse job id: ${submit_output}" >&2; exit 4; }
  seed_specs+=("${seed}:${job_id}")
  echo "${submit_output}"
done

joined="$(IFS='|'; echo "${seed_specs[*]}")"
echo "SEED_JOBS=${joined}"
