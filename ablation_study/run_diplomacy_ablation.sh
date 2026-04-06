#!/usr/bin/env bash
# ======================================================================
#  Diplomacy ablation: adapter × skill bank evaluation vs GPT-5.4
#
#  Adapter options:   base  — vanilla Qwen3-8B, no LoRA
#                     sft   — SFT cold-start LoRA
#                     coevo — co-evolution best LoRA (step 17)
#
#  Bank options:      none  — no skill bank
#                     first — step 0 skill bank (initial discovery)
#                     best  — step 17 skill bank (final)
#
#  Valid combinations (matches Table 2 of the paper):
#    --adapter base                        (base model, no bank)
#    --adapter sft   --bank none           (SFT only)
#    --adapter sft   --bank first          (SFT + initial bank)
#    --adapter sft   --bank best           (SFT + best bank)
#    --adapter coevo --bank none           (co-evo agent, no bank)
#    --adapter coevo --bank best           (full system)
#
#  Usage:
#    bash ablation_study/run_diplomacy_ablation.sh --adapter coevo --bank best
#    bash ablation_study/run_diplomacy_ablation.sh --adapter base
#    EVAL_GPUS=0 bash ablation_study/run_diplomacy_ablation.sh --adapter sft --bank none
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Parse arguments ──────────────────────────────────────────────────
ADAPTER_TYPE=""
BANK_TYPE="none"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter) ADAPTER_TYPE="$2"; shift 2 ;;
        --bank)    BANK_TYPE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "${ADAPTER_TYPE}" ]; then
    echo "Usage: $0 --adapter {base|sft|coevo} [--bank {none|first|best}]"
    exit 1
fi

# ── Environment ──────────────────────────────────────────────────────
export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

# ── Checkpoint paths ─────────────────────────────────────────────────
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
COEVO_RUN_DIR="${RUNS_DIR}/Qwen3-8B_20260327_062035"
COEVO_BEST_STEP="step_0017"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-4}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
EVAL_GPUS="${EVAL_GPUS:-0}"
VLLM_PORT="${VLLM_PORT:-8025}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
NO_SERVER="${NO_SERVER:-0}"
SEED="${SEED:-42}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Resolve adapter ──────────────────────────────────────────────────
ENABLE_LORA=0
ADAPTER_PATH=""
LORA_NAME=""
MODEL_FOR_EVAL="${BASE_MODEL}"

case "${ADAPTER_TYPE}" in
    base)
        ADAPTER_DESC="NONE (vanilla base model)"
        ;;
    sft)
        ADAPTER_PATH="${RUNS_DIR}/sft_coldstart/decision/action_taking/action_taking"
        LORA_NAME="qwen3-8b-diplomacy-sft"
        ENABLE_LORA=1
        MODEL_FOR_EVAL="${LORA_NAME}"
        ADAPTER_DESC="SFT cold-start → ${ADAPTER_PATH}"
        [ ! -f "${ADAPTER_PATH}/adapter_config.json" ] && echo "[ERROR] SFT adapter not found: ${ADAPTER_PATH}" && exit 1
        ;;
    coevo)
        ADAPTER_PATH="${COEVO_RUN_DIR}/checkpoints/${COEVO_BEST_STEP}/adapters/decision/action_taking"
        LORA_NAME="qwen3-8b-diplomacy-coevo"
        ENABLE_LORA=1
        MODEL_FOR_EVAL="${LORA_NAME}"
        ADAPTER_DESC="Co-evolution ${COEVO_BEST_STEP} → ${ADAPTER_PATH}"
        [ ! -f "${ADAPTER_PATH}/adapter_config.json" ] && echo "[ERROR] Co-evo adapter not found: ${ADAPTER_PATH}" && exit 1
        ;;
    *)  echo "[ERROR] Unknown adapter type: ${ADAPTER_TYPE}. Use: base, sft, coevo"; exit 1 ;;
esac

# ── Resolve skill bank ───────────────────────────────────────────────
BANK_ARGS=()
BANK_DESC="NONE (--no-bank)"

case "${BANK_TYPE}" in
    none)
        BANK_ARGS+=(--no-bank)
        ;;
    first|best)
        if [ "${BANK_TYPE}" = "first" ]; then
            BANK_STEP="step_0000"
        else
            BANK_STEP="${COEVO_BEST_STEP}"
        fi
        BANKS_DIR="${COEVO_RUN_DIR}/checkpoints/${BANK_STEP}/banks/diplomacy"
        BANK_FILES=()
        for power in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
            pbank="${BANKS_DIR}/${power}/skill_bank.jsonl"
            [ ! -f "${pbank}" ] && echo "[ERROR] Bank not found for ${power}: ${pbank}" && exit 1
            BANK_FILES+=("${pbank}")
        done
        ;;
    *)  echo "[ERROR] Unknown bank type: ${BANK_TYPE}. Use: none, first, best"; exit 1 ;;
esac

# ── Output directory ─────────────────────────────────────────────────
TAG="${ADAPTER_TYPE}_${BANK_TYPE}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_DIR:-${PROJECT_ROOT}/ablation_study/output/${TAG}_diplomacy_da_${TIMESTAMP}}"
mkdir -p "${OUTPUT_BASE}"

# Combine bank files if needed
if [ "${BANK_TYPE}" != "none" ]; then
    COMBINED_BANK="${OUTPUT_BASE}/combined_skill_bank.jsonl"
    cat "${BANK_FILES[@]}" | sort -u > "${COMBINED_BANK}"
    COMBINED_COUNT=$(wc -l < "${COMBINED_BANK}")
    BANK_ARGS+=(--bank "${COMBINED_BANK}")
    BANK_DESC="${COMBINED_BANK} (${COMBINED_COUNT} skills, ${BANK_STEP})"
fi

# ── Cleanup on exit ──────────────────────────────────────────────────
VLLM_PID=""
cleanup() {
    set +e
    echo ""
    echo "[${TAG}-diplomacy] Shutting down..."
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        kill -INT "${VLLM_PID}" 2>/dev/null || true
        for _ in {1..30}; do kill -0 "${VLLM_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    echo "[${TAG}-diplomacy] Done."
}
trap cleanup EXIT INT TERM

# ── Banner ───────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  Diplomacy Ablation: adapter=${ADAPTER_TYPE}  bank=${BANK_TYPE}"
echo "              vs ${OPPONENT_MODEL}"
echo "══════════════════════════════════════════════════════════════"
echo "  Base model:   ${BASE_MODEL}"
echo "  Adapter:      ${ADAPTER_DESC}"
echo "  Skill bank:   ${BANK_DESC}"
echo "  Opponent:     ${OPPONENT_MODEL}"
echo "  Episodes:     ${EPISODES} (${EPISODES_PER_POWER}/power × ${NUM_POWERS})"
echo "  GPU(s):       ${EVAL_GPUS}"
echo "  Output:       ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Launch vLLM server ───────────────────────────────────────────────
if [ "${NO_SERVER}" = "0" ]; then
    VLLM_LOG="${OUTPUT_BASE}/vllm_server.log"
    VLLM_ARGS=(
        --model "${BASE_MODEL}" --host "${VLLM_HOST}" --port "${VLLM_PORT}"
        --tensor-parallel-size "${TENSOR_PARALLEL}" --max-model-len 8192
        --gpu-memory-utilization 0.90 --dtype auto --trust-remote-code
    )
    if [ "${ENABLE_LORA}" = "1" ]; then
        VLLM_ARGS+=(--enable-lora --max-loras 2 --max-lora-rank 64)
        VLLM_ARGS+=(--lora-modules "${LORA_NAME}=${ADAPTER_PATH}")
    fi

    echo "[${TAG}-diplomacy] Launching vLLM server..."
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
        python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!

    MAX_WAIT=600; WAITED=0
    while [ ${WAITED} -lt ${MAX_WAIT} ]; do
        curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1 && break
        kill -0 "${VLLM_PID}" 2>/dev/null || { echo "[ERROR] vLLM exited. See ${VLLM_LOG}"; exit 1; }
        sleep 5; WAITED=$((WAITED + 5))
    done
    [ ${WAITED} -ge ${MAX_WAIT} ] && echo "[ERROR] vLLM did not start within ${MAX_WAIT}s." && exit 1
    echo "[${TAG}-diplomacy] vLLM ready (${WAITED}s)."
else
    echo "[${TAG}-diplomacy] Using existing server at ${VLLM_BASE_URL}"
fi

# ── Run evaluation ───────────────────────────────────────────────────
EVAL_ARGS=(
    --games diplomacy --episodes "${EPISODES}" --temperature "${TEMPERATURE}"
    --model "${MODEL_FOR_EVAL}" --seed "${SEED}" --output_dir "${OUTPUT_BASE}"
    --opponent_model "${OPPONENT_MODEL}" --per_power --verbose
    "${BANK_ARGS[@]}"
)

echo "[${TAG}-diplomacy] python -m inference.run_qwen3_8b_eval ${EVAL_ARGS[*]}"
echo ""

EXIT_CODE=0
python -m inference.run_qwen3_8b_eval "${EVAL_ARGS[@]}" || EXIT_CODE=$?

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "[${TAG}-diplomacy] Output: ${OUTPUT_BASE}"
echo "══════════════════════════════════════════════════════════════"
exit ${EXIT_CODE}
