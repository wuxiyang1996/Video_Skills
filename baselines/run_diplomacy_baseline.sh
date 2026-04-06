#!/usr/bin/env bash
# ======================================================================
#  Diplomacy baseline: LLM API model as controlled power vs GPT-5.4
#
#  Cycles through all 7 powers. Each run the LLM controls one power
#  while GPT-5.4 plays the other 6.
#
#  Supported models:
#    gpt-5.4, openai/gpt-oss-120b, google/gemini-3.1-pro-preview,
#    anthropic/claude-4.6-sonnet-20260217
#
#  Usage:
#    bash baselines/run_diplomacy_baseline.sh
#    bash baselines/run_diplomacy_baseline.sh --model google/gemini-3.1-pro-preview
#    EPISODES_PER_POWER=4 bash baselines/run_diplomacy_baseline.sh --model gpt-5.4
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

MODEL="gpt-5.4"
while [[ $# -gt 0 ]]; do
    case "$1" in --model) MODEL="$2"; shift 2 ;; *) break ;; esac
done

NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-8}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy

AGENTEVOLVER_ROOT=""
for d in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    [ -d "$d" ] && AGENTEVOLVER_ROOT="$d" && break
done
[ -z "$AGENTEVOLVER_ROOT" ] && echo "[ERROR] AgentEvolver repo not found." && exit 1

export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

[ -z "${OPENROUTER_API_KEY:-}" ] && echo "Warning: OPENROUTER_API_KEY not set. See .env.example."
[ -z "${OPENAI_API_KEY:-}" ] && echo "Warning: OPENAI_API_KEY not set. See .env.example."

MODEL_TAG="${MODEL//\//_}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${MODEL_TAG}_diplomacy_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ${MODEL} Baseline — Diplomacy (${EPISODES} episodes = ${EPISODES_PER_POWER}/power × ${NUM_POWERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}  (controlled power)"
echo "  Opponents: ${OPPONENT_MODEL}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_evolver.py" \
    --games diplomacy \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --opponent_model "${OPPONENT_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --per_power \
    --no_label \
    --verbose \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "  ${MODEL} Diplomacy Baseline COMPLETE — ${OUTPUT_DIR}"
