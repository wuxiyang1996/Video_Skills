#!/usr/bin/env bash
# ======================================================================
#  Avalon baseline: LLM API model as single agent vs GPT-5.4
#
#  Cycles through all 5 player positions. Each run the LLM controls one
#  seat while GPT-5.4 plays the other 4.
#
#  Supported models:
#    gpt-5.4, openai/gpt-oss-120b, google/gemini-3.1-pro-preview,
#    anthropic/claude-4.6-sonnet-20260217
#
#  Usage:
#    bash baselines/run_avalon_baseline.sh
#    bash baselines/run_avalon_baseline.sh --model anthropic/claude-4.6-sonnet-20260217
#    EPISODES_PER_PLAYER=4 bash baselines/run_avalon_baseline.sh --model gpt-5.4
# ======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

MODEL="gpt-5.4"
while [[ $# -gt 0 ]]; do
    case "$1" in --model) MODEL="$2"; shift 2 ;; *) break ;; esac
done

NUM_PLAYERS=5
EPISODES_PER_PLAYER="${EPISODES_PER_PLAYER:-8}"
EPISODES=$((NUM_PLAYERS * EPISODES_PER_PLAYER))
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
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/${MODEL_TAG}_avalon_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  ${MODEL} Baseline — Avalon (${EPISODES} episodes = ${EPISODES_PER_PLAYER}/player × ${NUM_PLAYERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}  (controlled player)"
echo "  Opponents: ${OPPONENT_MODEL}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/cold_start/generate_cold_start_evolver.py" \
    --games avalon \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --opponent_model "${OPPONENT_MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --per_role \
    --no_label \
    --verbose \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "  ${MODEL} Avalon Baseline COMPLETE — ${OUTPUT_DIR}"
