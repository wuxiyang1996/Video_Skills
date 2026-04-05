#!/usr/bin/env bash
# Claude Sonnet 4.6 baseline on Diplomacy (56 episodes = 8/power x 7) via OpenRouter
# Controlled power = Claude 4.6, all opponents = GPT-5.4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

AGENTEVOLVER_ROOT=""
for candidate in "$WORKSPACE_ROOT/AgentEvolver" "$PROJECT_ROOT/AgentEvolver"; do
    [ -d "$candidate" ] && AGENTEVOLVER_ROOT="$candidate" && break
done
[ -z "$AGENTEVOLVER_ROOT" ] && echo "[ERROR] AgentEvolver repo not found." && exit 1

AI_DIPLOMACY_ROOT=""
for candidate in "$WORKSPACE_ROOT/AI_Diplomacy" "$PROJECT_ROOT/AI_Diplomacy"; do
    [ -d "$candidate" ] && AI_DIPLOMACY_ROOT="$candidate" && break
done

NUM_POWERS=7
EPISODES_PER_POWER="${EPISODES_PER_POWER:-8}"
EPISODES=$((NUM_POWERS * EPISODES_PER_POWER))
TEMPERATURE="${TEMPERATURE:-0.4}"
MODEL="${MODEL:-anthropic/claude-4.6-sonnet-20260217}"
OPPONENT_MODEL="${OPPONENT_MODEL:-gpt-5.4}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${AGENTEVOLVER_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
[ -n "$AI_DIPLOMACY_ROOT" ] && export PYTHONPATH="${AI_DIPLOMACY_ROOT}:${PYTHONPATH}"

# ── Verify API key is set (see .env.example) ─────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. See .env.example for required keys."
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/claude46_diplomacy_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Claude Sonnet 4.6 Baseline — Diplomacy (${EPISODES} episodes = ${EPISODES_PER_POWER}/power x ${NUM_POWERS})"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}  (controlled power)"
echo "  Opponents: ${OPPONENT_MODEL}"
echo "  Episodes:  ${EPISODES}    Temperature: ${TEMPERATURE}"
echo "  Mode:      per-power (cycle through 7 powers)"
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
echo "  Claude Sonnet 4.6 Diplomacy Baseline COMPLETE — ${OUTPUT_DIR}"
