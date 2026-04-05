#!/usr/bin/env bash
# Claude Sonnet 4.6 baseline on Tetris (8 episodes) via OpenRouter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$PROJECT_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

if [ -z "$GAMINGAGENT_ROOT" ] || [ ! -d "$GAMINGAGENT_ROOT" ]; then
    echo "[ERROR] GamingAgent repo not found at $PROJECT_ROOT/../GamingAgent"; exit 1
fi

EPISODES="${EPISODES:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MODEL="${MODEL:-anthropic/claude-4.6-sonnet-20260217}"
SEED="${SEED:-42}"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${GAMINGAGENT_ROOT}:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"

# ── Verify API key is set (see .env.example) ─────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. See .env.example for required keys."
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/claude46_tetris_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  Claude Sonnet 4.6 Baseline — Tetris (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}"
echo "  Episodes:  ${EPISODES}    Max steps: ${MAX_STEPS}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${SCRIPT_DIR}/run_gpt54_tetris_macro.py" \
    --episodes "${EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --verbose \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "  Claude Sonnet 4.6 Tetris Baseline COMPLETE — ${OUTPUT_DIR}"
