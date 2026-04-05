#!/usr/bin/env bash
# GPT-OSS 120B baseline on Super Mario (8 episodes) via OpenRouter
#
# Requires orak-mario conda env for NES emulator (NumPy 1.x).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

export PYGLET_HEADLESS=1
export SDL_VIDEODRIVER=dummy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../GamingAgent:${PROJECT_ROOT}/../AgentEvolver:${PROJECT_ROOT}/../AI_Diplomacy:${PROJECT_ROOT}/../Orak:${PYTHONPATH:-}"
ORAK_SRC="${WORKSPACE_ROOT}/Orak/src"
[ -d "$ORAK_SRC" ] && export PYTHONPATH="${ORAK_SRC}:${PYTHONPATH}"

# ── Xvfb for NES rendering ──────────────────────────────────────────
if [ -z "${DISPLAY:-}" ]; then
    if command -v Xvfb &>/dev/null; then
        if ! pgrep -f "Xvfb :99" &>/dev/null; then
            Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
            sleep 1
        fi
        export DISPLAY=":99"
    fi
fi

EPISODES="${EPISODES:-8}"
TEMPERATURE="${TEMPERATURE:-0.3}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
SEED="${SEED:-42}"

# ── Verify API key is set (see .env.example) ─────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. See .env.example for required keys."
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/gptoss120b_super_mario_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

echo "══════════════════════════════════════════════════════════════"
echo "  GPT-OSS 120B Baseline — Super Mario (${EPISODES} episodes)"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:     ${MODEL}"
echo "  Episodes:  ${EPISODES}"
echo "  Output:    ${OUTPUT_DIR}"
echo "══════════════════════════════════════════════════════════════"
echo ""

python3 "${PROJECT_ROOT}/evaluate_orak/test_orak_mario_sc2_gpt54.py" \
    --game super_mario \
    --episodes "${EPISODES}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "  GPT-OSS 120B Super Mario Baseline COMPLETE — ${OUTPUT_DIR}"
