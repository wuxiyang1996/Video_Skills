#!/usr/bin/env bash
#
# run_coldstart_orak_sc2.sh — GPT-5.4 cold-start rollouts for StarCraft II (Orak)
#
# Activates the orak-sc2 conda environment, sets SC2PATH, and runs
# generate_cold_start_orak.py for star_craft.
#
# Usage:
#   bash cold_start/run_coldstart_orak_sc2.sh                       # 10 episodes (default)
#   bash cold_start/run_coldstart_orak_sc2.sh --episodes 3 -v       # quick test
#   bash cold_start/run_coldstart_orak_sc2.sh --workers 4           # 4 parallel SC2 instances
#   bash cold_start/run_coldstart_orak_sc2.sh --resume              # resume interrupted run
#   bash cold_start/run_coldstart_orak_sc2.sh --help                # all options
#
# All extra arguments are forwarded to generate_cold_start_orak.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(dirname "$SCRIPT_DIR")"
ORAK_SRC="${CODEBASE_ROOT}/../Orak/src"

# ── Conda ──────────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate orak-sc2

export SC2PATH="/workspace/game_agent/StarCraftII/StarCraftII"
export PYTHONPATH="${CODEBASE_ROOT}:${ORAK_SRC}:${PYTHONPATH:-}"

# ── API key ────────────────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    OPENROUTER_API_KEY="$(python -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import open_router_api_key; print(open_router_api_key or '')
" 2>/dev/null || echo "")"
    export OPENROUTER_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY="${OPENROUTER_API_KEY:-$(python -c "
import sys; sys.path.insert(0, '${CODEBASE_ROOT}')
from api_keys import openai_api_key; print(openai_api_key or '')
" 2>/dev/null || echo "")}"
    export OPENAI_API_KEY
fi

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] No API key found."
    echo "  Set OPENROUTER_API_KEY, or open_router_api_key in api_keys.py, or OPENAI_API_KEY."
    exit 1
fi

# ── Banner ─────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  GPT-5.4 Cold-Start — Orak StarCraft II"
echo "================================================================"
echo "  Python:     $(python --version 2>&1)"
echo "  Conda env:  $(conda info --envs | grep '*' | awk '{print $1}')"
echo "  SC2PATH:    ${SC2PATH}"
echo "  PYTHONPATH: ${PYTHONPATH}"
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "  API key:    ${OPENROUTER_API_KEY:0:12}... (OpenRouter)" || echo "  API key:    ${OPENAI_API_KEY:0:12}..."
echo "================================================================"
echo ""
echo "  SC2 maps available:"
ls -1 "${SC2PATH}/Maps/Ladder2019Season1/" 2>/dev/null | sed 's/\.SC2Map//' | sed 's/^/    /' || echo "    (maps directory not found)"
echo ""

# ── Run ────────────────────────────────────────────────────────────────────
EXTRA_ARGS=("$@")

if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=(--episodes 10 --max_steps 500 --workers 4 --no_label --resume -v)
fi

python "${SCRIPT_DIR}/generate_cold_start_orak.py" \
    --games star_craft \
    "${EXTRA_ARGS[@]}"

# ── Post-run ───────────────────────────────────────────────────────────────
OUTPUT_DIR="${SCRIPT_DIR}/output/gpt54_orak/star_craft"
echo ""
echo "================================================================"
echo "  Post-Run Summary — StarCraft II"
echo "================================================================"
if [ -d "$OUTPUT_DIR" ]; then
    COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'episode_*.json' ! -name 'episode_buffer.json' | wc -l)
    echo "  Episodes saved:  $COUNT"
    echo "  Output dir:      $OUTPUT_DIR"
    [ -f "$OUTPUT_DIR/rollout_summary.json" ] && echo "  Summary:         $OUTPUT_DIR/rollout_summary.json"
else
    echo "  (no output yet)"
fi
echo "================================================================"
