#!/usr/bin/env bash
#
# Run Sokoban-specialized GPT-5.4 cold-start rollouts.
#
# This uses the domain-specific agent (generate_cold_start_sokoban.py) which
# includes Sokoban rules/strategy in the system prompt, a spatial grid
# observation format, rolling memory, and periodic reflections — unlike the
# generic cold-start script.
#
# Usage:
#   bash cold_start/run_coldstart_sokoban.sh
#       # 60 episodes (default), no labeling
#
#   bash cold_start/run_coldstart_sokoban.sh --episodes 10 --verbose
#       # 10 episodes with step-by-step output
#
#   bash cold_start/run_coldstart_sokoban.sh --episodes 60 --resume
#       # Resume an interrupted run
#
#   bash cold_start/run_coldstart_sokoban.sh --episodes 60 --no_label
#       # Skip trajectory labeling (faster)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEBASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GAMINGAGENT_ROOT="$(cd "$CODEBASE_ROOT/../GamingAgent" 2>/dev/null && pwd || echo "")"

export PYTHONPATH="${CODEBASE_ROOT}:${GAMINGAGENT_ROOT}:${PYTHONPATH:-}"

# Verify API key is set (see .env.example)
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Warning: OPENROUTER_API_KEY not set. See .env.example for required keys."
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY not set. See .env.example for required keys."
fi

echo "================================================================"
echo "  Sokoban-Specialized GPT-5.4 Cold-Start"
echo "================================================================"
echo "  Agent: domain-specific (grid obs + memory + reflection)"
echo "  Args:  $*"
echo "================================================================"
echo ""

python3 "${SCRIPT_DIR}/generate_cold_start_sokoban.py" "$@"

echo ""
echo "================================================================"
echo "  Done. Output: cold_start/output/gpt54_sokoban/sokoban/"
echo "================================================================"
