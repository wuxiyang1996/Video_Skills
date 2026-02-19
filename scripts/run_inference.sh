#!/usr/bin/env bash
# Run scripts/run_inference.py with hyperparameters. Override via args (passed through).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Hyperparameters (override by passing e.g. --game overcooked --max-steps 500)
GAME="${GAME:-gamingagent}"
TASK="${TASK:-}"
MAX_STEPS="${MAX_STEPS:-1000}"
MODEL="${MODEL:-gpt-4o-mini}"
SAVE_PATH="${SAVE_PATH:-}"
EPISODE_BUFFER_SIZE="${EPISODE_BUFFER_SIZE:-100}"
EXPERIENCE_BUFFER_SIZE="${EXPERIENCE_BUFFER_SIZE:-10000}"

args=(--game "$GAME" --task "$TASK" --max-steps "$MAX_STEPS" --model "$MODEL" \
  --episode-buffer-size "$EPISODE_BUFFER_SIZE" --experience-buffer-size "$EXPERIENCE_BUFFER_SIZE")
[[ -n "$SAVE_PATH" ]] && args+=(--save-path "$SAVE_PATH")
exec python -m scripts.run_inference "${args[@]}" "$@"
