#!/usr/bin/env bash
# Run scripts/run_trainer.py with hyperparameters. Override via args (passed through).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Hyperparameters (override by passing e.g. --config path/to.yaml)
CONFIG="${CONFIG:-trainer/common/configs/decision_grpo.yaml}"
DECISION_CONFIG="${DECISION_CONFIG:-trainer/common/configs/decision_grpo.yaml}"
SKILLBANK_CONFIG="${SKILLBANK_CONFIG:-trainer/common/configs/skillbank_em.yaml}"

exec python -m scripts.run_trainer \
  --config "$CONFIG" \
  --decision-config "$DECISION_CONFIG" \
  --skillbank-config "$SKILLBANK_CONFIG" \
  "$@"
