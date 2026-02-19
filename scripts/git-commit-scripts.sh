#!/usr/bin/env bash
# Commit scripts (trainer + inference .sh and Python with hyperparameters).
set -e
cd "$(dirname "$0")/.."
git add scripts/
git commit -m "scripts: add run_trainer.sh and run_inference.sh to run Python with hyperparameters

- run_trainer.sh: runs scripts.run_trainer with CONFIG, DECISION_CONFIG, SKILLBANK_CONFIG
- run_inference.sh: runs scripts.run_inference with GAME, TASK, MAX_STEPS, MODEL, SAVE_PATH, buffer sizes
- Python: run_trainer.py, run_inference.py, trainer_defaults.py, inference_defaults.py"
