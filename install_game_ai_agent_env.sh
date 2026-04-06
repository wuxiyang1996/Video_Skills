#!/usr/bin/env bash
# =============================================================================
# install_game_ai_agent_env.sh
#
# Creates the "game-ai-agent" conda environment with all dependencies for:
#   - Game-AI-Agent trainer (GRPO, SkillBank, co-evolution)
#   - env_wrappers (2048, Candy Crush, Tetris, Super Mario, Avalon, Diplomacy)
#   - AgentEvolver (Avalon + Diplomacy)
#
# Prerequisites:
#   - miniconda3 or anaconda installed
#   - CUDA 12.x drivers on the host (for GPU training/inference)
#   - The following repos cloned as siblings under the same parent directory:
#       Game-AI-Agent/     (this repo)
#       GamingAgent/       (https://github.com/lmgame-org/GamingAgent)
#       AgentEvolver/      (https://github.com/modelscope/AgentEvolver)
#
# Usage:
#   cd /path/to/parent          # directory containing all repos above
#   bash Game-AI-Agent/install_game_ai_agent_env.sh [CONDA_PATH]
#
#   CONDA_PATH: optional path to conda binary (default: auto-detect)
#
# After install:
#   conda activate game-ai-agent
#   export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="game-ai-agent"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent of Game-AI-Agent = directory containing all sibling repos
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Locate conda
if [[ -n "${1:-}" ]]; then
    CONDA="$1"
elif command -v conda &>/dev/null; then
    CONDA="$(command -v conda)"
elif [[ -x "$PARENT_DIR/miniconda3/bin/conda" ]]; then
    CONDA="$PARENT_DIR/miniconda3/bin/conda"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Pass the path as an argument or install miniconda first."
    exit 1
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA")")"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  Game-AI-Agent environment installer"
echo "============================================================"
echo "  conda:       $CONDA"
echo "  env name:    $ENV_NAME"
echo "  python:      $PYTHON_VERSION"
echo "  parent dir:  $PARENT_DIR"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Repo checks
# ---------------------------------------------------------------------------
REPOS=(
    "Game-AI-Agent"
    "GamingAgent"
    "AgentEvolver"
)

MISSING=()
for repo in "${REPOS[@]}"; do
    if [[ ! -d "$PARENT_DIR/$repo" ]]; then
        MISSING+=("$repo")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "WARNING: The following repos are not found under $PARENT_DIR:"
    for m in "${MISSING[@]}"; do
        echo "  - $m"
    done
    echo
    echo "The installer will skip editable installs for missing repos."
    echo "Clone them later and run the relevant pip install -e commands."
    echo
fi

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[1/5] Conda env '$ENV_NAME' already exists, skipping creation."
else
    echo "[1/5] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    "$CONDA" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo

# ---------------------------------------------------------------------------
# Step 2: Trainer core dependencies
# ---------------------------------------------------------------------------
echo "[2/5] Installing trainer dependencies (torch, transformers, sentence-transformers, etc.) ..."
"$PIP" install --quiet \
    "numpy==1.26.4" \
    "pyyaml>=6.0" \
    "sentence-transformers>=2.7.0" \
    "transformers>=4.51.0" \
    "omegaconf>=2.3.0" \
    "hydra-core>=1.3.0"
echo

# ---------------------------------------------------------------------------
# Step 3: GamingAgent
# ---------------------------------------------------------------------------
echo "[3/5] Installing GamingAgent ..."
if [[ -d "$PARENT_DIR/GamingAgent" ]]; then
    "$PIP" install --quiet -e "$PARENT_DIR/GamingAgent"
    echo "  Installed GamingAgent (editable)"
    # GamingAgent pins numpy==1.24.4 but works fine with 1.26.4; restore it
    "$PIP" install --quiet "numpy==1.26.4"
    echo "  Restored numpy==1.26.4 (GamingAgent pin is overly strict)"
fi
echo

# ---------------------------------------------------------------------------
# Step 4: AgentEvolver (Avalon + Diplomacy)
# ---------------------------------------------------------------------------
echo "[4/5] Installing AgentEvolver eval dependencies (diplomacy, coloredlogs, loguru) ..."
"$PIP" install --quiet \
    "diplomacy>=1.1.2" \
    "coloredlogs" \
    "loguru>=0.7.0"

# AgentEvolver itself is added via PYTHONPATH (no pip install — its full
# requirements.txt would conflict with torch/transformers versions).
if [[ -d "$PARENT_DIR/AgentEvolver" ]]; then
    echo "  AgentEvolver found at $PARENT_DIR/AgentEvolver (use PYTHONPATH, not pip install)"
fi
echo

# ---------------------------------------------------------------------------
# Step 5: Final numpy pin & verification
# ---------------------------------------------------------------------------
echo "[5/5] Pinning numpy==1.26.4 and verifying ..."
"$PIP" install --quiet "numpy==1.26.4"

echo
echo "Running import checks ..."
PYTHONPATH="$PARENT_DIR/Game-AI-Agent:$PARENT_DIR/AgentEvolver:$PARENT_DIR/GamingAgent" \
"$PYTHON" -c "
import sys

failures = []
def check(label, fn):
    try:
        fn()
        print(f'  [OK]  {label}')
    except Exception as e:
        failures.append((label, str(e)))
        print(f'  [FAIL] {label}: {e}')

print(f'Python {sys.version}')
print()

# Trainer
check('numpy',                lambda: __import__('numpy'))
check('torch',                lambda: __import__('torch'))
check('transformers',         lambda: __import__('transformers'))
check('sentence_transformers',lambda: __import__('sentence_transformers'))
check('omegaconf',            lambda: __import__('omegaconf'))
check('hydra',                lambda: __import__('hydra'))

# GamingAgent
check('gymnasium',            lambda: __import__('gymnasium'))

# AgentEvolver (Avalon)
check('games.games.avalon',   lambda: __import__('games.games.avalon.engine'))

# Diplomacy
check('diplomacy',            lambda: __import__('diplomacy'))

# Trainer modules
check('trainer.common.metrics',         lambda: __import__('trainer.common.metrics'))
check('trainer.coevolution.config',     lambda: __import__('trainer.coevolution.config'))

print()
if failures:
    print(f'{len(failures)} check(s) failed:')
    for label, err in failures:
        print(f'  - {label}: {err}')
    sys.exit(1)
else:
    print('All checks passed.')
"

echo
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo
echo "  Activate:"
echo "    conda activate $ENV_NAME"
echo
echo "  Set PYTHONPATH (from the parent directory of all repos):"
echo "    export PYTHONPATH=\$(pwd)/Game-AI-Agent:\$(pwd)/AgentEvolver:\$(pwd)/GamingAgent:\$PYTHONPATH"
echo
echo "  Known nominal warning:"
echo "    gamingagent 0.1.0 requires numpy==1.24.4 (we use 1.26.4 — works fine)"
echo
echo "============================================================"
