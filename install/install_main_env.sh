#!/usr/bin/env bash
# =============================================================================
# install_main_env.sh
#
# Creates the "game-ai-agent" conda environment with ALL dependencies for:
#   - COS-PLAY co-evolution training (GRPO + FSDP + LoRA)
#   - Skill bank pipeline (boundary proposal, segmentation, contract, curation)
#   - RAG retrieval (Qwen3-Embedding-0.6B)
#   - Cold-start data generation & labeling
#   - Inference and evaluation (vLLM)
#   - Baselines (OpenRouter API)
#   - Game environments: 2048, Candy Crush, Tetris, Avalon, Diplomacy
#
# NOTE: Super Mario requires a SEPARATE environment (orak-mario).
#       See install/install_orak_mario.sh.
#
# Prerequisites:
#   - Miniconda3 or Anaconda installed
#   - CUDA 12.x drivers on the host (for GPU training / vLLM)
#   - The following repos cloned as siblings under the same parent directory:
#       Game-AI-Agent/     (this repo)
#       GamingAgent/       (https://github.com/lmgame-org/GamingAgent)
#       AgentEvolver/      (https://github.com/modelscope/AgentEvolver)
#
# Usage:
#   cd /path/to/parent           # directory containing all repos
#   bash Game-AI-Agent/install/install_main_env.sh [CONDA_PATH]
#
# After install:
#   conda activate game-ai-agent
#   export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
#   cp Game-AI-Agent/.env.example Game-AI-Agent/.env
#   # Edit .env with your API keys, then:
#   set -a && source Game-AI-Agent/.env && set +a
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="game-ai-agent"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"          # Game-AI-Agent/
PARENT_DIR="$(dirname "$REPO_DIR")"          # parent of all sibling repos
REQS="${SCRIPT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Locate conda
# ---------------------------------------------------------------------------
if [[ -n "${1:-}" ]]; then
    CONDA="$1"
elif command -v conda &>/dev/null; then
    CONDA="$(command -v conda)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA="$HOME/miniconda3/bin/conda"
elif [[ -x "/workspace/miniconda3/bin/conda" ]]; then
    CONDA="/workspace/miniconda3/bin/conda"
else
    echo "ERROR: conda not found. Pass the conda path as an argument or install Miniconda first."
    echo "  curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh | bash"
    exit 1
fi

CONDA_DIR="$(dirname "$(dirname "$CONDA")")"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"
PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"

echo "============================================================"
echo "  COS-PLAY — Main Environment Installer"
echo "============================================================"
echo "  conda:       $CONDA"
echo "  env name:    $ENV_NAME"
echo "  python:      $PYTHON_VERSION"
echo "  repo dir:    $REPO_DIR"
echo "  parent dir:  $PARENT_DIR"
echo "============================================================"
echo

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
if "$CONDA" env list | grep -q "^${ENV_NAME} "; then
    echo "[1/6] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[1/6] Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION ..."
    "$CONDA" create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi
echo

# ---------------------------------------------------------------------------
# Step 2: Install PyTorch with CUDA
# ---------------------------------------------------------------------------
echo "[2/6] Installing PyTorch with CUDA 12.x ..."
"$PIP" install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo

# ---------------------------------------------------------------------------
# Step 3: Install all pip requirements
# ---------------------------------------------------------------------------
echo "[3/6] Installing pip requirements from $REQS ..."
"$PIP" install --quiet -r "$REQS"
echo

# ---------------------------------------------------------------------------
# Step 4: Install GamingAgent (editable)
# ---------------------------------------------------------------------------
echo "[4/6] Installing GamingAgent ..."
if [[ -d "$PARENT_DIR/GamingAgent" ]]; then
    "$PIP" install --quiet -e "$PARENT_DIR/GamingAgent"
    echo "  ✓ Installed GamingAgent (editable)"
    # GamingAgent pins numpy==1.24.4 but works fine with 1.26.4; restore it
    "$PIP" install --quiet "numpy==1.26.4"
    echo "  ✓ Restored numpy==1.26.4"
else
    echo "  ⚠ GamingAgent not found at $PARENT_DIR/GamingAgent"
    echo "    Clone it:  git clone https://github.com/lmgame-org/GamingAgent.git $PARENT_DIR/GamingAgent"
    echo "    Then run:  $PIP install -e $PARENT_DIR/GamingAgent && $PIP install numpy==1.26.4"
fi
echo

# ---------------------------------------------------------------------------
# Step 5: Check AgentEvolver
# ---------------------------------------------------------------------------
echo "[5/6] Checking AgentEvolver ..."
if [[ -d "$PARENT_DIR/AgentEvolver" ]]; then
    echo "  ✓ AgentEvolver found at $PARENT_DIR/AgentEvolver (added via PYTHONPATH, not pip)"
else
    echo "  ⚠ AgentEvolver not found at $PARENT_DIR/AgentEvolver"
    echo "    Clone it:  git clone https://github.com/modelscope/AgentEvolver.git $PARENT_DIR/AgentEvolver"
fi
echo

# ---------------------------------------------------------------------------
# Step 6: Verify installation
# ---------------------------------------------------------------------------
echo "[6/6] Verifying installation ..."
echo

PYTHONPATH="${REPO_DIR}:${PARENT_DIR}/AgentEvolver:${PARENT_DIR}/GamingAgent:${PYTHONPATH:-}" \
"$PYTHON" -c "
import sys

failures = []
warnings = []

def check(label, fn, required=True):
    try:
        fn()
        print(f'  [OK]   {label}')
    except Exception as e:
        if required:
            failures.append((label, str(e)))
            print(f'  [FAIL] {label}: {e}')
        else:
            warnings.append((label, str(e)))
            print(f'  [WARN] {label}: {e}  (optional)')

print(f'Python {sys.version}')
print()

# --- Core ML ---
print('Core ML:')
check('numpy',                 lambda: __import__('numpy'))
check('torch',                 lambda: __import__('torch'))
check('torch.cuda',            lambda: (t:=__import__('torch'), print(f'           CUDA available: {t.cuda.is_available()}, devices: {t.cuda.device_count()}')))
check('transformers',          lambda: __import__('transformers'))
check('peft',                  lambda: __import__('peft'))
check('safetensors',           lambda: __import__('safetensors'))
check('datasets',              lambda: __import__('datasets'))
check('accelerate',            lambda: __import__('accelerate'))
print()

# --- Inference ---
print('Inference:')
check('vllm',                  lambda: __import__('vllm'))
check('httpx',                 lambda: __import__('httpx'))
print()

# --- RAG ---
print('RAG:')
check('sentence_transformers', lambda: __import__('sentence_transformers'))
check('PIL (Pillow)',          lambda: __import__('PIL'))
print()

# --- API clients ---
print('API Clients:')
check('openai',                lambda: __import__('openai'))
check('anthropic',             lambda: __import__('anthropic'))
check('google.genai',          lambda: __import__('google.genai'))
print()

# --- Configuration ---
print('Configuration:')
check('omegaconf',             lambda: __import__('omegaconf'))
check('hydra',                 lambda: __import__('hydra'))
check('yaml (pyyaml)',         lambda: __import__('yaml'))
print()

# --- Skill bank ---
print('Skill Bank:')
check('sklearn',               lambda: __import__('sklearn'))
print()

# --- Game environments ---
print('Game Environments:')
check('gymnasium',             lambda: __import__('gymnasium'), required=False)
check('diplomacy',             lambda: __import__('diplomacy'), required=False)
check('games.games.avalon',    lambda: __import__('games.games.avalon.engine'), required=False)
print()

# --- Logging ---
print('Logging & Testing:')
check('loguru',                lambda: __import__('loguru'))
check('tensorboard',           lambda: __import__('tensorboard'))
check('pytest',                lambda: __import__('pytest'))
check('wandb',                 lambda: __import__('wandb'), required=False)
print()

# --- Internal modules ---
print('Internal Modules:')
check('trainer.coevolution.config',  lambda: __import__('trainer.coevolution.config'))
check('trainer.common.metrics',      lambda: __import__('trainer.common.metrics'))
check('skill_agents.grpo',          lambda: __import__('skill_agents.grpo'))
check('skill_agents.lora',          lambda: __import__('skill_agents.lora'))
check('rag.retrieval',              lambda: __import__('rag.retrieval'))
check('decision_agents',            lambda: __import__('decision_agents'))
check('data_structure',             lambda: __import__('data_structure'))
check('API_func',                   lambda: __import__('API_func'))
print()

# --- Summary ---
print('=' * 50)
if failures:
    print(f'{len(failures)} REQUIRED check(s) FAILED:')
    for label, err in failures:
        print(f'  ✗ {label}: {err}')
    sys.exit(1)
else:
    print('All required checks passed.')
if warnings:
    print(f'{len(warnings)} optional check(s) skipped (install sibling repos to fix):')
    for label, err in warnings:
        print(f'  ⚠ {label}')
print('=' * 50)
"

echo
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo
echo "  Activate:"
echo "    conda activate $ENV_NAME"
echo
echo "  Set PYTHONPATH (run from the parent directory of all repos):"
echo "    export PYTHONPATH=\$(pwd)/Game-AI-Agent:\$(pwd)/AgentEvolver:\$(pwd)/GamingAgent:\$PYTHONPATH"
echo
echo "  Set API keys:"
echo "    cp $REPO_DIR/.env.example $REPO_DIR/.env"
echo "    # Edit .env with your API keys"
echo "    set -a && source $REPO_DIR/.env && set +a"
echo
echo "  Quick smoke test:"
echo "    python -c \"from API_func import api_call; print('API_func OK')\""
echo "    pytest tests/ -q"
echo
echo "  For Super Mario, install the orak-mario env separately:"
echo "    bash $REPO_DIR/install/install_orak_mario.sh"
echo
echo "  Known nominal warning:"
echo "    gamingagent 0.1.0 requires numpy==1.24.4 (we use 1.26.4 — works fine)"
echo
echo "============================================================"
