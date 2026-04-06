# Installation Guide

COS-PLAY uses **two conda environments**:

| Environment | Purpose | Python | GPU | Install Time |
|---|---|---|---|---|
| **`game-ai-agent`** | Training, inference, baselines, evaluation (all games except Super Mario) | 3.11 | CUDA 12.x | ~10 min |
| **`orak-mario`** | Super Mario Bros evaluation (nes-py requires numpy<2) | 3.11 | CUDA 12.x | ~5 min |

## Prerequisites

- **OS:** Linux (Ubuntu 20.04+ recommended). macOS for development only (no vLLM / FSDP).
- **GPU:** 8× A100-80GB recommended for full co-evolution training. 1× GPU sufficient for inference/baselines.
- **CUDA:** 12.x drivers installed on the host.
- **Conda:** Miniconda3 or Anaconda.

```bash
# Install Miniconda if not present
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init
```

## Quick Start

```bash
# 1. Clone all repos into the same parent directory
mkdir -p ~/cos-play && cd ~/cos-play

git clone https://github.com/<your-org>/Game-AI-Agent.git
git clone https://github.com/lmgame-org/GamingAgent.git
git clone https://github.com/modelscope/AgentEvolver.git
git clone https://github.com/nicholascpark/orak.git Orak   # for Super Mario

# 2. Install the main environment
bash Game-AI-Agent/install/install_main_env.sh

# 3. Install the orak-mario environment (if running Super Mario)
bash Game-AI-Agent/install/install_orak_mario.sh

# 4. Configure API keys
cp Game-AI-Agent/.env.example Game-AI-Agent/.env
# Edit .env with your keys (OpenAI, Anthropic, Google, OpenRouter)
```

---

## 1. Main Environment (`game-ai-agent`)

### What it covers

- Co-evolution training (GRPO + FSDP + LoRA on Qwen3-8B)
- Skill bank pipeline (boundary proposal, segmentation, contracts, curation)
- RAG retrieval (Qwen3-Embedding-0.6B)
- Cold-start data generation and labeling
- vLLM inference server
- Baselines evaluation (GPT-5.4, Claude-4.6, Gemini-3.1-Pro, GPT-OSS-120B)
- Games: 2048, Candy Crush, Tetris, Avalon, Diplomacy

### Install

```bash
cd ~/cos-play
bash Game-AI-Agent/install/install_main_env.sh
```

The script:
1. Creates the `game-ai-agent` conda env with Python 3.11
2. Installs PyTorch with CUDA 12.x
3. Installs all pip dependencies from `install/requirements.txt`
4. Installs GamingAgent in editable mode (if cloned)
5. Checks for AgentEvolver
6. Runs import verification (~30 checks)

### Activate

```bash
conda activate game-ai-agent
export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

### Set API keys

```bash
cp Game-AI-Agent/.env.example Game-AI-Agent/.env
# Edit .env — at minimum set OPENROUTER_API_KEY for baselines
set -a && source Game-AI-Agent/.env && set +a
```

### Verify

```bash
python -c "from API_func import api_call; print('OK')"
pytest Game-AI-Agent/tests/ -q
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | ≥2.1 | GRPO, FSDP, LoRA training |
| transformers | ≥4.51.0 | Qwen3-8B backbone |
| peft | ≥0.10 | LoRA adapter loading / merging |
| vllm | ≥0.4 | Fast inference server |
| sentence-transformers | ≥2.7.0 | RAG text embedder |
| openai | ≥1.0 | OpenRouter API calls |
| anthropic | ≥0.30 | Claude baselines |
| google-genai | ≥1.0 | Gemini baselines |
| numpy | ==1.26.4 | Pinned for GamingAgent compatibility |
| diplomacy | ≥1.1.2 | Diplomacy game engine |
| scikit-learn | ≥1.0 | Skill bank clustering |

Full list: [`install/requirements.txt`](requirements.txt)

---

## 2. Super Mario Environment (`orak-mario`)

### Why a separate environment?

`nes-py` (the NES emulator) requires **numpy<2** and **gym==0.26.2**, which conflict with the main environment's dependencies (vLLM, transformers, etc.).

### Install

```bash
cd ~/cos-play
bash Game-AI-Agent/install/install_orak_mario.sh
```

The script:
1. Creates the `orak-mario` conda env with Python 3.11
2. Installs PyTorch + torchvision with CUDA 12.x
3. Installs all pip dependencies from `install/requirements-orak-mario.txt`
4. Runs import verification

### Activate

Use the setup script (also sets `PYTHONPATH` and starts Xvfb for headless rendering):

```bash
source Game-AI-Agent/env_wrappers/setup_orak_mario.sh
```

Or activate manually:
```bash
conda activate orak-mario
export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/Orak/src:$PYTHONPATH
```

### Headless servers (no display)

`nes-py` / `pyglet` require a display. On headless machines, the setup script starts Xvfb automatically. If you see display errors:

```bash
sudo apt install -y xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

### Verify

```bash
python -c "import gym_super_mario_bros; print('Mario env OK')"
python -c "import nes_py; import numpy; print(f'numpy {numpy.__version__}')"
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| nes-py | ==8.2.1 | NES emulator |
| gym-super-mario-bros | ==7.4.0 | Mario environment |
| gym | ==0.26.2 | OpenAI Gym (nes-py compat) |
| numpy | <2 | nes-py incompatible with NumPy 2.x |
| torch + torchvision | latest | Orak object detection |
| opencv-python-headless | latest | Vision processing |
| openai | ≥1.0 | LLM baselines via OpenRouter |

Full list: [`install/requirements-orak-mario.txt`](requirements-orak-mario.txt)

---

## Troubleshooting

### `conda: command not found`

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
# Or pass the path directly:
bash Game-AI-Agent/install/install_main_env.sh $HOME/miniconda3/bin/conda
```

### `gamingagent requires numpy==1.24.4`

This is a known nominal warning. GamingAgent pins numpy==1.24.4 but works correctly with 1.26.4. The install script restores 1.26.4 after the editable install.

### `ModuleNotFoundError: No module named 'games'`

AgentEvolver is loaded via `PYTHONPATH`, not pip. Make sure it's set:

```bash
export PYTHONPATH=$(pwd)/Game-AI-Agent:$(pwd)/AgentEvolver:$(pwd)/GamingAgent:$PYTHONPATH
```

### `vllm` installation fails

vLLM requires CUDA 12.x. On machines without GPU:

```bash
# CPU-only (inference via API only, no local vLLM):
pip install -r install/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Comment out vllm in requirements.txt if you only need API baselines
```

### `nes_py` or `pyglet` display errors (Super Mario)

```bash
sudo apt install -y xvfb
# The setup script (env_wrappers/setup_orak_mario.sh) starts Xvfb automatically.
```

### CUDA version mismatch

The install scripts default to CUDA 12.4 wheels. If you need a different CUDA version:

```bash
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## File Structure

```
install/
├── README.md                       # This file
├── install_main_env.sh             # Main environment installer
├── install_orak_mario.sh           # Super Mario environment installer
├── requirements.txt                # Main env pip dependencies
└── requirements-orak-mario.txt     # orak-mario pip dependencies
```

## Environment Summary

```
┌──────────────────────────────────────────────────────────┐
│  game-ai-agent  (main)                                   │
│  ├── Training: GRPO, FSDP, SFT, co-evolution            │
│  ├── Inference: vLLM server                              │
│  ├── Games: 2048, Candy Crush, Tetris, Avalon, Diplomacy │
│  └── Baselines: GPT-5.4, Claude-4.6, Gemini-3.1-Pro     │
├──────────────────────────────────────────────────────────┤
│  orak-mario  (separate — numpy<2)                        │
│  ├── Game: Super Mario Bros                              │
│  └── Reason: nes-py requires numpy<2, gym==0.26.2        │
└──────────────────────────────────────────────────────────┘
```
