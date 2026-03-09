# Installation

This document covers installing Game-AI-Agent and (optionally) VERL/verl-agent for training and inference.

## Requirements

- **Python**: 3.9+
- **CUDA**: 12.1+ (for VERL training/inference with vLLM/sglang)

## 1. Clone and set up Game-AI-Agent

This repo is used as a library by adding it to `PYTHONPATH` (no `setup.py`). From the repository root or parent workspace:

```bash
cd /path/to/Game-AI-Agent
export PYTHONPATH="$(pwd):$PYTHONPATH"
# Or on Windows: set PYTHONPATH=%CD%;%PYTHONPATH%
```

When using the **conda environment** (section 4), activate it first; the scripts assume the repo root is on `PYTHONPATH` (e.g. run commands from `Game-AI-Agent` or set `PYTHONPATH` to include it).

## 2. Optional dependencies

- **RAG (embeddings)**: `pip install -r rag/requirements.txt`  
  See [rag/README.md](rag/README.md) for text and multimodal embedding setup.

- **Skill agents / boundary proposal**: `pip install -r skill_agents/boundary_proposal/requirements.txt`  
  **Infer segmentation**: `pip install -r skill_agents/infer_segmentation/requirements.txt`

- **Game environments**: Install per environment (Overcooked, AgentEvolver, GamingAgent, VideoGameBench).  
  See [evaluate_overcooked](evaluate_overcooked/), [evaluation_evolver/setup_evolver_eval_env.md](evaluation_evolver/setup_evolver_eval_env.md), [evaluate_gamingagent/setup_gamingagent_eval_env.md](evaluate_gamingagent/setup_gamingagent_eval_env.md), [evaluate_videogamebench/setup_videogamebench_eval_env.md](evaluate_videogamebench/setup_videogamebench_eval_env.md).

## 3. VERL and verl-agent (training and VERL inference)

For **VERL-based training** and **VERL-based inference** (vLLM/sglang, Ray, FSDP), install [VERL](https://github.com/verl-project/verl) via [verl-agent](https://github.com/verl-project/verl-agent).

### 3.1 Clone verl-agent

Clone verl-agent as a **sibling** of Game-AI-Agent so both are on the same parent path (e.g. `ICML2026/Game-AI-Agent` and `ICML2026/verl-agent`):

```bash
cd /path/to/parent   # e.g. ICML2026
git clone https://github.com/verl-project/verl-agent.git
cd verl-agent
```

### 3.2 Install verl-agent

Follow verl-agent’s install instructions (Python 3.9+, CUDA 12.1+). Typical install:

```bash
pip install -e .          # core
# Or with vLLM/SGLang (see verl-agent docs):
# pip install -e .[vllm]
# pip install -e .[sglang]
```

Install VERL/vLLM/SGLang as required by verl-agent (see [verl-agent docs](https://github.com/verl-project/verl-agent) and [VERL install](https://verl.readthedocs.io/en/latest/start/install.html)).

### 3.3 Run training and inference with VERL

From the **parent** of both repos (or with `PYTHONPATH` set so both are importable):

**Training:**

```bash
cd /path/to/Game-AI-Agent
python -m scripts.run_trainer --verl
python -m scripts.run_trainer --verl algorithm.adv_estimator=gigpo trainer.nnodes=2
```

**Inference (eval-only):**

```bash
python -m scripts.run_inference --verl
python -m inference.run_verl_inference data.val_batch_size=8
```

The scripts set `PYTHONPATH` to include both Game-AI-Agent and verl-agent when invoking `verl.trainer.main_gameai`.

## 4. Conda environment

Use the provided conda environment file for a reproducible environment:

```bash
conda env create -f environment.yml
conda activate game-ai-agent
```

Then install VERL/verl-agent as in section 3 if you need VERL training/inference.  
See [environment.yml](environment.yml) for the list of channels and base dependencies.

## 5. Standalone (no VERL)

You can run **standalone training** (in-repo GRPO) and **local inference** without verl-agent:

- **Training**: `python -m scripts.run_trainer --config trainer/common/configs/decision_grpo.yaml`
- **Inference**: `python -m scripts.run_inference --game <game> --task "..."`  
  (requires a game env; see [inference/README.md](inference/README.md))

No Ray/vLLM/SGLang or verl-agent install is required for this path.
