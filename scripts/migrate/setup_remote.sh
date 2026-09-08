#!/usr/bin/env bash
# One-time environment setup on the rented machine (Ubuntu + CUDA 12.x driver, python3.10 available). Run inside $WORK.
set -euo pipefail; WORK=${WORK:-/workspace}; cd $WORK/Video_Skills
export HF_HOME=$WORK/hf_cache
python3.10 -m venv $WORK/venv-serve && $WORK/venv-serve/bin/pip install -q -U pip
# serving + evaluation + training in one env: vllm (CUDA 12), transformers, peft, fla, opencv, av, openai, requests
$WORK/venv-serve/bin/pip install -q "vllm>=0.10" "transformers>=4.57" peft accelerate safetensors flash-linear-attention opencv-python-headless av openai requests numpy
# flash-attn (optional, faster training); causal-conv1d prebuilt wheel if it matches, else the torch shim below
$WORK/venv-serve/bin/pip install -q flash-attn --no-build-isolation || echo "flash-attn skipped (sdpa fallback)"
$WORK/venv-serve/bin/pip install -q causal-conv1d --no-build-isolation || { SP=$($WORK/venv-serve/bin/python -c "import site;print(site.getsitepackages()[0])"); mkdir -p $SP/causal_conv1d $SP/causal_conv1d-1.5.2.dist-info; cp /fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/lib/python3.10/site-packages/causal_conv1d/__init__.py $SP/causal_conv1d/ 2>/dev/null || true; printf 'Metadata-Version: 2.1\nName: causal-conv1d\nVersion: 1.5.2\n' > $SP/causal_conv1d-1.5.2.dist-info/METADATA; echo "causal_conv1d torch shim installed"; }
$WORK/venv-serve/bin/python -c "import vllm, transformers, peft, fla; print('ok', vllm.__version__, transformers.__version__)"
echo "set: export HF_HOME=$WORK/hf_cache; datasets root -> $WORK/datasets (pass --dataset-root / edit DATASET_ROOT in trainer/reader/rewards.py); keys -> $WORK/keys.py"
