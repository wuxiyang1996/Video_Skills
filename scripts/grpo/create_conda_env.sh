#!/usr/bin/env bash
# Create a dedicated conda env for Video Skills GRPO / LoRA.
# Stack: Python 3.10 + torch2.6/cu124 + FlashAttention-2 + HF/PEFT
# Does NOT install verl / ms-swift / vllm (add later only if needed).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
CONDA_ROOT="${CONDA_ROOT:-/fs/gamma-projects/vlm-robot/conda}"
ENV_NAME="${ENV_NAME:-video-skills-grpo}"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_ROOT}/envs/${ENV_NAME}}"
FA_VERSION="${FA_VERSION:-2.7.4.post1}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
CUDA_TAG="${CUDA_TAG:-cu124}"

export PATH="${CONDA_ROOT}/bin:${PATH}"
cd "${REPO_ROOT}"

echo "Creating conda env at ${ENV_PREFIX}"
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Env already exists: ${ENV_PREFIX}"
else
  conda create -y -p "${ENV_PREFIX}" python=3.10 pip setuptools wheel
fi

PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"

"${PIP}" install -U pip packaging ninja
"${PIP}" install \
  "torch==${TORCH_VERSION}" torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Qwen3.5 needs transformers>=5.x (model_type qwen3_5); pin to serve-proven 5.13.0.
"${PIP}" install \
  "transformers==5.13.0" \
  "tokenizers>=0.22.0,<0.23" \
  "peft>=0.14.0,<0.21" \
  "accelerate>=1.0.0" \
  "datasets>=3.0.0" \
  "sentencepiece" \
  "protobuf" \
  "einops" \
  "safetensors" \
  "liger-kernel" \
  "numpy" \
  "requests" \
  "tqdm"

# FlashAttention-2 prebuilt wheel matching torch/python/cxx11 ABI.
WHEEL_URL="$("${PY}" - <<PY
import sys
import torch
fa = "${FA_VERSION}"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
name = f"flash_attn-{fa}+cu12torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{fa}/{name}")
PY
)"
echo "Installing flash-attn wheel: ${WHEEL_URL}"
"${PIP}" install "${WHEEL_URL}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
"${PY}" - <<'PY'
import torch
import transformers
import peft
import flash_attn
from trainer.grpo.attn_utils import resolve_attn_implementation

print({
    "python": __import__("sys").version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "flash_attn": flash_attn.__version__,
    "attn": resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False),
    "env": "video-skills-grpo",
    "verl": False,
    "ms_swift": False,
    "vllm": False,
})
PY

echo "Done. Activate with:"
echo "  source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_PREFIX}"
echo "Or set: GRPO_PYTHON=${ENV_PREFIX}/bin/python"
