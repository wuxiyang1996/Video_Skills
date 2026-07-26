#!/usr/bin/env bash
# Install FlashAttention-2 into the Qwen training venv.
# Prefer Dao-AILab prebuilt wheels matching torch/python/cxx11 ABI.
# Fallback: source build with CUDA_HOME from video-r1 toolkit (has nvcc).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
# Prefer dedicated GRPO conda env when present.
DEFAULT_CONDA_ENV="/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo"
if [[ -n "${VENV_ROOT:-}" ]]; then
  :
elif [[ -x "${DEFAULT_CONDA_ENV}/bin/python" ]]; then
  VENV_ROOT="${DEFAULT_CONDA_ENV}"
else
  VENV_ROOT="${REPO_ROOT}/.venv-qwen35-serve"
fi
PYTHON="${VENV_ROOT}/bin/python"
FA_VERSION="${FA_VERSION:-2.7.4.post1}"
CUDA_HOME_CANDIDATE="${CUDA_HOME_CANDIDATE:-/fs/gamma-projects/vlm-robot/conda/envs/video-r1}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing venv python: ${PYTHON}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
hostname
nvidia-smi || true

"${PYTHON}" - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
})
PY

if "${PYTHON}" -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null; then
  echo "flash_attn already installed"
  "${PYTHON}" - <<'PY'
from trainer.grpo.attn_utils import resolve_attn_implementation
print({"attn": resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False)})
PY
  exit 0
fi

echo "Installing flash-attn ${FA_VERSION} into ${VENV_ROOT}"
"${PYTHON}" -m pip install -U pip wheel ninja packaging

WHEEL_URL="$("${PYTHON}" - <<PY
import sys
import torch
fa = "${FA_VERSION}"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
# torch.__version__ like 2.6.0+cu124
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
name = f"flash_attn-{fa}+cu12torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{fa}/{name}")
PY
)"

echo "Trying prebuilt wheel: ${WHEEL_URL}"
if "${PYTHON}" -m pip install "${WHEEL_URL}"; then
  echo "Installed flash-attn from prebuilt wheel"
else
  echo "Wheel install failed; falling back to source build" >&2
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -x "${CUDA_HOME_CANDIDATE}/bin/nvcc" ]]; then
      export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      export CUDA_HOME=/usr/local/cuda
    else
      echo "CUDA_HOME unset and no nvcc candidate found" >&2
      exit 2
    fi
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export MAX_JOBS="${MAX_JOBS:-4}"
  echo "CUDA_HOME=${CUDA_HOME}"
  "${CUDA_HOME}/bin/nvcc" --version | head -5 || true
  FLASH_ATTENTION_FORCE_BUILD=TRUE "${PYTHON}" -m pip install "flash-attn==${FA_VERSION}" --no-build-isolation
fi

"${PYTHON}" - <<'PY'
import flash_attn
from trainer.grpo.attn_utils import resolve_attn_implementation
print({
    "flash_attn": flash_attn.__version__,
    "attn": resolve_attn_implementation("flash_attention_2", allow_sdpa_fallback=False),
})
PY
