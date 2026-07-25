#!/usr/bin/env bash
# Install FlashAttention-2 into the Qwen training venv (must run on a GPU node).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
PYTHON="${VENV_ROOT}/bin/python"

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
print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
PY

# Match torch 2.6 + cu124 wheel when available; otherwise build from source.
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-FALSE}"

if "${PYTHON}" -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null; then
  echo "flash_attn already installed"
  exit 0
fi

echo "Installing flash-attn into ${VENV_ROOT}"
"${PYTHON}" -m pip install -U pip wheel ninja packaging
# Prefer prebuilt wheel; fall back to source build on the GPU node.
if ! "${PYTHON}" -m pip install "flash-attn==2.7.4.post1" --no-build-isolation; then
  echo "wheel install failed; building flash-attn from source (slow)" >&2
  FLASH_ATTENTION_FORCE_BUILD=TRUE "${PYTHON}" -m pip install "flash-attn==2.7.4.post1" --no-build-isolation
fi

"${PYTHON}" - <<'PY'
import flash_attn
from trainer.grpo.attn_utils import resolve_attn_implementation
print({"flash_attn": flash_attn.__version__, "attn": resolve_attn_implementation("flash_attention_2")})
PY
