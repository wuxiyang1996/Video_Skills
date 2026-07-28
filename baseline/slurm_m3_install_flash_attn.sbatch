#!/bin/bash
#SBATCH --job-name=m3-flash
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-flash-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-flash-%j.err

set -euo pipefail

VENV=${VENV:-/mnt/is_data/xwu/video_skills/code/m3_memorization_venv}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.7.4.post1}
BUILD_TMPDIR=/scratch/xwu/m3-flash-${SLURM_JOB_ID}
export TMPDIR=$BUILD_TMPDIR
export MAX_JOBS=${MAX_JOBS:-2}
export PIP_CACHE_DIR=$BUILD_TMPDIR/pip-cache
mkdir -p "$BUILD_TMPDIR"
trap 'rm -rf "$BUILD_TMPDIR"' EXIT

"$VENV/bin/python" -m pip install wheel
"$VENV/bin/python" -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation
"$VENV/bin/python" - <<'PY'
import flash_attn
import torch
print({"flash_attn": flash_attn.__version__, "torch": torch.__version__, "cuda": torch.version.cuda})
PY
