#!/bin/bash
#SBATCH --job-name=m3-control
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --array=0-2%1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-control-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-control-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/m3_control_venv}
BACKEND_ENV=${BACKEND_ENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MANIFEST_ROOT=${MANIFEST_ROOT:-$PROJECT/outputs/atomic_skills_for_video/m3_agent_adapted/manifests/full_3bench_20260727}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/atomic_skills_for_video/m3_agent_adapted/control/full_3bench_20260727}
DATASETS=(ovo_bench videomme streaming_bench)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export HOME=${HOME_OVERRIDE:-$PROJECT/data/models/m3_runtime_home}
export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export M3_LOCAL_FILES_ONLY=1
export M3_LOCAL_EMBEDDING_MODEL=${M3_LOCAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
export M3_LOCAL_EMBEDDING_DEVICE=${M3_LOCAL_EMBEDDING_DEVICE:-cuda:2}
export M3_LOCAL_BACKEND_URL=http://127.0.0.1:18765

mkdir -p "$HOME" "$TRANSFORMERS_CACHE" "$OUT_ROOT/$DATASET"
cd "$ATOMIC_REPO"
"$BACKEND_ENV/bin/python" -m baseline.m3_local_backend_server --port 18765 &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if "$BACKEND_ENV/bin/python" -c "import urllib.request; urllib.request.urlopen('$M3_LOCAL_BACKEND_URL/health', timeout=2)" 2>/dev/null; then
    break
  fi
  sleep 2
done
"$VENV/bin/python" -m baseline.m3_agent_adapted_control \
  --data-file "$MANIFEST_ROOT/control_${DATASET}.json" \
  --output-dir "$OUT_ROOT/$DATASET" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.82
