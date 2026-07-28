#!/bin/bash
#SBATCH --job-name=m3-memorize
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --array=0-63%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-memorize-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-memorize-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/m3_memorization_venv}
BACKEND_ENV=${BACKEND_ENV:-$PROJECT/code/vllm_qwen_cu124_venv}
SHARD_ROOT=${SHARD_ROOT:-$PROJECT/outputs/atomic_skills_for_video/m3_agent_adapted/manifests/full_3bench_20260727/shards_64}
DATA_FILE=$(printf "%s/memorization_%04d.jsonl" "$SHARD_ROOT" "$SLURM_ARRAY_TASK_ID")

export HOME=${HOME_OVERRIDE:-$PROJECT/data/models/m3_runtime_home}
export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export M3_LOCAL_FILES_ONLY=1
export M3_LOCAL_EMBEDDING_MODEL=${M3_LOCAL_EMBEDDING_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
export M3_LOCAL_EMBEDDING_DEVICE=${M3_LOCAL_EMBEDDING_DEVICE:-cuda:0}
export M3_LOCAL_WHISPER_MODEL=${M3_LOCAL_WHISPER_MODEL:-openai/whisper-large-v3-turbo}
export M3_LOCAL_WHISPER_DEVICE=${M3_LOCAL_WHISPER_DEVICE:-cuda:0}
BACKEND_PORT=$((18000 + SLURM_ARRAY_TASK_ID))
export M3_LOCAL_BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"

mkdir -p "$HOME" "$TRANSFORMERS_CACHE"
cd "$ATOMIC_REPO"
"$BACKEND_ENV/bin/python" -m baseline.m3_local_backend_server --port "$BACKEND_PORT" &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if "$BACKEND_ENV/bin/python" -c "import urllib.request; urllib.request.urlopen('$M3_LOCAL_BACKEND_URL/health', timeout=2)" 2>/dev/null; then
    break
  fi
  sleep 2
done
"$VENV/bin/python" -m baseline.run_m3_agent_snapshot_memorization \
  --data-file "$DATA_FILE" \
  --fps 1 \
  --max-new-tokens 1024 \
  --attention-implementation flash_attention_2 \
  --disable-face \
  --disable-speaker
