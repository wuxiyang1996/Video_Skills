#!/bin/bash
#SBATCH --job-name=dispider-videomme
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/dispider-official-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/dispider-official-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-$PROJECT/code/Dispider}
VENV=${VENV:-$PROJECT/code/dispider_venv}
MODEL=${MODEL:-$PROJECT/data/models/dispider/Mar2Ding_Dispider}
QA_FILE=${QA_FILE:?Set QA_FILE to a manifest produced by baseline.prepare_dispider_videomme_manifest}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/atomic_skills_for_video/dispider_official_videomme/${SLURM_ARRAY_JOB_ID:-manual}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

mkdir -p "$OUT_ROOT" "$PROJECT/outputs/atomic_skills_for_video/logs"
cd "$REPO"

"$VENV/bin/python" ./dispider/eval/model_videomme_long.py \
  --model-path "$MODEL" \
  --chat_conversation_output_folder "$OUT_ROOT/${NUM_SHARDS}_${SHARD_INDEX}.json" \
  --Eval_QA_root "$QA_FILE" \
  --num-chunks "$NUM_SHARDS" \
  --chunk-idx "$SHARD_INDEX" \
  --temperature 0 \
  --num_beams 1 \
  --max_new_tokens 256 \
  --conv-mode qwen
