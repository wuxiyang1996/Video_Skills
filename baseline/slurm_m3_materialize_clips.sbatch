#!/bin/bash
#SBATCH --job-name=m3-clips
#SBATCH --partition=prod
#SBATCH --array=0-63%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-clips-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/m3-clips-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
SHARD_ROOT=${SHARD_ROOT:-$PROJECT/outputs/atomic_skills_for_video/m3_agent_adapted/manifests/full_3bench_20260727/shards_64}
FFMPEG=${FFMPEG:-$PROJECT/code/dispider_venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
DATA_FILE=$(printf "%s/memorization_%04d.jsonl" "$SHARD_ROOT" "$SLURM_ARRAY_TASK_ID")

cd "$ATOMIC_REPO"
"$VENV/bin/python" -m baseline.materialize_m3_agent_clips \
  --clip-plan "$DATA_FILE" \
  --ffmpeg-bin "$FFMPEG"
