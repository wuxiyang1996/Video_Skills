#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
RECOVERY_ROOT="${RECOVERY_ROOT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/sft_recovery_20260720_full}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${RECOVERY_ROOT}/snapshot}"
OUTPUT_DIR="${OUTPUT_DIR:-${RECOVERY_ROOT}/sft_v2}"
TARGET_TOTAL="${TARGET_TOTAL:-200}"
DEV_PERCENT="${DEV_PERCENT:-10}"
MAX_CHARACTERS="${MAX_CHARACTERS:-48000}"
MIN_L2_RETRIEVAL="${MIN_L2_RETRIEVAL:-14}"
SALT="${SALT:-video-skills-sft-v2}"

cd "${REPO_ROOT}"
/fs/gamma-projects/vlm-robot/conda/bin/python -m dataset_clip_wrapper.training.build_sft_splits \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dev-percent "${DEV_PERCENT}" \
  --salt "${SALT}" \
  --target-total "${TARGET_TOTAL}" \
  --l1-percent 35 \
  --l2-percent 35 \
  --verifier-percent 20 \
  --motif-percent 10 \
  --example-video-map "${RECOVERY_ROOT}/example_video_map.json" \
  --exclude-dataset vrbench \
  --exclude-dataset videomme \
  --exclude-dataset ovo_bench \
  --controller-minimum "l2_retrieval=${MIN_L2_RETRIEVAL}" \
  --max-characters "${MAX_CHARACTERS}" \
  --strict

echo "SFT v2 data ready: ${OUTPUT_DIR}"
