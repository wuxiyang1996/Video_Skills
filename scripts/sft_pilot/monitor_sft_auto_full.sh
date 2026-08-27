#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PIPELINE_TAG="${PIPELINE_TAG:-sft_auto_20260713_full}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/dataset_clip_wrapper/output/sft_cold_start/$PIPELINE_TAG}"
BASE_DIR="${BASE_DIR:-$REPO_ROOT/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first}"
POLL_S="${POLL_S:-600}"

cd "$REPO_ROOT" || exit 1

while true; do
  ts="$(date -Is)"
  echo "===== $ts ====="

  echo "-- squeue --"
  squeue -u "$USER" -o "%.18i %.34j %.8T %.10M %.20R" | sed -n "1,80p" || true

  echo "-- repair counts --"
  total=0
  for lane in $(seq 0 19); do
    d="$BASE_DIR/${PIPELINE_TAG}_repair_lane${lane}"
    if [[ -d "$d" ]]; then
      n="$(find "$d/stages" -name repair_05_report.json 2>/dev/null | wc -l)"
      total=$((total + n))
      printf "lane%02d reports=%s\n" "$lane" "$n"
    fi
  done
  echo "total_repair_reports=$total"

  echo "-- pipeline tail --"
  tail -20 "$RUN_DIR/pipeline.log" 2>/dev/null || true

  echo "-- final outputs --"
  find "$RUN_DIR" -maxdepth 3 -type f \
    \( -name split_report.json -o -name snapshot_report.json -o -name motif_summary.json -o -name train_sft.jsonl -o -name dev_sft.jsonl \) \
    -printf "%p %s bytes\n" 2>/dev/null || true

  echo
  sleep "$POLL_S"
done
