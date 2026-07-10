#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PILOT_TAG="${PILOT_TAG:-pilot_20260710}"
RUN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${PILOT_TAG}"

squeue -u "${USER}" -o '%i|%P|%j|%T|%M|%R|%b' | {
  read -r header || true
  printf '%s\n' "${header}"
  rg 'vs-.*(cg_bench|video_holmes)' || true
}

for directory in "${RUN_ROOT}"/{cg_bench,video_holmes}/start_*; do
  [[ -d "${directory}" ]] || continue
  successful_calls=$(rg -c 'POST /v1/chat/completions HTTP/1.1" 200' "${directory}/transformers_server.log" 2>/dev/null || true)
  schema_rows=$(find "${directory}/stages" \
    \( -name '02_clip_schemas.jsonl' -o -name '00b_coarse_clip_schemas.jsonl' \) \
    -type f -print0 2>/dev/null | xargs -0 -r cat | wc -l)
  neighbor_rows=$(find "${directory}/stages" -name '03_neighbor_vlm_l1_clip_results.jsonl' \
    -type f -print0 2>/dev/null | xargs -0 -r cat | wc -l)
  examples=0
  [[ -f "${directory}/examples.jsonl" ]] && examples=$(wc -l < "${directory}/examples.jsonl")
  http_errors=$(rg -c 'HTTP/1.1" (4|5)' "${directory}/transformers_server.log" 2>/dev/null || true)
  printf '%s calls=%s schemas=%s neighbor_graph_rows=%s examples=%s http_errors=%s\n' \
    "${directory#${REPO_ROOT}/}" "${successful_calls:-0}" "${schema_rows}" "${neighbor_rows}" "${examples}" "${http_errors:-0}"
done
