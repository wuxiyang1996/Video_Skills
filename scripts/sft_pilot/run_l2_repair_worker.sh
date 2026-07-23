#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-120b:free}"
DATASETS="${DATASETS:-cg_bench}"
QUALITY_REPORT="${QUALITY_REPORT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/free_pilot_quality_report.json}"
EXAMPLE_IDS_FILE="${EXAMPLE_IDS_FILE:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/l2_repair_candidate_ids.txt}"
REPAIR_TAG="${REPAIR_TAG:-l2_repair_20260713_free}"
MAX_EXAMPLES="${MAX_EXAMPLES:-9}"
EXAMPLE_OFFSET="${EXAMPLE_OFFSET:-0}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-6}"
REPAIR_RETRY_SLEEP_S="${REPAIR_RETRY_SLEEP_S:-900}"
PORT="${PORT:-$((19000 + (${SLURM_JOB_ID:-0} % 1000)))}"
REQUEST_FRAMES="${REQUEST_FRAMES:-4}"
MAX_REPAIR_CLIPS="${MAX_REPAIR_CLIPS:-4}"
REPAIR_CLIP_SCHEMA_WORKERS="${REPAIR_CLIP_SCHEMA_WORKERS:-1}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false
export SETUPTOOLS_USE_DISTUTILS=stdlib

RUN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/${REPAIR_TAG}"
STAGE_DIR="${RUN_ROOT}/stages"
mkdir -p "${RUN_ROOT}" "${STAGE_DIR}"
SERVER_LOG="${RUN_ROOT}/transformers_server.log"
RUN_LOG="${RUN_ROOT}/repair.log"
OUTPUT="${RUN_ROOT}/repair_report.json"

mapfile -t EXAMPLE_IDS < <(grep -v '^[[:space:]]*$' "${EXAMPLE_IDS_FILE}" | tail -n +"$((EXAMPLE_OFFSET + 1))" | head -n "${MAX_EXAMPLES}")
if [[ "${#EXAMPLE_IDS[@]}" -eq 0 ]]; then
  echo "No example ids found in ${EXAMPLE_IDS_FILE}" | tee -a "${RUN_LOG}"
  exit 1
fi

printf '{"status":"starting_l2_repair","examples":%s,"example_offset":%s,"clip_model":"%s","gptoss_model":"%s","output":"%s"}\n' \
  "${#EXAMPLE_IDS[@]}" "${EXAMPLE_OFFSET}" "${MODEL}" "${GPTOSS_MODEL}" "${OUTPUT}" | tee -a "${RUN_LOG}"

cleanup() {
  trap - EXIT INT TERM
  local child_pids
  child_pids="$(pgrep -P "$$" 2>/dev/null || true)"
  if [[ -n "${child_pids}" ]]; then
    kill ${child_pids} 2>/dev/null || true
    sleep 1
    kill -9 ${child_pids} 2>/dev/null || true
    wait ${child_pids} 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
"${VENV_ROOT}/bin/transformers" serve "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --reasoning off \
  --attn-implementation sdpa >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -100 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

LOCAL_ENDPOINT="http://127.0.0.1:${PORT}/v1/chat/completions"
DATASETS_SPACED="${DATASETS//,/ }"
read -r -a DATASET_ARGS <<< "${DATASETS_SPACED}"
attempt=1
while true; do
  set +e
  /fs/gamma-projects/vlm-robot/conda/bin/python -m dataset_clip_wrapper.run_repair_protocol \
    --quality-report "${QUALITY_REPORT}" \
    --stage-dir "${STAGE_DIR}" \
    --output "${OUTPUT}" \
    --datasets "${DATASET_ARGS[@]}" \
    --example-ids "${EXAMPLE_IDS[@]}" \
    --repair-mode reroute \
    --keys-py /fs/gamma-projects/vlm-robot/keys.py \
    --clip-schema-model "${MODEL}" \
    --clip-schema-api-base "${LOCAL_ENDPOINT}" \
    --verifier-model "${GPTOSS_MODEL}" \
    --clue-planner-model "${GPTOSS_MODEL}" \
    --bridge-model "${GPTOSS_MODEL}" \
    --allow-lexical-fallback \
    --request-frames "${REQUEST_FRAMES}" \
    --max-repair-clips "${MAX_REPAIR_CLIPS}" \
    --repair-clip-schema-workers "${REPAIR_CLIP_SCHEMA_WORKERS}" \
    --clip-schema-reasoning-effort none 2>&1 | tee -a "${RUN_LOG}"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    break
  fi
  if [[ "${attempt}" -ge "${REPAIR_ATTEMPTS}" ]] || ! tail -200 "${RUN_LOG}" | grep -Eiq '429 (client )?error|temporarily rate-limited|too many requests'; then
    exit "${rc}"
  fi
  printf '{"status":"retrying_after_free_api_rate_limit","attempt":%s,"sleep_s":%s}\n' \
    "${attempt}" "${REPAIR_RETRY_SLEEP_S}" | tee -a "${RUN_LOG}"
  sleep "${REPAIR_RETRY_SLEEP_S}"
  attempt=$((attempt + 1))
done
