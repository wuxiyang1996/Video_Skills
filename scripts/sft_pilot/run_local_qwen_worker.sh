#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
GRAPH_MODEL="${GRAPH_MODEL:-openai/gpt-oss-120b:free}"
DATASET="${DATASET:?set DATASET to cg_bench or video_holmes}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-1}"
PILOT_TAG="${PILOT_TAG:-pilot_20260710}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 1000)))}"
CLIP_WORKERS="${CLIP_WORKERS:-2}"
GRAPH_WORKERS="${GRAPH_WORKERS:-2}"
SMOKE="${SMOKE:-0}"
QUERY_TIME_RETRIEVAL="${QUERY_TIME_RETRIEVAL:-1}"
CLIP_FRAMES="${CLIP_FRAMES:-4}"
CLIP_MAX_TOKENS="${CLIP_MAX_TOKENS:-1600}"
LLM_COARSE_SELECTOR="${LLM_COARSE_SELECTOR:-1}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false

RUN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${PILOT_TAG}/${DATASET}/start_${START_INDEX}_limit_${LIMIT}"
mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/transformers_server.log"
RUN_LOG="${RUN_ROOT}/pipeline.log"

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
if [[ "${SMOKE}" == "1" ]]; then
  /fs/gamma-projects/vlm-robot/conda/bin/python -m dataset_clip_wrapper.run_llm_pipeline \
    --dataset "${DATASET}" \
    --split train \
    --mode video_only \
    --limit 1 \
    --clip-schema-model "${MODEL}" \
    --clip-schema-api-base "${LOCAL_ENDPOINT}" \
    --clip-schema-max-clips 1 \
    --clip-schema-frames 2 \
    --graph-model "${GRAPH_MODEL}" \
    --graph-neighbor-workers 1 \
    --output "${RUN_ROOT}/smoke.jsonl" 2>&1 | tee "${RUN_LOG}"
else
  RETRIEVAL_ARGS=()
  if [[ "${QUERY_TIME_RETRIEVAL}" == "1" ]]; then
    RETRIEVAL_ARGS+=(--query-time-retrieval)
  fi
  if [[ "${LLM_COARSE_SELECTOR}" == "1" ]]; then
    RETRIEVAL_ARGS+=(--llm-coarse-selector)
  fi
  /fs/gamma-projects/vlm-robot/conda/bin/python -m dataset_clip_wrapper.run_staged_llm_pipeline \
    --dataset "${DATASET}" \
    --split train \
    --mode video_only \
    --unique-videos \
    --start-index "${START_INDEX}" \
    --limit "${LIMIT}" \
    --clip-schema-model "${MODEL}" \
    --clip-schema-api-base "${LOCAL_ENDPOINT}" \
    --clip-schema-frames "${CLIP_FRAMES}" \
    --clip-schema-max-tokens "${CLIP_MAX_TOKENS}" \
    --clip-schema-workers "${CLIP_WORKERS}" \
    --graph-model "${GRAPH_MODEL}" \
    --graph-neighbor-workers "${GRAPH_WORKERS}" \
    "${RETRIEVAL_ARGS[@]}" \
    --output "${RUN_ROOT}/examples.jsonl" \
    --stage-dir "${RUN_ROOT}/stages" 2>&1 | tee "${RUN_LOG}"
fi
