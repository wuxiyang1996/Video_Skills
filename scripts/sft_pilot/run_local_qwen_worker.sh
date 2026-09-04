#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-serve}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
GRAPH_MODEL="${GRAPH_MODEL:-openai/gpt-oss-120b:free}"
DATASET="${DATASET:?set DATASET to cg_bench or video_holmes}"
SPLIT="${SPLIT:-train}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-1}"
PILOT_TAG="${PILOT_TAG:-pilot_20260710}"
PORT="${PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 1000)))}"
CLIP_WORKERS="${CLIP_WORKERS:-2}"
GRAPH_WORKERS="${GRAPH_WORKERS:-2}"
SMOKE="${SMOKE:-0}"
QUERY_TIME_RETRIEVAL="${QUERY_TIME_RETRIEVAL:-1}"
CLIP_FRAMES="${CLIP_FRAMES:-4}"
# ``transformers serve`` runs without continuous batching, so concurrent clip
# workers queue behind one another and blow the per-request timeout: a pilot
# completed 7 of 98 clips in three hours, the rest timing out.  vLLM batches, and
# each clip is a ~4.5k-token prompt with a ~1k-token structured-JSON completion.
SERVE_BACKEND="${SERVE_BACKEND:-transformers}"
VLLM_VENV_ROOT="${VLLM_VENV_ROOT:-${REPO_ROOT}/.venv-qwen35-vllm}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-16}"
CLIP_MAX_TOKENS="${CLIP_MAX_TOKENS:-1600}"
CLIP_TIMEOUT_S="${CLIP_TIMEOUT_S:-120}"
LLM_COARSE_SELECTOR="${LLM_COARSE_SELECTOR:-1}"
CONTINUE_ON_ITEM_ERROR="${CONTINUE_ON_ITEM_ERROR:-1}"
RETRY_FAILED_CLIP_SCHEMAS="${RETRY_FAILED_CLIP_SCHEMAS:-1}"
MAX_INLINE_REPAIR_PASSES="${MAX_INLINE_REPAIR_PASSES:-2}"

[[ "${SPLIT}" == "train" || "${SPLIT}" == "test" ]] || {
  echo "SPLIT must be train or test: ${SPLIT}" >&2
  exit 2
}

export HF_HOME
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export VLLM_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false
# The serving venv intentionally inherits the Swift conda environment for
# PyTorch. Force stdlib distutils so setuptools does not assert when both
# environments expose different distutils implementations.
export SETUPTOOLS_USE_DISTUTILS=stdlib

if [[ "${SPLIT}" == "train" ]]; then
  RUN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${PILOT_TAG}/${DATASET}/start_${START_INDEX}_limit_${LIMIT}"
else
  RUN_ROOT="${REPO_ROOT}/dataset_clip_wrapper/output/${PILOT_TAG}/${DATASET}/${SPLIT}/start_${START_INDEX}_limit_${LIMIT}"
fi
mkdir -p "${RUN_ROOT}"
SERVER_LOG="${RUN_ROOT}/transformers_server.log"
RUN_LOG="${RUN_ROOT}/pipeline.log"

count_completed_examples() {
  if [[ -f "${RUN_ROOT}/examples.jsonl" ]]; then
    wc -l < "${RUN_ROOT}/examples.jsonl"
  else
    printf '0\n'
  fi
}

count_cached_schema_issues() {
  "${VENV_ROOT}/bin/python" - "${RUN_ROOT}/stages" <<'PY'
import sys
from pathlib import Path
from dataset_clip_wrapper.runners.run_staged_llm_pipeline import _cached_clip_schema_error_count

root = Path(sys.argv[1])
print(sum(_cached_clip_schema_error_count(path) for path in root.glob("*")))
PY
}

completed_examples="$(count_completed_examples)"
cached_schema_issues="$(count_cached_schema_issues)"

if [[ "${SMOKE}" != "1" && "${completed_examples}" -ge "${LIMIT}" && "${cached_schema_issues}" -eq 0 ]]; then
  printf '{"status":"already_complete","dataset":"%s","start_index":%s,"limit":%s,"examples":%s,"run_root":"%s"}\n' \
    "${DATASET}" "${START_INDEX}" "${LIMIT}" "${completed_examples}" "${RUN_ROOT}" | tee -a "${RUN_LOG}"
  exit 0
fi

printf '{"status":"starting","dataset":"%s","split":"%s","start_index":%s,"limit":%s,"examples":%s,"cached_schema_issues":%s,"retry_failed_clip_schemas":%s,"pilot_tag":"%s","clip_model":"%s","graph_model":"%s"}\n' \
  "${DATASET}" "${SPLIT}" "${START_INDEX}" "${LIMIT}" "${completed_examples}" "${cached_schema_issues}" "${RETRY_FAILED_CLIP_SCHEMAS}" "${PILOT_TAG}" "${MODEL}" "${GRAPH_MODEL}" | tee -a "${RUN_LOG}"

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
if [[ "${SERVE_BACKEND}" == "vllm" ]]; then
  "${VLLM_VENV_ROOT}/bin/vllm" serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --limit-mm-per-prompt "{\"image\": ${VLLM_LIMIT_MM_PER_PROMPT}}" \
    --disable-log-requests >"${SERVER_LOG}" 2>&1 &
else
  "${VENV_ROOT}/bin/transformers" serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --reasoning off \
    --attn-implementation sdpa >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!

SERVER_WAIT_TICKS="${SERVER_WAIT_TICKS:-$([[ "${SERVE_BACKEND}" == "vllm" ]] && echo 600 || echo 180)}"
for _ in $(seq 1 "${SERVER_WAIT_TICKS}"); do
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
    --split "${SPLIT}" \
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
  if [[ "${CONTINUE_ON_ITEM_ERROR}" == "1" ]]; then
    RETRIEVAL_ARGS+=(--continue-on-item-error)
  fi
  if [[ "${RETRY_FAILED_CLIP_SCHEMAS}" == "1" ]]; then
    RETRIEVAL_ARGS+=(--retry-failed-clip-schemas)
  fi
  run_staged_pipeline() {
    /fs/gamma-projects/vlm-robot/conda/bin/python -m dataset_clip_wrapper.run_staged_llm_pipeline \
      --dataset "${DATASET}" \
      --split "${SPLIT}" \
      --mode video_only \
      --unique-videos \
      --start-index "${START_INDEX}" \
      --limit "${LIMIT}" \
      --clip-schema-model "${MODEL}" \
      --clip-schema-api-base "${LOCAL_ENDPOINT}" \
      --clip-schema-frames "${CLIP_FRAMES}" \
      --clip-schema-max-tokens "${CLIP_MAX_TOKENS}" \
      --clip-schema-timeout-s "${CLIP_TIMEOUT_S}" \
      --clip-schema-workers "${CLIP_WORKERS}" \
      --graph-model "${GRAPH_MODEL}" \
      --graph-neighbor-workers "${GRAPH_WORKERS}" \
      --skill-model "${MODEL}" \
      --skill-api-base "${LOCAL_ENDPOINT}" \
      "${RETRIEVAL_ARGS[@]}" \
      --output "${RUN_ROOT}/examples.jsonl" \
      --stage-dir "${RUN_ROOT}/stages" 2>&1 | tee -a "${RUN_LOG}"
  }

  run_staged_pipeline
  if [[ "${RETRY_FAILED_CLIP_SCHEMAS}" == "1" ]]; then
    for repair_pass in $(seq 1 "${MAX_INLINE_REPAIR_PASSES}"); do
      completed_examples="$(count_completed_examples)"
      cached_schema_issues="$(count_cached_schema_issues)"
      if [[ "${completed_examples}" -ge "${LIMIT}" && "${cached_schema_issues}" -eq 0 ]]; then
        break
      fi
      printf '{"status":"inline_repair","pass":%s,"examples":%s,"limit":%s,"cached_schema_issues":%s}\n' \
        "${repair_pass}" "${completed_examples}" "${LIMIT}" "${cached_schema_issues}" | tee -a "${RUN_LOG}"
      run_staged_pipeline
    done
  fi
  completed_examples="$(count_completed_examples)"
  cached_schema_issues="$(count_cached_schema_issues)"
  if [[ "${completed_examples}" -lt "${LIMIT}" || "${cached_schema_issues}" -ne 0 ]]; then
    printf '{"status":"incomplete","examples":%s,"limit":%s,"cached_schema_issues":%s}\n' \
      "${completed_examples}" "${LIMIT}" "${cached_schema_issues}" | tee -a "${RUN_LOG}" >&2
    exit 3
  fi
fi
