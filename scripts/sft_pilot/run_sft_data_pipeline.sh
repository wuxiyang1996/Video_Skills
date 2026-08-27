#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
BASE_OUTPUT="${BASE_OUTPUT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start}"
PIPELINE_TAG="${PIPELINE_TAG:-auto_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${BASE_OUTPUT}/${PIPELINE_TAG}"
LANES="${LANES:-20}"
EXAMPLES_PER_LANE="${EXAMPLES_PER_LANE:-5}"
TOTAL_REPAIR_EXAMPLES="${TOTAL_REPAIR_EXAMPLES:-100}"
GPTOSS_MODEL="${GPTOSS_MODEL:-openai/gpt-oss-120b}"
DATASETS="${DATASETS:-cg_bench,video_holmes}"
QUALITY_REPORT="${QUALITY_REPORT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/free_pilot_quality_report.json}"
SOURCE_SNAPSHOT_DIR="${SOURCE_SNAPSHOT_DIR:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_with_paid_repair}"
DEV_PERCENT="${DEV_PERCENT:-10}"
SFT_TARGET_TOTAL="${SFT_TARGET_TOTAL:-200}"
POLL_S="${POLL_S:-120}"
RUN_RETRIEVAL_JOBS="${RUN_RETRIEVAL_JOBS:-1}"
RETRIEVAL_TAG="${RETRIEVAL_TAG:-${PIPELINE_TAG}_retrieval}"
RETRIEVAL_LIMIT="${RETRIEVAL_LIMIT:-25}"
RETRIEVAL_REPEATS="${RETRIEVAL_REPEATS:-2}"
RETRIEVAL_CG_STARTS="${RETRIEVAL_CG_STARTS:-100 125 150 175}"
RETRIEVAL_VIDEO_HOLMES_STARTS="${RETRIEVAL_VIDEO_HOLMES_STARTS:-100 125 150 175}"

mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

echo "{\"status\":\"starting\",\"pipeline_tag\":\"${PIPELINE_TAG}\",\"run_dir\":\"${RUN_DIR}\"}"

DONE_PATHS=(
  "${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first"
)
CANDIDATES="${RUN_DIR}/repair_candidates.txt"
DONE_ARGS=()
for done_path in "${DONE_PATHS[@]}"; do
  DONE_ARGS+=(--done-path "${done_path}")
done
DATASET_ARGS=()
DATASETS_SPACED="${DATASETS//,/ }"
read -r -a DATASET_LIST <<< "${DATASETS_SPACED}"
for dataset in "${DATASET_LIST[@]}"; do
  DATASET_ARGS+=(--dataset "${dataset}")
done
python "${REPO_ROOT}/scripts/sft_pilot/build_repair_candidates.py" \
  --quality-report "${QUALITY_REPORT}" \
  --output "${CANDIDATES}" \
  "${DATASET_ARGS[@]}" \
  --limit "${TOTAL_REPAIR_EXAMPLES}" \
  "${DONE_ARGS[@]}"

mapfile -t IDS < <(grep -v '^[[:space:]]*$' "${CANDIDATES}" || true)
if [[ "${#IDS[@]}" -eq 0 ]]; then
  echo '{"status":"no_candidates"}'
  exit 1
fi

JOB_IDS=()
for lane in $(seq 0 "$((LANES - 1))"); do
  offset=$((lane * EXAMPLES_PER_LANE))
  if [[ "${offset}" -ge "${#IDS[@]}" ]]; then
    break
  fi
  tag="${PIPELINE_TAG}_repair_lane${lane}"
  job_id="$(
    REPAIR_TAG="${tag}" \
    QUALITY_REPORT="${QUALITY_REPORT}" \
    EXAMPLE_IDS_FILE="${CANDIDATES}" \
    MAX_EXAMPLES="${EXAMPLES_PER_LANE}" \
    EXAMPLE_OFFSET="${offset}" \
    GPTOSS_MODEL="${GPTOSS_MODEL}" \
    DATASETS="${DATASETS}" \
    REPAIR_ATTEMPTS=3 \
    REPAIR_RETRY_SLEEP_S=300 \
    PARTITION=gamma ACCOUNT=gamma QOS=default GRES=gpu:l40s:1 \
    "${REPO_ROOT}/scripts/sft_pilot/submit_l2_repair.sh"
  )"
  JOB_IDS+=("${job_id}")
  echo "{\"status\":\"submitted_repair_lane\",\"lane\":${lane},\"job_id\":\"${job_id}\",\"offset\":${offset}}"
done

RETRIEVAL_JOB_IDS=()
if [[ "${RUN_RETRIEVAL_JOBS}" == "1" ]]; then
  for start in ${RETRIEVAL_CG_STARTS}; do
    while read -r line; do
      job_id="${line%% *}"
      [[ -n "${job_id}" ]] && RETRIEVAL_JOB_IDS+=("${job_id}")
      echo "{\"status\":\"submitted_retrieval_lane\",\"dataset\":\"cg_bench\",\"start\":${start},\"line\":\"${line}\"}"
    done < <(
      DATASET=cg_bench STARTS="${start}" LIMIT="${RETRIEVAL_LIMIT}" REPEATS="${RETRIEVAL_REPEATS}" \
      PILOT_TAG="${RETRIEVAL_TAG}" GRAPH_MODEL="${GPTOSS_MODEL}" \
      PARTITION=gamma ACCOUNT=gamma QOS=default GRES=gpu:l40s:1 \
      CLIP_WORKERS=1 GRAPH_WORKERS=2 \
      "${REPO_ROOT}/scripts/sft_pilot/submit_resumable_lane.sh"
    )
  done
  for start in ${RETRIEVAL_VIDEO_HOLMES_STARTS}; do
    while read -r line; do
      job_id="${line%% *}"
      [[ -n "${job_id}" ]] && RETRIEVAL_JOB_IDS+=("${job_id}")
      echo "{\"status\":\"submitted_retrieval_lane\",\"dataset\":\"video_holmes\",\"start\":${start},\"line\":\"${line}\"}"
    done < <(
      DATASET=video_holmes STARTS="${start}" LIMIT="${RETRIEVAL_LIMIT}" REPEATS="${RETRIEVAL_REPEATS}" \
      PILOT_TAG="${RETRIEVAL_TAG}" GRAPH_MODEL="${GPTOSS_MODEL}" \
      PARTITION=gamma ACCOUNT=gamma QOS=default GRES=gpu:l40s:1 \
      CLIP_WORKERS=1 GRAPH_WORKERS=2 \
      "${REPO_ROOT}/scripts/sft_pilot/submit_resumable_lane.sh"
    )
  done
fi

if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
  echo '{"status":"no_jobs_submitted"}'
  exit 1
fi

ALL_WAIT_JOB_IDS=("${JOB_IDS[@]}" "${RETRIEVAL_JOB_IDS[@]}")
JOB_CSV="$(IFS=,; echo "${ALL_WAIT_JOB_IDS[*]}")"
while true; do
  active="$(squeue -h -j "${JOB_CSV}" -o '%i' | wc -l)"
  echo "{\"status\":\"waiting_repair_jobs\",\"active\":${active},\"job_ids\":\"${JOB_CSV}\"}"
  if [[ "${active}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_S}"
done

sacct -j "${JOB_CSV}" --format=JobID,JobName%25,State,ExitCode,Elapsed,NodeList -P > "${RUN_DIR}/jobs_sacct.txt" 2>/dev/null || true

REPAIR_STAGE_ARGS=()
for lane in $(seq 0 "$((LANES - 1))"); do
  stage="${BASE_OUTPUT}/collection_20260713_l2_first/${PIPELINE_TAG}_repair_lane${lane}/stages"
  if [[ -d "${stage}" ]]; then
    REPAIR_STAGE_ARGS+=(--repair-stage-root "${stage}")
  fi
done
for stage in "${REPO_ROOT}"/dataset_clip_wrapper/output/sft_cold_start/collection_20260713_l2_first/l2_repair_20260713_paid_*lane*/stages; do
  if [[ -d "${stage}" ]]; then
    REPAIR_STAGE_ARGS+=(--repair-stage-root "${stage}")
  fi
done

SNAPSHOT_DIR="${RUN_DIR}/snapshot"
PILOT_ROOT_ARGS=(
  --pilot-root dataset_clip_wrapper/output/pilot_20260710_free
  --pilot-root dataset_clip_wrapper/output/pilot_20260710
  --pilot-root dataset_clip_wrapper/output/pilot_expand_20260710
  --pilot-root dataset_clip_wrapper/output/pilot_corrected_v2_20260710
)
if [[ -d "${REPO_ROOT}/dataset_clip_wrapper/output/${RETRIEVAL_TAG}" ]]; then
  PILOT_ROOT_ARGS+=(--pilot-root "dataset_clip_wrapper/output/${RETRIEVAL_TAG}")
fi
python -m dataset_clip_wrapper.collect_sft_snapshot \
  --output-dir "${SNAPSHOT_DIR}" \
  "${PILOT_ROOT_ARGS[@]}" \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  --expert-demos ../video_skills_relaunched/dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --motif-bank motif/output/batch3_strict_qwen_motif_bank.jsonl \
  "${REPAIR_STAGE_ARGS[@]}" \
  --balance-verifier

MOTIF_BANK="${RUN_DIR}/motif_bank.jsonl"
MOTIF_SUMMARY="${RUN_DIR}/motif_summary.json"
REPAIR_REPORTS_JSONL="${RUN_DIR}/repair_reports_for_motif.jsonl"
python - <<'PY' "${REPAIR_REPORTS_JSONL}" "${BASE_OUTPUT}/collection_20260713_l2_first" "${RUN_DIR}"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
roots = [Path(arg) for arg in sys.argv[2:]]
seen = set()
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/repair_05_report.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = (payload.get("example_id"), path.parent.name)
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY
MOTIF_INPUTS=("${REPAIR_REPORTS_JSONL}")
python -m motif.mine_existing_l1_l2 "${MOTIF_INPUTS[@]}" \
  --output-bank "${MOTIF_BANK}" \
  --summary-output "${MOTIF_SUMMARY}" \
  --min-support 2 || true

SNAPSHOT_WITH_MOTIF="${RUN_DIR}/snapshot_with_motif"
python -m dataset_clip_wrapper.collect_sft_snapshot \
  --output-dir "${SNAPSHOT_WITH_MOTIF}" \
  "${PILOT_ROOT_ARGS[@]}" \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  --expert-demos ../video_skills_relaunched/dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --motif-bank "${MOTIF_BANK}" \
  "${REPAIR_STAGE_ARGS[@]}" \
  --balance-verifier

python -m dataset_clip_wrapper.training.build_sft_splits \
  --snapshot-dir "${SNAPSHOT_WITH_MOTIF}" \
  --output-dir "${RUN_DIR}/splits" \
  --dev-percent "${DEV_PERCENT}" \
  --target-total "${SFT_TARGET_TOTAL}"

echo "{\"status\":\"complete\",\"run_dir\":\"${RUN_DIR}\",\"snapshot\":\"${SNAPSHOT_WITH_MOTIF}\",\"splits\":\"${RUN_DIR}/splits\"}"
