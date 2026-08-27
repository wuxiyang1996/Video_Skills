#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
PIPELINE_TAG="${PIPELINE_TAG:-sft_recovery_$(date +%Y%m%d_%H%M%S)}"
SOURCE_REPAIR_TAG="${SOURCE_REPAIR_TAG:-sft_auto_20260713_full}"
RETRIEVAL_TAG="${RETRIEVAL_TAG:-sft_auto_20260713_full_retrieval}"
BASE_OUTPUT="${BASE_OUTPUT:-${REPO_ROOT}/dataset_clip_wrapper/output/sft_cold_start}"
RUN_DIR="${BASE_OUTPUT}/${PIPELINE_TAG}"
MIN_REPAIR_REPORTS="${MIN_REPAIR_REPORTS:-99}"
MIN_RETRIEVAL_EXAMPLES="${MIN_RETRIEVAL_EXAMPLES:-200}"
DEV_PERCENT="${DEV_PERCENT:-10}"
SFT_TARGET_TOTAL="${SFT_TARGET_TOTAL:-200}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

repair_root="${BASE_OUTPUT}/collection_20260713_l2_first"
retrieval_root="${REPO_ROOT}/dataset_clip_wrapper/output/${RETRIEVAL_TAG}"
repair_count="$(find "${repair_root}" -path "*/${SOURCE_REPAIR_TAG}_repair_lane*/stages/*/repair_05_report.json" -type f 2>/dev/null | wc -l)"
retrieval_count="$(find "${retrieval_root}" -path '*/start_*/examples.jsonl' -type f -print0 2>/dev/null | xargs -0 -r cat | wc -l)"

jq -n \
  --arg pipeline_tag "${PIPELINE_TAG}" \
  --arg source_repair_tag "${SOURCE_REPAIR_TAG}" \
  --arg retrieval_tag "${RETRIEVAL_TAG}" \
  --argjson repair_reports "${repair_count}" \
  --argjson retrieval_examples "${retrieval_count}" \
  --argjson min_repair_reports "${MIN_REPAIR_REPORTS}" \
  --argjson min_retrieval_examples "${MIN_RETRIEVAL_EXAMPLES}" \
  '{pipeline_tag:$pipeline_tag,source_repair_tag:$source_repair_tag,retrieval_tag:$retrieval_tag,repair_reports:$repair_reports,retrieval_examples:$retrieval_examples,min_repair_reports:$min_repair_reports,min_retrieval_examples:$min_retrieval_examples}' \
  > "${RUN_DIR}/collection_readiness.json"

if (( repair_count < MIN_REPAIR_REPORTS || retrieval_count < MIN_RETRIEVAL_EXAMPLES )); then
  echo "SFT collection is incomplete; see ${RUN_DIR}/collection_readiness.json" >&2
  exit 2
fi

REPAIR_STAGE_ARGS=()
for stage in "${repair_root}"/${SOURCE_REPAIR_TAG}_repair_lane*/stages; do
  [[ -d "${stage}" ]] && REPAIR_STAGE_ARGS+=(--repair-stage-root "${stage}")
done
for stage in "${repair_root}"/l2_repair_20260713_paid_*lane*/stages; do
  [[ -d "${stage}" ]] && REPAIR_STAGE_ARGS+=(--repair-stage-root "${stage}")
done

PILOT_ROOT_ARGS=(
  --pilot-root dataset_clip_wrapper/output/pilot_20260710_free
  --pilot-root dataset_clip_wrapper/output/pilot_20260710
  --pilot-root dataset_clip_wrapper/output/pilot_expand_20260710
  --pilot-root dataset_clip_wrapper/output/pilot_corrected_v2_20260710
  --pilot-root "dataset_clip_wrapper/output/${RETRIEVAL_TAG}"
)

REPAIR_REPORTS_JSONL="${RUN_DIR}/repair_reports_for_motif.jsonl"
python - "${REPAIR_REPORTS_JSONL}" "${repair_root}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
seen = set()
with out.open("w", encoding="utf-8") as handle:
    for path in sorted(root.glob("**/repair_05_report.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload.get("example_id"), path.parent.name)
        if key in seen:
            continue
        seen.add(key)
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PY

MINED_MOTIF_BANK="${RUN_DIR}/mined_motif_bank.jsonl"
MOTIF_BANK="${RUN_DIR}/motif_bank.jsonl"
MOTIF_SUMMARY="${RUN_DIR}/motif_summary.json"
python -m motif.mine_existing_l1_l2 "${REPAIR_REPORTS_JSONL}" \
  --output-bank "${MINED_MOTIF_BANK}" \
  --summary-output "${MOTIF_SUMMARY}" \
  --min-support 2

python - "${MOTIF_BANK}" motif/output/batch3_strict_qwen_motif_bank.jsonl "${MINED_MOTIF_BANK}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
seen = set()
with out.open("w", encoding="utf-8") as handle:
    for source in map(Path, sys.argv[2:]):
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            motif_id = str(row.get("motif_id") or "")
            if motif_id in seen:
                continue
            seen.add(motif_id)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

SNAPSHOT_DIR="${RUN_DIR}/snapshot"
python -m dataset_clip_wrapper.collect_sft_snapshot \
  --output-dir "${SNAPSHOT_DIR}" \
  "${PILOT_ROOT_ARGS[@]}" \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl \
  --extra-rollout-jsonl ../video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl \
  --expert-demos ../video_skills_relaunched/dataset_clip_wrapper/output/expert_demos/batch3_p5_video_only_expert_demos_compact.jsonl \
  --motif-bank "${MOTIF_BANK}" \
  "${REPAIR_STAGE_ARGS[@]}" \
  --balance-verifier

EXAMPLE_VIDEO_MAP="${RUN_DIR}/example_video_map.json"
python - "${EXAMPLE_VIDEO_MAP}" "${REPO_ROOT}" "${RETRIEVAL_TAG}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
repo = Path(sys.argv[2])
retrieval_tag = sys.argv[3]
roots = [
    repo / "dataset_clip_wrapper/output/pilot_20260710_free",
    repo / "dataset_clip_wrapper/output/pilot_20260710",
    repo / "dataset_clip_wrapper/output/pilot_expand_20260710",
    repo / "dataset_clip_wrapper/output/pilot_corrected_v2_20260710",
    repo / f"dataset_clip_wrapper/output/{retrieval_tag}",
]
paths = [path for root in roots for path in root.glob("**/examples.jsonl")]
paths.extend([
    repo.parent / "video_skills_relaunched/dataset_clip_wrapper/output/batch3_cg_strict_qwen.jsonl",
    repo.parent / "video_skills_relaunched/dataset_clip_wrapper/output/batch3_video_holmes_strict_qwen.jsonl",
])
mapping = {}
for path in paths:
    if not path.exists():
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row.get("example_id")
        video = row.get("video") if isinstance(row.get("video"), dict) else {}
        video_id = video.get("video_id")
        dataset = row.get("dataset") or str(example_id or "unknown").split(":", 1)[0]
        if example_id and video_id:
            mapping[str(example_id)] = f"{dataset}:video:{video_id}"
out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

python -m dataset_clip_wrapper.training.build_sft_splits \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --output-dir "${RUN_DIR}/splits" \
  --dev-percent "${DEV_PERCENT}" \
  --target-total "${SFT_TARGET_TOTAL}" \
  --example-video-map "${EXAMPLE_VIDEO_MAP}"

echo "SFT snapshot complete: ${RUN_DIR}"
