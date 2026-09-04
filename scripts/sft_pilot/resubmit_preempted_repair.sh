#!/usr/bin/env bash
# Idempotently resubmit CG-catalog repair shards that were preempted, timed out,
# or failed, and are not currently queued or running.  The repair path is
# resumable (--retry-failed-clip-schemas keeps good rows and retries
# placeholders), so a preempted shard loses no completed clips, only wall clock.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills}"
LANE="${LANE:-l2_expansion_20260831}"
LOG="${REPO_ROOT}/dataset_clip_wrapper/output/${LANE}/slurm_logs_repair_vllm"
mkdir -p "$LOG"
cd "$REPO_ROOT"
active="$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -E '^(cg-repair|re-cg)-' | sed -E 's/^(cg-repair|re-cg)-//' | sort -u || true)"
for d in dataset_clip_wrapper/output/${LANE}/cg_bench/start_*_limit_25; do
  start="$(basename "$d" | sed -E 's/start_([0-9]+)_limit_25/\1/')"
  if grep -qx "$start" <<<"$active"; then echo "  start_${start}: active, skip"; continue; fi
  ph="$(python3 - "$d" <<'PY'
import json,glob,sys
d=sys.argv[1]; t=e=0
for f in glob.glob(f'{d}/stages/*/*clip_schemas.jsonl'):
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        t+=1; e+=bool(r.get('model_error'))
print(f"{100*e/max(t,1):.1f}")
PY
)"
  if awk -v p="$ph" 'BEGIN{exit !(p<=1.0)}'; then echo "  start_${start}: placeholders ${ph}% <= 1%, done"; continue; fi
  jid="$(sbatch --parsable --job-name="cg-repair-${start}" \
    --partition=scavenger --account=scavenger --qos=scavenger \
    --gres=gpu:rtxa6000:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
    --output="$LOG/repair-${start}-%j.out" --error="$LOG/repair-${start}-%j.err" \
    --export="ALL,DATASET=cg_bench,SPLIT=train,START_INDEX=${start},LIMIT=25,SMOKE=0,PILOT_TAG=${LANE},SERVE_BACKEND=vllm,CLIP_WORKERS=16,GRAPH_WORKERS=2,CLIP_TIMEOUT_S=300,GRAPH_MODEL=openai/gpt-oss-120b:free,RETRY_FAILED_CLIP_SCHEMAS=1" \
    scripts/sft_pilot/run_local_qwen_worker.sh)"
  echo "  start_${start}: placeholders ${ph}%, resubmitted -> ${jid}"
done
