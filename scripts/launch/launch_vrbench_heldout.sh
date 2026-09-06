#!/usr/bin/env bash
# VRBench held-out 60 (videos 61-120 by duration): 6 L1 shards of 10 videos. Usage: launch_vrbench_heldout.sh [shard ...]
set -euo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; P=/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1/heldout60
SHARDS=("$@"); [ ${#SHARDS[@]} -gt 0 ] || SHARDS=(0 1 2 3 4 5)
for s in "${SHARDS[@]}"; do
  jid=$(sbatch --parsable --job-name=vrbh-$s --partition=scavenger --account=scavenger --qos=scavenger \
    --gres=gpu:1 --constraint="Ampere|Ada|Hopper|Blackwell" --exclude=cbcb26,clip04,clip11,clip14,cml12,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24,cml25,cml26,cml27,cml28,gammagpu00,gammagpu01,gammagpu02,gammagpu03,gammagpu04,gammagpu05,gammagpu06,gammagpu07,gammagpu08,gammagpu09,tron06,tron07,tron08,tron09,tron10,tron11,tron12,tron13,tron14,tron15,tron16,tron17,tron18,tron19,tron20,tron21,tron22,tron23,tron24,tron25,tron26,tron27,tron28,tron29,tron30,tron31,tron32,tron33,tron34,tron35,tron36,tron37,tron38,tron39,tron40,tron41,tron42,tron43,tron44,tron45,tron46,tron47,tron48,tron49,tron50,tron51,tron52,tron53,tron54,tron55,tron56,tron57,tron58,tron59,tron60,tron61,tron64,tron67,vulcan33,vulcan34,vulcan35,vulcan36,vulcan37,vulcan38,vulcan39,vulcan40,vulcan41,vulcan42,vulcan43,vulcan44 --cpus-per-task=8 --mem=64G --time=10:00:00 \
    --output=$P/slurm_logs/shard$s-%j.out --error=$P/slurm_logs/shard$s-%j.err \
    --export=ALL,DATASET=vrbench,SPLIT=train,START_INDEX=0,LIMIT=960,SMOKE=0,PILOT_TAG=vrbench_pilot_v1/heldout60/shard$s,SERVE_BACKEND=vllm,CLIP_WORKERS=16,GRAPH_WORKERS=2,CLIP_TIMEOUT_S=300,GRAPH_MODEL=openai/gpt-oss-120b,RETRY_FAILED_CLIP_SCHEMAS=1,EXAMPLE_ID_ALLOWLIST=$P/allowlists/shard$s.txt \
    scripts/sft_pilot/run_local_qwen_worker.sh)
  echo "vrbh-$s -> $jid"
done
