#!/usr/bin/env bash
#SBATCH --job-name=rdr-eval --partition=scavenger --account=scavenger --qos=scavenger
#SBATCH --gres=gpu:1 --constraint="Ampere|Ada|Hopper|Blackwell" --exclude=cbcb26,clip04,clip11,clip14,cml12,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24,cml25,cml26,cml27,cml28,gammagpu00,gammagpu01,gammagpu02,gammagpu03,gammagpu04,gammagpu05,gammagpu06,gammagpu07,gammagpu08,gammagpu09,tron06,tron07,tron08,tron09,tron10,tron11,tron12,tron13,tron14,tron15,tron16,tron17,tron18,tron19,tron20,tron21,tron22,tron23,tron24,tron25,tron26,tron27,tron28,tron29,tron30,tron31,tron32,tron33,tron34,tron35,tron36,tron37,tron38,tron39,tron40,tron41,tron42,tron43,tron44,tron45,tron46,tron47,tron48,tron49,tron50,tron51,tron52,tron53,tron54,tron55,tron56,tron57,tron58,tron59,tron60,tron61,tron64,tron67,vulcan33,vulcan34,vulcan35,vulcan36,vulcan37,vulcan38,vulcan39,vulcan40,vulcan41,vulcan42,vulcan43,vulcan44 --cpus-per-task=8 --mem=64G --time=06:00:00
#SBATCH --output=/fs/nexus-scratch/wuxiyang/reader_train/slurm_logs/eval-%j.out --error=/fs/nexus-scratch/wuxiyang/reader_train/slurm_logs/eval-%j.err
# Serve Qwen3.5-9B (+ optional LoRA adapter $ADAPTER) with vLLM and answer $IDS over $L1GLOB in the rationale format. Env: ADAPTER, L1GLOB, IDS, OUT (prefix)
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
export HF_HOME=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache TRANSFORMERS_CACHE=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache/hub
export CUDA_HOME=/opt/common/cuda/cuda-13.0.2 CUDA_PATH=/opt/common/cuda/cuda-13.0.2 PATH=/opt/common/cuda/cuda-13.0.2/bin:$PWD/.venv-qwen35-vllm/bin:$PATH VLLM_USE_DEEP_GEMM=0 TOKENIZERS_PARALLELISM=false SETUPTOOLS_USE_DISTUTILS=stdlib PYTHONPATH=$PWD
PORT=$((19000 + SLURM_JOB_ID % 1000)); BASE=Qwen/Qwen3.5-9B; NAME=$BASE
LORA_ARGS=()
if [ -n "${ADAPTER:-}" ]; then
  # vLLM's LoRA serving mis-applies Qwen3.5 adapters (collapse to one option); merge into the weights instead.
  # merged weights are 17 GB each: keep them off the 200 GB scratch quota, under the project tree, and delete after use
  MERGED=${MERGED:-/fs/gamma-projects/vlm-robot/Video_Skills/dataset_clip_wrapper/output/reader_merged/$(basename $(dirname "$ADAPTER"))_$(basename "$ADAPTER")}
  # several evaluation jobs may share one adapter: one merges (lock), the others wait for config.json
  if [ ! -f "$MERGED/config.json" ]; then
    if mkdir "$MERGED.lock" 2>/dev/null; then
      PYTHONPATH=$PWD /fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python -m trainer.reader.merge_adapter --adapter "$ADAPTER" --out "$MERGED" 2>&1 | grep -v Warning | tail -n 2
      rmdir "$MERGED.lock"
    else
      for i in $(seq 1 360); do [ -f "$MERGED/config.json" ] && [ ! -d "$MERGED.lock" ] && break; sleep 10; done
    fi
  fi
  BASE=$MERGED; NAME=qwen3.5-9b-reader   # name must contain "qwen3": the client only sends enable_thinking=false for Qwen3-family names
fi
.venv-qwen35-vllm/bin/vllm serve $BASE --host 127.0.0.1 --port $PORT --served-model-name $NAME --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 "${LORA_ARGS[@]}" > /fs/nexus-scratch/wuxiyang/reader_train/slurm_logs/vllm-$SLURM_JOB_ID.log 2>&1 &
for i in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 2; done; curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "vllm did not start"; exit 1; }
SPY=.venv-qwen35-serve/bin/python
$SPY -m scripts.eval.measure_answer_chain --l1-glob "$L1GLOB" --indices-from all --conditions direct --rationale --example-ids "$IDS" --workers 8 --timeout-s 600 --skill-model openai/gpt-oss-120b --answer-model $NAME --answer-api-base http://127.0.0.1:$PORT/v1/chat/completions --dump-rollouts "$OUT.rollouts.jsonl" --output "$OUT.json" > "$OUT.log" 2>&1
f=$OUT.rows.jsonl; r=$(cat $f|wc -l); c=$(grep -c '"correct": true' $f); echo "$(basename $OUT): $r rows, acc $(python3 -c "print(round(100*$c/max($r,1),1))")%"
if [ -n "${ADAPTER:-}" ] && [ "${KEEP_MERGED:-0}" != "1" ]; then rm -rf "$MERGED"; echo "merged weights removed: $MERGED"; fi
