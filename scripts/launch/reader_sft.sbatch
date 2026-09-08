#!/usr/bin/env bash
#SBATCH --job-name=rdr-sft --partition=scavenger --account=scavenger --qos=scavenger
#SBATCH --gres=gpu:1 --constraint="Ampere|Ada|Hopper|Blackwell" --exclude=cbcb26,clip04,clip11,clip14,cml12,cml17,cml18,cml19,cml20,cml21,cml22,cml23,cml24,cml25,cml26,cml27,cml28,gammagpu00,gammagpu01,gammagpu02,gammagpu03,gammagpu04,gammagpu05,gammagpu06,gammagpu07,gammagpu08,gammagpu09,tron06,tron07,tron08,tron09,tron10,tron11,tron12,tron13,tron14,tron15,tron16,tron17,tron18,tron19,tron20,tron21,tron22,tron23,tron24,tron25,tron26,tron27,tron28,tron29,tron30,tron31,tron32,tron33,tron34,tron35,tron36,tron37,tron38,tron39,tron40,tron41,tron42,tron43,tron44,tron45,tron46,tron47,tron48,tron49,tron50,tron51,tron52,tron53,tron54,tron55,tron56,tron57,tron58,tron59,tron60,tron61,tron64,tron67,vulcan33,vulcan34,vulcan35,vulcan36,vulcan37,vulcan38,vulcan39,vulcan40,vulcan41,vulcan42,vulcan43,vulcan44 --cpus-per-task=8 --mem=96G --time=08:00:00
#SBATCH --output=/fs/nexus-scratch/wuxiyang/reader_train/slurm_logs/sft-%j.out --error=/fs/nexus-scratch/wuxiyang/reader_train/slurm_logs/sft-%j.err
# LoRA SFT of Qwen3.5-9B on verified reader data. Env: DATA (jsonl), OUT (dir), EVAL (jsonl, optional), EPOCHS, LR
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
export HF_HOME=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache TRANSFORMERS_CACHE=/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache/hub PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
PY=/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python
$PY -m trainer.reader.sft_lora --data "$DATA" --output-dir "$OUT" --max-len 24576 --epochs "${EPOCHS:-1}" --lr "${LR:-1e-4}" --grad-accum 8 ${EVAL:+--eval-data "$EVAL"} 2>&1 | tail -n 40
