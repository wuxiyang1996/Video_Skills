#!/usr/bin/env bash
# Resume the training chain after the OOM fix: SFT job -> adapter check -> fresh-300 eval with the adapter -> watershed.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad; R=/fs/nexus-scratch/wuxiyang/reader_train; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm
TAG=${TAG:-sft_v1}
jid=$(DATA=$R/sft_v1.jsonl OUT=$R/$TAG EPOCHS=1 LIMIT=${LIMIT:-} MAXLEN=${MAXLEN:-} LR=${LR:-1e-4} LR=${LR:-1e-4} sbatch --parsable scripts/launch/reader_sft.sbatch); echo "$(date +%m-%d\ %H:%M) SFT job $jid"
while [ -n "$(squeue -h -j $jid -o %i 2>/dev/null)" ]; do sleep 300; done
ADP=$R/$TAG/adapter; if [ ! -f $ADP/adapter_config.json ]; then ADP=$(ls -d $R/$TAG/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1); fi
if [ -z "$ADP" ] || [ ! -f $ADP/adapter_config.json ]; then echo "$(date +%m-%d\ %H:%M) SFT produced no adapter or checkpoint; tail of err:"; tail -n 5 $R/slurm_logs/sft-$jid.err | cut -c1-200; exit 1; fi
echo "$(date +%m-%d\ %H:%M) adapter ready: $ADP"; tail -n 3 $R/slurm_logs/sft-$jid.out | cut -c1-200
jid2=$(ADAPTER=$ADP L1GLOB="$P/narr_px_plus/stages/*/04_l1_example.json" IDS=$M/fresh_ids_300.txt OUT=$M/rdr_${TAG}_rationale_fresh300 sbatch --parsable scripts/launch/reader_eval.sbatch); echo "$(date +%m-%d\ %H:%M) eval job $jid2"
while [ -n "$(squeue -h -j $jid2 -o %i 2>/dev/null)" ]; do sleep 300; done
f=$M/rdr_${TAG}_rationale_fresh300.rows.jsonl; [ -f $f ] || { echo "eval produced no rows; see $R/slurm_logs/eval-$jid2.err"; tail -n 5 $R/slurm_logs/eval-$jid2.err | cut -c1-200; exit 1; }
echo; echo "=== WATERSHED: $TAG (Qwen3.5-9B + LoRA) on the fresh 300, rationale format, vs base 38.0 ==="
python3 $S/boot_pair.py $M/rdr_q35_9b_base_rationale_fresh300.rows.jsonl $f base $TAG
echo "## vs 235B rationale on the same catalog (45.7)"; python3 $S/boot_pair.py $M/direct_rationale_narrpx_fresh300.rows.jsonl $f b235 $TAG | tail -1
