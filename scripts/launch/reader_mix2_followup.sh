#!/usr/bin/env bash
# After the sft_mix2 SFT (all 1,539 CG rows) finishes: evaluate on CG 237 (primary), VRBench held-out 495, VH fresh 300; paired vs sft_mix.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
R=/fs/nexus-scratch/wuxiyang/reader_train; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; V=/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1
M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm; C=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/cg_qa
jid=$(cat $S/sft_mix2.jid); while [ -n "$(squeue -h -j $jid -o %i 2>/dev/null)" ]; do sleep 300; done
ls $R/sft_mix2/adapter/adapter_config.json >/dev/null || { echo "SFT mix2 produced no adapter; see $R/slurm_logs/sft-$jid.err"; exit 1; }
A=$R/sft_mix2/adapter; export KEEP_MERGED=1
j2=$(ADAPTER=$A MAXLEN=65536 L1GLOB="$P/cg_narr_px_plus/stages/*/04_l1_example.json" IDS=$P/cg_ids_237.txt OUT=$C/cg_rdr_sft_mix2_rationale_237 sbatch --parsable scripts/launch/reader_eval.sbatch)
j3=$(ADAPTER=$A L1GLOB="$V/heldout60/derived/vrbench/*/derived/stages/*/04_l1_example.json" IDS=$V/heldout60/all_ids.txt OUT=$V/heldout60/measure/rdr_sft_mix2_rationale sbatch --parsable scripts/launch/reader_eval.sbatch)
j1=$(ADAPTER=$A L1GLOB="$P/narr_px_plus/stages/*/04_l1_example.json" IDS=$M/fresh_ids_300.txt OUT=$M/rdr_sft_mix2_rationale_fresh300 sbatch --parsable scripts/launch/reader_eval.sbatch)
echo "$(date +%m-%d\ %H:%M) mix2 eval jobs CG=$j2 VRB=$j3 VH=$j1"
while [ -n "$(squeue -h -j $j1,$j2,$j3 -o %i)" ]; do sleep 600; done
echo; echo "=== MIX2 (all 1,539 CG rows) vs MIX (613 precise CG rows) ==="
echo "## CG 237"; python3 $S/boot_pair.py $C/cg_rdr_sft_mix_rationale_237.rows.jsonl $C/cg_rdr_sft_mix2_rationale_237.rows.jsonl mix mix2 | tail -1; python3 $S/boot_pair.py $C/cg_direct_rationale_237.rows.jsonl $C/cg_rdr_sft_mix2_rationale_237.rows.jsonl b235 mix2 | tail -1
echo "## VRBench held-out 495"; python3 $S/boot_pair.py $V/heldout60/measure/rdr_sft_mix_rationale.rows.jsonl $V/heldout60/measure/rdr_sft_mix2_rationale.rows.jsonl mix mix2 | tail -1; python3 $S/boot_pair.py $V/heldout60/measure/rationale.rows.jsonl $V/heldout60/measure/rdr_sft_mix2_rationale.rows.jsonl b235 mix2 | tail -1
echo "## VH fresh 300"; python3 $S/boot_pair.py $M/rdr_sft_mix_rationale_fresh300.rows.jsonl $M/rdr_sft_mix2_rationale_fresh300.rows.jsonl mix mix2 | tail -1
rm -rf dataset_clip_wrapper/output/reader_merged/sft_mix2_adapter; echo "=== MIX2 FOLLOWUP DONE ==="
