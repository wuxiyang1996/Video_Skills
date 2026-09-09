#!/usr/bin/env bash
# Combined VH + VRBench(pilot 480) + CG(1,080) reader SFT: wait for the in-domain teacher data, mix, train (scavenger), evaluate on all three held-out sets.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
R=/fs/nexus-scratch/wuxiyang/reader_train; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; V=/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1; T=/fs/nexus-scratch/wuxiyang/cg_train_l1
M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm; C=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/cg_qa
until grep -q "CG training data ready" $S/cg_train_data.log 2>/dev/null && grep -q "VRBench training data ready" $S/vrb_train_data.log 2>/dev/null; do sleep 600; done
echo "$(date +%m-%d\ %H:%M) in-domain data ready"
# same filter as VH sft_v1: right answer + citation precision >= 0.5 where gold spans exist
$SPY -m trainer.reader.build_sft_data --example-index $T/narr_px_plus/example_index.json --rollouts $R/cg_teacher_s0.rollouts.jsonl $R/cg_teacher_s1.rollouts.jsonl $R/cg_teacher_s2.rollouts.jsonl --output $R/sft_cg_p50.jsonl --min-precision 0.5 2>&1 | tail -3
$SPY -m trainer.reader.build_sft_data --example-index $V/derived_pilot60/example_index.json --rollouts $R/vrb_teacher_s0.rollouts.jsonl $R/vrb_teacher_s1.rollouts.jsonl $R/vrb_teacher_s2.rollouts.jsonl --output $R/sft_vrb_p50.jsonl --min-precision 0.5 2>&1 | tail -3
python3 - <<'PY'
import json,random
R='/fs/nexus-scratch/wuxiyang/reader_train'; rows=[]
for name in ['sft_v1','sft_vrb_p50','sft_cg_p50']:
    rs=[json.loads(l) for l in open(f'{R}/{name}.jsonl')]; print(name,len(rs)); rows+=rs
random.Random(0).shuffle(rows); open(f'{R}/sft_mix.jsonl','w').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)); print('sft_mix',len(rows))
PY
/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python -m trainer.reader.sft_lora --data $R/sft_mix.jsonl --output-dir $R/sft_mix --max-len 32768 --dry-run 2>&1 | tail -1
jid=${SFT_JID:-}; [ -z "$jid" ] && jid=$(DATA=$R/sft_mix.jsonl OUT=$R/sft_mix MAXLEN=32768 SAVE_STEPS=80 sbatch --parsable scripts/launch/reader_sft.sbatch); echo "$(date +%m-%d\ %H:%M) SFT mix job $jid"
while [ -n "$(squeue -h -j $jid -o %i)" ]; do sleep 300; done; ls $R/sft_mix/adapter/adapter_config.json >/dev/null || { echo "SFT mix produced no adapter; see $R/slurm_logs/sft-$jid.err"; exit 1; }
A=$R/sft_mix/adapter; export KEEP_MERGED=1
j1=$(ADAPTER=$A L1GLOB="$P/narr_px_plus/stages/*/04_l1_example.json" IDS=$M/fresh_ids_300.txt OUT=$M/rdr_sft_mix_rationale_fresh300 sbatch --parsable scripts/launch/reader_eval.sbatch)
j2=$(ADAPTER=$A MAXLEN=65536 L1GLOB="$P/cg_narr_px_plus/stages/*/04_l1_example.json" IDS=$P/cg_ids_237.txt OUT=$C/cg_rdr_sft_mix_rationale_237 sbatch --parsable scripts/launch/reader_eval.sbatch)
j3=$(ADAPTER=$A L1GLOB="$V/heldout60/derived/vrbench/*/derived/stages/*/04_l1_example.json" IDS=$V/heldout60/all_ids.txt OUT=$V/heldout60/measure/rdr_sft_mix_rationale sbatch --parsable scripts/launch/reader_eval.sbatch)
j4=$(ADAPTER=$R/sft_v2/adapter L1GLOB="$V/heldout60/derived/vrbench/*/derived/stages/*/04_l1_example.json" IDS=$V/heldout60/all_ids.txt OUT=$V/heldout60/measure/rdr_sft_v2_rationale sbatch --parsable scripts/launch/reader_eval.sbatch)
echo "$(date +%m-%d\ %H:%M) eval jobs VH=$j1 CG=$j2 VRB=$j3 VRB(sft_v2)=$j4"
while [ -n "$(squeue -h -j $j1,$j2,$j3,$j4 -o %i)" ]; do sleep 600; done
echo; echo "=== MIX READER (VH+VRB+CG SFT) ==="
echo "## VH fresh 300: vs sft_v2 / base / 235B"; python3 $S/boot_pair.py $M/rdr_sft_v2_rationale_fresh300.rows.jsonl $M/rdr_sft_mix_rationale_fresh300.rows.jsonl sft_v2 mix | tail -1; python3 $S/boot_pair.py $M/rdr_q35_9b_base_rationale_fresh300.rows.jsonl $M/rdr_sft_mix_rationale_fresh300.rows.jsonl base mix | tail -1; python3 $S/boot_pair.py $M/direct_rationale_narrpx_fresh300.rows.jsonl $M/rdr_sft_mix_rationale_fresh300.rows.jsonl b235 mix | tail -1
echo "## CG 237: vs sft_v2 / 235B"; python3 $S/boot_pair.py $C/cg_rdr_sft_v2_rationale_237.rows.jsonl $C/cg_rdr_sft_mix_rationale_237.rows.jsonl sft_v2 mix | tail -1; python3 $S/boot_pair.py $C/cg_direct_rationale_237.rows.jsonl $C/cg_rdr_sft_mix_rationale_237.rows.jsonl b235 mix | tail -1
echo "## VRBench held-out 495: vs sft_v2 / 235B rationale / 235B direct"; python3 $S/boot_pair.py $V/heldout60/measure/rdr_sft_v2_rationale.rows.jsonl $V/heldout60/measure/rdr_sft_mix_rationale.rows.jsonl sft_v2 mix | tail -1; python3 $S/boot_pair.py $V/heldout60/measure/rationale.rows.jsonl $V/heldout60/measure/rdr_sft_mix_rationale.rows.jsonl b235 mix | tail -1; python3 $S/boot_pair.py $V/heldout60/measure/direct.rows.jsonl $V/heldout60/measure/rdr_sft_mix_rationale.rows.jsonl b235direct mix | tail -1
echo "=== MIX CHAIN DONE ==="
