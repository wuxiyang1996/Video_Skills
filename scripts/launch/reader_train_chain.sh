#!/usr/bin/env bash
# After the train-split L1 completes: derive slim per-question examples -> narrative+dialogue catalog -> teacher rationales (3 option orders)
# -> verified SFT data -> LoRA SFT (slurm) -> fresh-300 evaluation with the adapter -> watershed verdict vs 38.0.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
T=/fs/nexus-scratch/wuxiyang/vh_train_l1; R=/fs/nexus-scratch/wuxiyang/reader_train; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm
until grep -q "VH TRAIN L1 COMPLETE" $S/vh_train_l1.log 2>/dev/null; do sleep 900; done
echo "$(date +%m-%d\ %H:%M) train L1 complete"
# 1. derive slim per-question examples (1,551)
timeout 7200 $SPY scripts/eval/derive_full_question_examples.py --frozen-l1-glob "$T/shard*/video_holmes/start_0_limit_300/stages/*/04_l1_example.json" --dataset video_holmes --split train --output-root $T/derived --slim 2>&1 | tail -3
IDX=$T/derived/example_index.json; python3 -c "import json; idx=json.load(open('$IDX')); open('$T/train_ids.txt','w').write('\n'.join(idx)+'\n'); print('train questions', len(idx))"
# 2. narrative + dialogue + clips catalog (one generation; a second later for variance robustness)
timeout 21600 $SPY scripts/eval/build_narrative_catalog.py --example-index $IDX --example-ids $T/train_ids.txt --output-root $T/narr_px_plus --window-s 30 --frames-per-window 16 --asr-dir $T/asr/whisper --no-clip-text --keep-clips --slim --workers 8 2>&1 | tail -2
CIDX=$T/narr_px_plus/example_index.json; L="$T/narr_px_plus/stages/*/04_l1_example.json"; echo "train catalog examples: $(ls -d $T/narr_px_plus/stages/*/ | wc -l)"
# 3. teacher rationales in three option orders (original + two permutations)
export OPENROUTER_PROVIDER_ORDER=Alibaba
for seed in 0 1 2; do
  extra=""; [ $seed -gt 0 ] && extra="--shuffle-options $seed"
  timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --conditions direct --rationale $extra --example-ids $T/train_ids.txt --workers 6 --timeout-s 300 \
    --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $R/teacher_s$seed.rollouts.jsonl --output $R/teacher_s$seed.json > $R/teacher_s$seed.log 2>&1
  f=$R/teacher_s$seed.rows.jsonl; echo "teacher order $seed: $(cat $f|wc -l) rows, acc $(python3 -c "print(round(100*$(grep -c '"correct": true' $f)/max($(cat $f|wc -l),1),1))")%"
done
# 4. verified SFT data (right answer; citation precision >= 0.5 where gold spans exist)
$SPY -m trainer.reader.build_sft_data --example-index $CIDX --rollouts $R/teacher_s0.rollouts.jsonl $R/teacher_s1.rollouts.jsonl $R/teacher_s2.rollouts.jsonl --output $R/sft_v1.jsonl --min-precision 0.5 2>&1 | tail -9
/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo/bin/python -m trainer.reader.sft_lora --data $R/sft_v1.jsonl --output-dir $R/sft_v1 --max-len 24576 --dry-run 2>&1 | tail -1
# 5. SFT on slurm, then evaluate the adapter on the fresh 300 (rationale format) vs the base 38.0
jid=$(DATA=$R/sft_v1.jsonl OUT=$R/sft_v1 sbatch --parsable scripts/launch/reader_sft.sbatch); echo "SFT job $jid"
while squeue -h -j $jid >/dev/null 2>&1 && [ -n "$(squeue -h -j $jid -o %i)" ]; do sleep 300; done; ls $R/sft_v1/adapter/adapter_config.json >/dev/null || { echo "SFT produced no adapter; see $R/slurm_logs/sft-$jid.err"; exit 1; }
jid2=$(ADAPTER=$R/sft_v1/adapter L1GLOB="$P/narr_px_plus/stages/*/04_l1_example.json" IDS=$M/fresh_ids_300.txt OUT=$M/rdr_sft_v1_rationale_fresh300 sbatch --parsable scripts/launch/reader_eval.sbatch); echo "eval job $jid2"
while [ -n "$(squeue -h -j $jid2 -o %i)" ]; do sleep 300; done
echo; echo "=== WATERSHED: SFT v1 (Qwen3.5-9B + LoRA) on the fresh 300, rationale format, vs base 38.0 ==="
python3 $S/boot_pair.py $M/rdr_q35_9b_base_rationale_fresh300.rows.jsonl $M/rdr_sft_v1_rationale_fresh300.rows.jsonl base sft_v1
echo "## vs 235B rationale on the same catalog (45.7)"; python3 $S/boot_pair.py $M/direct_rationale_narrpx_fresh300.rows.jsonl $M/rdr_sft_v1_rationale_fresh300.rows.jsonl b235 sft_v1 | tail -1
