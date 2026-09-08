#!/usr/bin/env bash
# VRBench in-domain training data: teacher rationales (x3 orders) on the pilot-60 catalog (480 q; held-out 60 stays untouched) -> verified SFT rows.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; SPY=.venv-qwen35-serve/bin/python; V=/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1; R=/fs/nexus-scratch/wuxiyang/reader_train
export OPENROUTER_PROVIDER_ORDER=Alibaba; L="$V/derived_pilot60/vrbench/train/derived/stages/*/04_l1_example.json"
for seed in 0 1 2; do extra=""; [ $seed -gt 0 ] && extra="--shuffle-options $seed"
  timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --conditions direct --rationale $extra --example-ids $V/pilot_all_ids.txt --workers 6 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $R/vrb_teacher_s$seed.rollouts.jsonl --output $R/vrb_teacher_s$seed.json > $R/vrb_teacher_s$seed.log 2>&1
  f=$R/vrb_teacher_s$seed.rows.jsonl; echo "VRB teacher order $seed: $(cat $f|wc -l) rows, acc $(python3 -c "print(round(100*$(grep -c '"correct": true' $f)/max($(cat $f|wc -l),1),1))")%"; done
$SPY -m trainer.reader.build_sft_data --example-index $V/derived_pilot60/example_index.json --rollouts $R/vrb_teacher_s0.rollouts.jsonl $R/vrb_teacher_s1.rollouts.jsonl $R/vrb_teacher_s2.rollouts.jsonl --output $R/sft_vrb.jsonl 2>&1 | tail -8
echo "=== VRBench training data ready: $(cat $R/sft_vrb.jsonl | wc -l) rows ==="
