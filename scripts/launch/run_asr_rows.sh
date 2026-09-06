#!/usr/bin/env bash
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm
S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
$SPY scripts/eval/build_asr_rows_catalog.py --example-index dataset_clip_wrapper/output/vh_full_questions_v1/example_index.json --example-ids $M/fresh_ids_300.txt --asr-dir $P/asr/whisper --output-root $P/asr_rows | tail -4
OUT=$M/lever_asr_rows_fresh300
timeout 14400 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$P/asr_rows/stages/*/04_l1_example.json" --indices-from all --conditions direct --example-ids $M/fresh_ids_300.txt --workers 3 --timeout-s 300 \
  --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $OUT.json > $OUT.log 2>&1
r=$(cat $OUT.rows.jsonl | wc -l); c=$(grep -c '"correct": true' $OUT.rows.jsonl)
echo "=== OUR CLIPS + DIALOGUE ROWS (fresh 300) ==="; echo "$r rows, accuracy $(python3 -c "print(round(100*$c/max($r,1),1))")%, errors=$(grep -c '"error"' $OUT.rows.jsonl)"
echo "##### clips+dialogue vs ours (41.0)"; python3 $S/boot_pair.py $M/vl235b_text_1837.rows.jsonl $OUT.rows.jsonl ours asr_rows
echo "##### narr_px_plus (47.0) vs clips+dialogue"; python3 $S/boot_pair.py $OUT.rows.jsonl $M/lever_narr_px_plus_fresh300.rows.jsonl asr_rows narr_px_plus | tail -1
