#!/usr/bin/env bash
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm
S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad; OUT=$M/full_narr_px_plus_noptr_1837
timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$P/narr_px_plus_full/stages/*/04_l1_example.json" --indices-from all --conditions direct \
  --example-ids $P/all_ids_1837.txt --workers 4 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $OUT.json > $OUT.log 2>&1
r=$(cat $OUT.rows.jsonl | wc -l); c=$(grep -c '"correct": true' $OUT.rows.jsonl)
echo; echo "=== FULL VH TEST: narr_px_plus catalog, NO pointer (the fresh-300-selected configuration) ==="; echo "$r rows, accuracy $(python3 -c "print(round(100*$c/max($r,1),1))")%, errors=$(grep -c '"error"' $OUT.rows.jsonl)"
echo "##### no-pointer vs pointer (same catalog)"; python3 $S/boot_pair.py $M/full_narr_px_plus_1837.rows.jsonl $OUT.rows.jsonl pointer noptr
echo "##### no-pointer vs ours 42.6"; python3 $S/boot_pair.py $M/vl235b_text_1837.rows.jsonl $OUT.rows.jsonl ours noptr | tail -1
