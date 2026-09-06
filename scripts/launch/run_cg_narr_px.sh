#!/usr/bin/env bash
# Generality test: narrative-from-pixels (60-s windows, 16 frames, subtitles) + original clips on the CG-Bench 237-question heldout.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; R=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901; M=$R/cg_qa
S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
timeout 21600 $SPY scripts/eval/build_narrative_catalog.py --example-index $P/cg_v1_index_237.json --example-ids $P/cg_ids_237.txt --output-root $P/cg_narr_px_plus --window-s 60 --frames-per-window 16 --asr-dir $P/cg_asr --no-clip-text --keep-clips --workers 8 2>&1 | tail -4
echo "cg narratives cached: $(ls $P/cg_narr_px_plus/narratives/*.json | wc -l) / 67; examples $(ls -d $P/cg_narr_px_plus/stages/*/ | wc -l)"
OUT=$M/cg_narr_px_plus_237
timeout 21600 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$P/cg_narr_px_plus/stages/*/04_l1_example.json" --indices-from all --highlight-from bm25 --conditions direct \
  --example-ids $P/cg_ids_237.txt --workers 3 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $OUT.json > $OUT.log 2>&1
r=$(cat $OUT.rows.jsonl | wc -l); c=$(grep -c '"correct": true' $OUT.rows.jsonl)
echo; echo "=== CG-BENCH 237 heldout: narr_px_plus catalog vs cg_direct_237 (40.5) ==="; echo "$r rows, accuracy $(python3 -c "print(round(100*$c/max($r,1),1))")%, errors=$(grep -c '"error"' $OUT.rows.jsonl)"
python3 $S/boot_pair.py $M/cg_direct_237.rows.jsonl $OUT.rows.jsonl direct narr_px_plus | tail -2
