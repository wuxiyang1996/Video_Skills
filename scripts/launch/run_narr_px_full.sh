#!/usr/bin/env bash
# Full Video-Holmes test (1,837 q, 270 videos) with the narr_px_plus catalog, vs the 235B full-test 42.5.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; M=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/full_vh_mm
S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad; IDX=dataset_clip_wrapper/output/vh_full_questions_v1/example_index.json
python3 -c "
import json; idx=json.load(open('$IDX')); open('$P/asr/all_video_ids.txt','w').write('\n'.join(sorted({m['video_id'] for m in idx.values()}))+'\n'); open('$P/all_ids_1837.txt','w').write('\n'.join(idx)+'\n'); print('videos',len({m['video_id'] for m in idx.values()}),'ids',len(idx))"
$SPY scripts/eval/transcribe_videos.py --video-dir /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/videos_cropped --video-ids $P/asr/all_video_ids.txt --out-dir $P/asr/whisper --workers 4 2>&1 | tail -2
echo "ASR cached: $(ls $P/asr/whisper/*.json | wc -l) / 270"
mkdir -p $P/narr_px_plus_full/narratives; cp $P/narr_px/narratives/*.json $P/narr_px_plus_full/narratives/
timeout 14400 $SPY scripts/eval/build_narrative_catalog.py --example-index $IDX --example-ids $P/all_ids_1837.txt --output-root $P/narr_px_plus_full --window-s 30 --frames-per-window 16 --asr-dir $P/asr/whisper --no-clip-text --keep-clips --workers 8 2>&1 | tail -4
echo "narratives cached: $(ls $P/narr_px_plus_full/narratives/*.json | wc -l) / 270; examples: $(ls -d $P/narr_px_plus_full/stages/*/ | wc -l)"
OUT=$M/full_narr_px_plus_1837
timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$P/narr_px_plus_full/stages/*/04_l1_example.json" --indices-from all --conditions direct --workers 6 --timeout-s 300 \
  --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $OUT.json > $OUT.log 2>&1
r=$(cat $OUT.rows.jsonl | wc -l); c=$(grep -c '"correct": true' $OUT.rows.jsonl)
echo; echo "=== FULL VH TEST: narr_px_plus catalog, 235B reader ==="; echo "$r rows, accuracy $(python3 -c "print(round(100*$c/max($r,1),1))")%, errors=$(grep -c '"error"' $OUT.rows.jsonl)   (ours 235B full = 42.5; Gemini-2.5-Pro 45.0)"
echo "##### full test paired vs ours (vl235b_text_1837)"; python3 $S/boot_pair.py $M/vl235b_text_1837.rows.jsonl $OUT.rows.jsonl ours narr_px_plus
