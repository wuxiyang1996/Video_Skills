#!/usr/bin/env bash
# Monitor one L1 lever (frames8|repass16): resubmit preempted/timed-out shards until every allowlisted
# example has 04_l1_example.json, then derive (frames8 only), answer the fresh 300 with the 235B reader, and compare.
set -uo pipefail
cd /fs/gamma-projects/vlm-robot/Video_Skills
LEVER=${1:?}; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
SPY=.venv-qwen35-serve/bin/python; R=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901; M=$R/full_vh_mm; FRESH=$M/fresh_ids_300.txt
case $LEVER in frames8) AL=$P/allowlists8; RUN=start_0_limit_400;; repass16) AL=$P/allowlists_q300; RUN=start_0_limit_2000;; *) exit 1;; esac
declare -A resub; MAX_RESUB=6
while :; do
  total=0; done_n=0; line=""
  for f in $AL/shard*.txt; do s=$(basename $f .txt | sed 's/shard//'); n=$(grep -c . $f); c=0
    while read -r eid; do [ -n "$eid" ] && [ -f "$P/$LEVER/shard$s/video_holmes/test/$RUN/stages/${eid//:/_}/04_l1_example.json" ] && c=$((c+1)); done < $f
    total=$((total+n)); done_n=$((done_n+c)); line="$line s$s:$c/$n"
    if [ $c -lt $n ] && [ -z "$(squeue -u wuxiyang -h -n vhl1-$LEVER-$s -o %i)" ]; then
      if [ "${resub[$s]:-0}" -lt $MAX_RESUB ]; then resub[$s]=$(( ${resub[$s]:-0} + 1 )); echo "$(date +%m-%d\ %H:%M) shard$s has no job (resubmit #${resub[$s]}); last log tail:"; tail -3 $(ls -t $P/slurm_logs/$LEVER-$s-*.out 2>/dev/null | head -1) 2>/dev/null | cut -c1-300; bash $S/launch_vh_l1_levers.sh $LEVER $s; else echo "$(date +%m-%d\ %H:%M) shard$s exhausted resubmits"; fi
    fi
  done
  echo "$(date +%m-%d\ %H:%M) $LEVER progress $done_n/$total [$line] running=$(squeue -u wuxiyang -h -n $(ls $AL | sed "s/\(.*\)\.txt/vhl1-$LEVER-\1/" | sed 's/shard//' | paste -sd,) -t R -o %i | wc -l)"
  [ $done_n -ge $total ] && break; sleep 600
done
echo "=== $LEVER L1 complete: $done_n examples ==="
if [ $LEVER = frames8 ]; then
  timeout 7200 $SPY scripts/eval/derive_full_question_examples.py --frozen-l1-glob "$P/frames8/shard*/video_holmes/test/$RUN/stages/*/04_l1_example.json" --dataset video_holmes --split test --output-root $P/frames8_q 2>&1 | tail -3
  D=$P/frames8_q/video_holmes/test/derived/stages; kept=0; for d in $D/*/; do id=$(basename $d); id=${id/video_holmes_test_/video_holmes:test:}; id=${id/_q/:q}; if grep -qxF "$id" $FRESH; then kept=$((kept+1)); else rm -rf "$d"; fi; done; echo "derived kept (fresh): $kept"
  GLOB="$D/*/04_l1_example.json"
else
  GLOB="$P/repass16/shard*/video_holmes/test/$RUN/stages/*/04_l1_example.json"
fi
OUT=$M/lever_${LEVER}_fresh300
timeout 14400 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$GLOB" --indices-from all --conditions direct --example-ids $FRESH --workers 3 --timeout-s 300 \
  --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $OUT.json > $OUT.log 2>&1
r=$(cat $OUT.rows.jsonl | wc -l); c=$(grep -c '"correct": true' $OUT.rows.jsonl)
echo; echo "=== L1 LEVER $LEVER (fresh 300, 235B reader) ==="; echo "$OUT: $r rows, accuracy $(python3 -c "print(round(100*$c/max($r,1),1))")%, errors=$(grep -c '"error"' $OUT.rows.jsonl)"
echo "##### $LEVER vs ours (41.0)"; python3 $S/boot_pair.py $M/vl235b_text_1837.rows.jsonl $OUT.rows.jsonl ours $LEVER
echo "##### $LEVER vs human narrative rows (53.3)"; python3 $S/boot_pair.py $M/ceiling_segonly_fresh300.rows.jsonl $OUT.rows.jsonl human $LEVER | tail -1
