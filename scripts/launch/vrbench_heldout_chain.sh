#!/usr/bin/env bash
# Held-out confirmation of reasoning-grounded accuracy. Criterion fixed before any result: grounded = correct AND citation precision >= 0.5
# (also report accuracy and correct AND any gold hit). Monitor/resubmit L1 shards -> derive all questions -> graph2 + cited rationale -> score.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; P=/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1; H=$P/heldout60; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad
E=/fs/gamma-projects/vlm-robot/datasets/VRBench/VRBench_eval.jsonl; RUN=vrbench/start_0_limit_960
declare -A resub
while :; do total=0; done_n=0; line=""
  for s in 0 1 2 3 4 5; do n=$(grep -c . $H/allowlists/shard$s.txt); c=0
    while read -r eid; do [ -n "$eid" ] && [ -f "$H/shard$s/$RUN/stages/$(echo "$eid" | sed -E "s/[^A-Za-z0-9_.-]+/_/g")/04_l1_example.json" ] && c=$((c+1)); done < $H/allowlists/shard$s.txt
    total=$((total+n)); done_n=$((done_n+c)); line="$line s$s:$c/$n"
    if [ $c -lt $n ] && [ -z "$(squeue -u wuxiyang -h -n vrbh-$s -o %i)" ]; then
      if [ "${resub[$s]:-0}" -lt 6 ]; then resub[$s]=$(( ${resub[$s]:-0} + 1 )); echo "$(date +%m-%d\ %H:%M) shard$s no job -> resubmit #${resub[$s]}"; bash $S/launch_vrbench_heldout.sh $s; fi
    fi
  done
  echo "$(date +%m-%d\ %H:%M) vrbench heldout L1 $done_n/$total [$line]"; [ $done_n -ge $total ] && break; sleep 900
done
timeout 7200 $SPY scripts/eval/derive_full_question_examples.py --frozen-l1-glob "$H/shard*/$RUN/stages/*/04_l1_example.json" --dataset vrbench --split train --output-root $H/derived 2>&1 | tail -3
IDX=$H/derived/example_index.json; L="$H/derived/vrbench/train/derived/stages/*/04_l1_example.json"; python3 -c "import json; idx=json.load(open('$IDX')); open('$H/all_ids.txt','w').write('\n'.join(idx)+'\n'); print('questions',len(idx))"
M=$H/measure; mkdir -p $M
nohup timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --highlight-from bm25 --conditions graph2 --rank-probabilities --rank-margin 0 --frames-per-clip 0 --example-ids $H/all_ids.txt --workers 3 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --probe-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $M/graph2.rollouts.jsonl --output $M/graph2.json > $M/graph2.log 2>&1 &
nohup timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --highlight-from bm25 --conditions direct --rationale --example-ids $H/all_ids.txt --workers 3 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $M/rationale.rollouts.jsonl --output $M/rationale.json > $M/rationale.log 2>&1 &
nohup timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --highlight-from bm25 --conditions direct --example-ids $H/all_ids.txt --workers 3 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --output $M/direct.json > $M/direct.log 2>&1 &
N=$(grep -c . $H/all_ids.txt)
while :; do d=1; for n in graph2 rationale direct; do if [ "$(cat $M/$n.rows.jsonl 2>/dev/null|wc -l)" -lt $N ] && pgrep -f "measure_answer_chain.*$M/$n.json" >/dev/null; then d=0; fi; done; [ $d = 1 ] && break; sleep 600; done
$SPY scripts/eval/grounded_accuracy.py --eval-jsonl $E --rollouts $M/graph2.rollouts.jsonl --l1-index $IDX --cite chain | grep -v per_question
$SPY scripts/eval/grounded_accuracy.py --eval-jsonl $E --rollouts $M/rationale.rollouts.jsonl --l1-index $IDX --cite prose | grep -v per_question
python3 - <<'PY'
import json,random
H='/fs/nexus-scratch/wuxiyang/vrbench_pilot_v1/heldout60/measure'
def load(n): return {json.loads(l)['example_id']:json.loads(l) for l in open(f'{H}/{n}.rollouts.grounded.jsonl')}
r=load('rationale'); g=load('graph2'); ids=sorted(set(r)&set(g)); m=len(ids); rng=random.Random(2026)
crit=[('PRE-REGISTERED: correct & citation precision>=0.5',lambda x:x['correct'] and x['timed'] and x['cited']>0 and x['citation_precision']>=0.5),
      ('accuracy',lambda x:x['correct']),('correct & any gold step hit',lambda x:x['correct'] and x['timed'] and x['step_recall']>0),
      ('correct & step recall>=0.5',lambda x:x['correct'] and x['timed'] and x['step_recall']>=0.5)]
print(f'=== VRBench HELD-OUT 60 videos (61-120 by duration), paired n={m}: graph2 − cited direct ===')
for k,f in crit:
    a=[int(bool(f(r[i]))) for i in ids]; b=[int(bool(f(g[i]))) for i in ids]; d=[y-x for x,y in zip(a,b)]
    bs=sorted(100*sum(d[rng.randrange(m)] for _ in range(m))/m for _ in range(10000))
    print(f'{k:48s} direct {100*sum(a)/m:5.1f}  graph2 {100*sum(b)/m:5.1f}  diff {100*sum(d)/m:+5.1f} [{bs[250]:+.1f}, {bs[9750]:+.1f}]')
PY
f=$M/direct.rows.jsonl; echo "plain direct accuracy: $(python3 -c "
import json; rows=[json.loads(l) for l in open('$f')]; print(round(100*sum(1 for r in rows if r.get('correct'))/len(rows),1), len(rows))")"
