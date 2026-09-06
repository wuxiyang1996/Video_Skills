#!/usr/bin/env bash
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
SPY=.venv-qwen35-serve/bin/python; C=dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/cg_qa; P=/fs/nexus-scratch/wuxiyang/vh_l1_levers; G=/fs/gamma-projects/vlm-robot/datasets/CG-Bench/cgbench.json
while :; do d=1; for n in cg_direct_rationale_237 cg_graph2_237; do if [ "$(cat $C/$n.rows.jsonl 2>/dev/null|wc -l)" -lt 237 ] && pgrep -f "$n.json" >/dev/null; then d=0; fi; done; [ $d = 1 ] && break; sleep 300; done
echo "rows: rationale $(cat $C/cg_direct_rationale_237.rows.jsonl | wc -l), graph2 $(cat $C/cg_graph2_237.rows.jsonl | wc -l)"
$SPY scripts/eval/grounded_accuracy.py --gold cgbench --eval-jsonl $G --rollouts $C/cg_direct_rationale_237.rollouts.jsonl --l1-index $P/cg_v1_index_237.json --cite prose | grep -v per_question
$SPY scripts/eval/grounded_accuracy.py --gold cgbench --eval-jsonl $G --rollouts $C/cg_graph2_237.rollouts.jsonl --l1-index $P/cg_v1_index_237.json --cite chain | grep -v per_question
python3 - <<'PY'
import json,random
C='dataset_clip_wrapper/output/l2_paper_cg_vh_20260901/cg_qa'
def load(n): return {json.loads(l)['example_id']:json.loads(l) for l in open(f'{C}/{n}.grounded.jsonl')}
r=load('cg_direct_rationale_237.rollouts'); g=load('cg_graph2_237.rollouts'); ids=sorted(set(r)&set(g)); m=len(ids); rng=random.Random(5)
crit={'correct':lambda x:x['correct'],'correct & any cited clue hit':lambda x:x['correct'] and x['step_recall']>0,
      'correct & clue recall>=0.5':lambda x:x['correct'] and x['step_recall']>=0.5,'correct & citation precision>=0.5':lambda x:x['correct'] and x['cited']>0 and x['citation_precision']>=0.5}
print(f'=== CG-BENCH 237: reasoning-grounded accuracy, direct+rationale (prose citations) vs graph2 (chain citations), paired n={m} ===')
for k,f in crit.items():
    a=[int(bool(f(r[i]))) for i in ids]; b=[int(bool(f(g[i]))) for i in ids]; d=[y-x for x,y in zip(a,b)]
    bs=sorted(100*sum(d[rng.randrange(m)] for _ in range(m))/m for _ in range(10000))
    print(f'{k:36s} rationale {100*sum(a)/m:5.1f}  graph2 {100*sum(b)/m:5.1f}  diff {100*sum(d)/m:+5.1f} [{bs[250]:+.1f}, {bs[9750]:+.1f}]')
print('with any citation: rationale %d graph2 %d'%(sum(1 for i in ids if r[i]['cited']),sum(1 for i in ids if g[i]['cited'])))
PY
