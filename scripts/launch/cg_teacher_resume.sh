#!/usr/bin/env bash
# Resume the CG in-domain teacher passes after the OpenRouter top-up (catalog + subtitles cached). Checks the balance before each pass and logs spend.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; SPY=.venv-qwen35-serve/bin/python; T=/fs/nexus-scratch/wuxiyang/cg_train_l1; R=/fs/nexus-scratch/wuxiyang/reader_train
export OPENROUTER_PROVIDER_ORDER=Alibaba; L="$T/narr_px_plus/stages/*/04_l1_example.json"
bal() { $SPY - <<'PY'
import sys; sys.path.insert(0,'/fs/gamma-projects/vlm-robot'); import keys, requests
k=[v for n,v in vars(keys).items() if 'openrouter' in n.lower() and isinstance(v,str)][0]
d=requests.get('https://openrouter.ai/api/v1/credits',headers={'Authorization':f'Bearer {k}'},timeout=30).json().get('data',{}); print(round(d.get('total_credits',0)-d.get('total_usage',0),2))
PY
}
for seed in 0 1 2; do b0=$(bal); echo "$(date +%m-%d\ %H:%M) balance before order $seed: \$$b0"
  if python3 -c "import sys; sys.exit(0 if float('$b0')>25 else 1)"; then :; else echo "balance too low; stopping before order $seed"; break; fi
  extra=""; [ $seed -gt 0 ] && extra="--shuffle-options $seed"
  timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --conditions direct --rationale $extra --example-ids $T/train_ids.txt --workers 6 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $R/cg_teacher_s$seed.rollouts.jsonl --output $R/cg_teacher_s$seed.json > $R/cg_teacher_s$seed.log 2>&1
  f=$R/cg_teacher_s$seed.rows.jsonl; b1=$(bal); echo "CG teacher order $seed: $(cat $f|wc -l) rows, errors $(grep -c '"error"' $f), acc $(python3 -c "print(round(100*$(grep -c '"correct": true' $f)/max($(cat $f|wc -l),1),1))")%, spent \$$(python3 -c "print(round($b0-$b1,2))"), balance \$$b1"; done
$SPY -m trainer.reader.build_sft_data --example-index $T/narr_px_plus/example_index.json --rollouts $R/cg_teacher_s0.rollouts.jsonl $R/cg_teacher_s1.rollouts.jsonl $R/cg_teacher_s2.rollouts.jsonl --output $R/sft_cg.jsonl 2>&1 | tail -8
echo "=== CG training data ready: $(cat $R/sft_cg.jsonl | wc -l) rows ==="
