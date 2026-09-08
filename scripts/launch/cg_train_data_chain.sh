#!/usr/bin/env bash
# CG in-domain training data from already-built L1 (161 videos, 1,080 contamination-free questions): subtitles -> narrative catalog -> teacher rationales x3 orders -> verified SFT rows.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; SPY=.venv-qwen35-serve/bin/python; T=/fs/nexus-scratch/wuxiyang/cg_train_l1; R=/fs/nexus-scratch/wuxiyang/reader_train
export OPENROUTER_PROVIDER_ORDER=Alibaba
python3 - <<'PY'
import json,os,re
T='/fs/nexus-scratch/wuxiyang/cg_train_l1'; os.makedirs(T+'/asr',exist_ok=True); idx=json.load(open(T+'/train_index.json'))
def ts(s):
    h,m,rest=s.split(':'); sec,ms=rest.split(','); return int(h)*3600+int(m)*60+int(sec)+int(ms)/1000
n=0
for eid,m in idx.items():
    out=f"{T}/asr/{m['video_id']}.json"
    if os.path.exists(out): continue
    ex=json.load(open(m['path'])); segs=[]
    for t in (ex.get('video') or {}).get('subtitle_tracks') or []:
        p=t.get('path')
        if not p or not os.path.exists(p): continue
        for block in re.split(r'\n\s*\n', open(p,encoding='utf-8',errors='ignore').read()):
            lines=[l.strip() for l in block.strip().splitlines() if l.strip()]; tl=[l for l in lines if '-->' in l]
            if not tl: continue
            a,b=[x.strip() for x in tl[0].split('-->')[:2]]
            try: segs.append({'start_s':round(ts(a),2),'end_s':round(ts(b.split()[0]),2),'text':' '.join(lines[lines.index(tl[0])+1:])})
            except Exception: pass
    json.dump({'video_id':m['video_id'],'language':'srt','segments':segs},open(out,'w'),ensure_ascii=False); n+=1
print('subtitle files written', n)
PY
timeout 14400 $SPY scripts/eval/build_narrative_catalog.py --example-index $T/train_index.json --example-ids $T/train_ids.txt --output-root $T/narr_px_plus --window-s 60 --frames-per-window 16 --asr-dir $T/asr --no-clip-text --keep-clips --slim --workers 8 2>&1 | tail -2
CIDX=$T/narr_px_plus/example_index.json; L="$T/narr_px_plus/stages/*/04_l1_example.json"; echo "CG train catalog examples: $(ls -d $T/narr_px_plus/stages/*/ | wc -l)"
for seed in 0 1 2; do extra=""; [ $seed -gt 0 ] && extra="--shuffle-options $seed"
  timeout 28800 $SPY -m scripts.eval.measure_answer_chain --l1-glob "$L" --indices-from all --conditions direct --rationale $extra --example-ids $T/train_ids.txt --workers 6 --timeout-s 300 --skill-model openai/gpt-oss-120b --answer-model qwen/qwen3-vl-235b-a22b-instruct --dump-rollouts $R/cg_teacher_s$seed.rollouts.jsonl --output $R/cg_teacher_s$seed.json > $R/cg_teacher_s$seed.log 2>&1
  f=$R/cg_teacher_s$seed.rows.jsonl; echo "CG teacher order $seed: $(cat $f|wc -l) rows, acc $(python3 -c "print(round(100*$(grep -c '"correct": true' $f)/max($(cat $f|wc -l),1),1))")%"; done
$SPY -m trainer.reader.build_sft_data --example-index $CIDX --rollouts $R/cg_teacher_s0.rollouts.jsonl $R/cg_teacher_s1.rollouts.jsonl $R/cg_teacher_s2.rollouts.jsonl --output $R/sft_cg.jsonl 2>&1 | tail -8
echo "=== CG training data ready: $(cat $R/sft_cg.jsonl | wc -l) rows ==="
