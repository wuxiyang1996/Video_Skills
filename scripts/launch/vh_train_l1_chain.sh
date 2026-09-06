#!/usr/bin/env bash
# Monitor/resubmit the train L1 shards until every allowlisted video has 04_l1_example.json.
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills; P=/fs/nexus-scratch/wuxiyang/vh_train_l1; S=/tmp/claude-17237/-fs-gamma-projects-vlm-robot/373c29f6-2b6a-4ecf-8d7c-76c68a83d10a/scratchpad; RUN=video_holmes/train/start_0_limit_300
declare -A resub
while :; do total=0; done_n=0; line=""
  for s in 0 1 2 3 4 5 6 7; do n=$(grep -c . $P/allowlists/shard$s.txt); c=0
    while read -r eid; do [ -n "$eid" ] && [ -f "$P/shard$s/$RUN/stages/${eid//:/_}/04_l1_example.json" ] && c=$((c+1)); done < $P/allowlists/shard$s.txt
    total=$((total+n)); done_n=$((done_n+c)); line="$line s$s:$c/$n"
    if [ $c -lt $n ] && [ -z "$(squeue -u wuxiyang -h -n vhtr-$s -o %i)" ]; then
      if [ "${resub[$s]:-0}" -lt 8 ]; then resub[$s]=$(( ${resub[$s]:-0} + 1 )); echo "$(date +%m-%d\ %H:%M) shard$s no job -> resubmit #${resub[$s]}"; bash $S/launch_vh_train_l1.sh $s; fi
    fi
  done
  echo "$(date +%m-%d\ %H:%M) VH train L1 $done_n/$total [$line] running=$(squeue -u wuxiyang -h -t R -o %j | grep -c vhtr-)"; [ $done_n -ge $total ] && break; sleep 900
done
echo "=== VH TRAIN L1 COMPLETE: $done_n videos ==="
