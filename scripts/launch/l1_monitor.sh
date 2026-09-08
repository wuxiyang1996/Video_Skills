#!/usr/bin/env bash
# Generic L1 shard monitor: resubmit a shard while any allowlisted example lacks 04_l1_example.json and no job of its name is queued.
# Usage: l1_monitor.sh <root> <run_subdir> <jobtag> <launcher.sh>   e.g. l1_monitor.sh /fs/nexus-scratch/wuxiyang/cg_mini_l1 cg_bench/start_0_limit_1200 cgmini scripts/launch/launch_cg_mini_l1.sh
set -uo pipefail; cd /fs/gamma-projects/vlm-robot/Video_Skills
ROOT=$1; RUN=$2; TAG=$3; LAUNCH=$4; declare -A resub
while :; do total=0; done_n=0; line=""
  for f in $ROOT/allowlists/shard*.txt; do s=$(basename $f .txt | sed 's/shard//'); n=$(grep -c . $f); c=0
    while read -r eid; do [ -n "$eid" ] && [ -f "$ROOT/shard$s/$RUN/stages/${eid//:/_}/04_l1_example.json" ] && c=$((c+1)); done < $f
    total=$((total+n)); done_n=$((done_n+c)); line="$line s$s:$c/$n"
    if [ $c -lt $n ] && [ -z "$(squeue -u wuxiyang -h -n $TAG-$s -o %i)" ]; then
      if [ "${resub[$s]:-0}" -lt 10 ]; then resub[$s]=$(( ${resub[$s]:-0} + 1 )); echo "$(date +%m-%d\ %H:%M) $TAG shard$s no job -> resubmit #${resub[$s]}"; bash $LAUNCH $s; fi
    fi
  done
  echo "$(date +%m-%d\ %H:%M) $TAG L1 $done_n/$total [$line] running=$(squeue -u wuxiyang -h -t R -o %j | grep -c "^$TAG-")"; [ $done_n -ge $total ] && break; sleep 1200
done
echo "=== $TAG L1 COMPLETE: $done_n ==="
