#!/bin/bash
# One line per launched point (run_point.py): run, Slurm job and state, phase from the
# logs, whether the launcher still follows it, and result files collected.
# Usage: K3VR_WORK=/path/to/work bash examples/vr_k3/status.sh
K=${K3VR_WORK:-.}
for R in "$K"/runs/*/; do
  [ -d "$R" ] || continue
  R=${R%/}
  job=$(grep -o '"slurm_job_id": "[0-9]*"' "$R/launch.log" 2>/dev/null | grep -o '[0-9]*' | tail -1)
  [ -z "$job" ] && { echo "$(basename "$R") no-job $(grep -m1 ERROR "$R/launch.log" | cut -c1-80)"; continue; }
  state=$(sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  log=
  for f in "$R"/srt-slurm-2*/outputs/"$job"/logs/sweep_"$job".log "$R"/LOGS/sweep_"$job".log; do
    [ -f "$f" ] && { log=$f; break; }
  done
  phase=queued
  if [ -n "$log" ]; then
    phase=starting
    grep -q "Server is healthy" "$log" && phase=healthy
    grep -q "Starting benchmark" "$log" && phase=benchmark
    grep -q "Sweep failed\|Benchmark failed" "$log" && phase=FAILED
    grep -q "Sweep completed\|completed successfully" "$log" && phase="done"
  fi
  alive=$(kill -0 "$(cat "$R/launch.pid" 2>/dev/null)" 2>/dev/null && echo launcher-up || echo launcher-exited)
  res=0
  for f in "$R"/*_conc*.json; do
    case ${f##*/} in agg_*) ;; *) [ -f "$f" ] && res=$((res + 1)) ;; esac
  done
  printf "%-78s %-8s %-10s %-9s %s results=%s\n" "$(basename "$R" | cut -c1-78)" "$job" "$state" "$phase" "$alive" "$res"
done
