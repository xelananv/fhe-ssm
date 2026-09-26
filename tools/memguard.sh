#!/bin/bash
# memguard.sh -- run a command under a hard MEMORY FOOTPRINT ceiling and a swap-growth tripwire.
# usage: memguard.sh <limit_gb> <command...>        kills the command's process tree when its physical
#                                                   footprint (compressed pages INCLUDED) exceeds the limit,
#                                                   or when system swap has grown by more than
#                                                   MEMGUARD_SWAP_GROW_GB (default 2) since the start
#        memguard.sh --probe <pid>                   print one JSON line with rss and footprint of a pid tree
# exit: the command's, 137 if killed by the guard, 2 on usage.
#
# HISTORY. v1 (2026-09-05) sampled `ps rss` every 0.25 s. On 2026-09-10 it let a ring-2^16 CPU twin run to a
# 15 GB footprint at 1.86 GB rss while swap hit 10.8 GB (results/s37-impl-20260905/V1_eps/
# INCIDENT_20260910_footprint.txt), because rss does not see compressed or swapped pages. v2 uses
# phys_footprint via /usr/bin/footprint (tools/memlib.sh) and adds the swap tripwire.
# STILL A BACKSTOP, NOT A CAP: the sample period is 0.25 s and a fast allocator overshoots before it dies.
set -u
. "$(cd "$(dirname "$0")" && pwd)/memlib.sh"
if [ "${1:-}" = "--probe" ]; then
  [ -n "${2:-}" ] || { echo "usage: memguard.sh --probe <pid>" >&2; exit 2; }
  printf '{"pid":%s,"rssKB":%s,"footprintKB":%s,"swapUsedMB":%s}\n' "$2" "$(ps -o rss= -p "$2" | tr -d ' ')" "$(tree_fp_kb "$2")" "$(swap_used_mb)"; exit 0
fi
LIM_GB="${1:-}"; shift || true
{ [ -z "$LIM_GB" ] || [ $# -eq 0 ]; } && { echo "usage: memguard.sh <limit_gb> <command...>" >&2; exit 2; }
LIM_KB=$(( LIM_GB * 1048576 ))
SWAP_GROW_MB=$(( ${MEMGUARD_SWAP_GROW_GB:-2} * 1024 ))
SWAP0=${MEMGUARD_TEST_SWAP_BASE_MB:-$(swap_used_mb)}; SWAP0=${SWAP0:-0}   # baseline (test injection: MEMGUARD_TEST_SWAP_BASE_MB)
"$@" &
PID=$!
kill_tree() { pkill -9 -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
# HYBRID SAMPLER. rss is cheap and catches a fast allocator within a sample (v1 killed at 1.33 GB against 1 GB);
# footprint costs ~30 ms per process and is what catches compressed/swapped growth, so it runs every 4th sample
# (about once a second). Either measure over the limit kills.
n=0
while kill -0 "$PID" 2>/dev/null; do
  r=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' '); r=${r:-0}
  if [ $(( n % 4 )) -eq 0 ]; then f=$(tree_fp_kb "$PID"); else f=0; fi
  n=$(( n + 1 ))
  if [ "$r" -gt "$LIM_KB" ] || [ "$f" -gt "$LIM_KB" ]; then
    [ "$f" -eq 0 ] && f=$(tree_fp_kb "$PID")
    echo "{\"memguard\":\"killed\",\"why\":\"$([ "$r" -gt "$LIM_KB" ] && echo rss || echo footprint)\",\"footprintKB\":$f,\"rssKB\":$r,\"limitGB\":$LIM_GB,\"cmd\":\"$*\"}" >&2
    kill_tree; exit 137
  fi
  s=$(swap_used_mb); s=${s:-0}
  if [ $(( s - SWAP0 )) -gt "$SWAP_GROW_MB" ]; then
    echo "{\"memguard\":\"killed\",\"why\":\"swap-growth\",\"swapUsedMB\":$s,\"swapStartMB\":$SWAP0,\"growLimitMB\":$SWAP_GROW_MB,\"footprintKB\":$f,\"cmd\":\"$*\"}" >&2
    kill_tree; exit 137
  fi
  sleep 0.25
done
wait "$PID"; exit $?
