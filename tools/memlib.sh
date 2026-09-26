#!/bin/bash
# memlib.sh -- shared memory accounting for the guards (sourced, not run). 2026-09-10.
#
# WHY FOOTPRINT, NOT RSS. On macOS `ps rss` counts only pages currently resident and uncompressed. Under
# memory pressure the kernel compresses or swaps a process's pages, and rss FALLS while the process keeps
# every byte. Measured 2026-09-10 (results/s37-impl-20260905/V1_eps/INCIDENT_20260910_footprint.txt): a CPU
# twin at ring 2^16 showed rss 1.86 GB while `footprint -p` reported phys_footprint 15 GB and system swap
# stood at 10.8 of 11.3 GB. The 2026-09-05 incident (three such twins at "1.2-1.3 GB" each, 56 GB of
# pressure) was the same blindness. phys_footprint includes compressed pages; `footprint -p` costs ~0.03 s.
fp_kb() {  # physical footprint of ONE pid in KB (falls back to rss if footprint is unavailable)
  local v
  v=$(footprint -p "$1" 2>/dev/null | awk '/^ *phys_footprint:/{n=$2; u=$3;
        if(u=="GB")printf "%d",n*1048576; else if(u=="MB")printf "%d",n*1024; else if(u=="KB")printf "%d",n; else printf "%d",n/1024; exit}')
  if [ -z "$v" ]; then v=$(ps -o rss= -p "$1" 2>/dev/null | tr -d ' '); fi
  echo "${v:-0}"
}
tree_fp_kb() {  # footprint summed over $1 and all descendants
  local total=0 q=("$1") p kids
  while [ ${#q[@]} -gt 0 ]; do
    p=${q[0]}; q=("${q[@]:1}")
    total=$(( total + $(fp_kb "$p") ))
    kids=$(pgrep -P "$p" 2>/dev/null); for k in $kids; do q+=("$k"); done
  done
  echo $total
}
swap_used_mb() {  # system swap in use, MB (MEMGUARD_TEST_SWAP_USED_MB overrides the reading, guard tests only;
                  # the guard's BASELINE is taken by memguard.sh itself so a test can inject the two separately)
  if [ -n "${MEMGUARD_TEST_SWAP_USED_MB:-}" ]; then echo "$MEMGUARD_TEST_SWAP_USED_MB"; return; fi
  sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1
}
