#!/bin/bash
# One line per minute, from BEFORE the first request: the serving process's host memory (RSS, peak, anonymous part, heap
# extent, mapping count) and each card's used memory, beside the requests served so far. Cost: one ps-like read + one
# nvidia-smi query per minute.
L=/root/demo/mem_sampler.log
echo "# utc | pid | serve dir | served | VmRSS kB | VmHWM kB | RssAnon kB | heap kB | mappings | gpu0 MiB | gpu1 MiB" >> $L
while true; do
  P=$(pgrep -f "^/root/fhe-main-demo/build-demo/gpu_real_model(_x)? " | head -1)
  if [ -n "$P" ]; then
    SD=$(tr '\0' ' ' < /proc/$P/cmdline | grep -o "\-\-serve [^ ]*" | cut -d' ' -f2)
    SJ="$(dirname "$SD")/logs/server.jsonl"; [ -f "$SJ" ] || SJ=/root/demo/server.log
    N=$(grep -ac '"serve":"served"' "$SJ" 2>/dev/null)
    RSS=$(awk '/VmRSS/{print $2}' /proc/$P/status); HWM=$(awk '/VmHWM/{print $2}' /proc/$P/status); ANON=$(awk '/RssAnon/{print $2}' /proc/$P/status)
    HEAP=$(awk '/\[heap\]/{split($1,a,"-"); printf "%d", (strtonum("0x" a[2]) - strtonum("0x" a[1]))/1024}' /proc/$P/maps 2>/dev/null); MAPS=$(wc -l < /proc/$P/maps 2>/dev/null)
    G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    echo "$(date -u +%H:%M:%SZ) | $P | $SD | $N | $RSS | $HWM | $ANON | $HEAP | $MAPS | $G" >> $L
  fi
  sleep 60
done
