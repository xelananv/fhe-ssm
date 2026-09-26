#!/bin/bash
# Light memory sampler (one line per minute): the serving process's host RSS / peak RSS and each card's used memory,
# beside the number of requests served so far by the arm that is running. Evidence for "memory does not grow with
# context". Cost: one ps + one nvidia-smi query per minute.
L=/root/demo/mem_sampler.log
echo "# utc | server pid | arm serve dir | served so far | VmRSS kB | VmHWM kB | gpu0 used MiB | gpu1 used MiB" >> $L
while true; do
  P=$(pgrep -f "^/root/fhe-main-demo/build-demo/gpu_real_model(_x)? " | head -1)   # comm is truncated to 15 chars: match the argv start
  if [ -n "$P" ]; then
    SD=$(tr '\0' ' ' < /proc/$P/cmdline | grep -o "\-\-serve [^ ]*" | cut -d' ' -f2)
    SJ="$(dirname "$SD")/logs/server.jsonl"; [ -f "$SJ" ] || SJ=/root/demo/server.log
    N=$(grep -ac '"serve":"served"' "$SJ" 2>/dev/null)
    RSS=$(awk '/VmRSS/{print $2}' /proc/$P/status); HWM=$(awk '/VmHWM/{print $2}' /proc/$P/status)
    G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
    echo "$(date -u +%H:%M:%SZ) | $P | $SD | $N | $RSS | $HWM | $G" >> $L
  fi
  sleep 60
done
