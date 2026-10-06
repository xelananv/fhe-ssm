#!/bin/bash
# SPARE-TIME CONTROL ARM (author, 2026-09-19 23:30Z: "use the spare time to test anything else we've wanted"): the SAME binary, store, prompts, decode rule
# and recorder as the long run, but the STOCK bootstrap table (K = 512, FIDESLIB_BOOT_UNIFORM_EXT=0). Like-for-like pair with ticks 0..n of arm LONG
# (one build, one box, same inputs): does the table change the LOGIT error? Also: /proc/PID/smaps snapshots after requests 1, 2, 3 (the +512 kB / +1,024 kB
# host-memory step at request 2 seen in both earlier arms: which mapping grows?).
# usage: ctrl512_pod.sh "<stop-at UTC>"   e.g. "2026-09-20 00:00:30"
STOP_AT="$1"; ARMN=CTRL512; L=/root/demo/ctrl512.log; u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
u "=== control arm requested; waiting for the GPUs to be free (the previous arm must have returned) ==="
for i in $(seq 1 180); do pgrep -f "gpu_real_model" > /dev/null || break; sleep 5; done
pgrep -f "gpu_real_model" > /dev/null && { u "a server is still alive after 900 s: NOT starting"; exit 3; }
rm -f /root/demo/.busy
export FIDESLIB_BOOT_UNIFORM_EXT=0 TICKS=40 BOOT_RECORD=1 ARMN=$ARMN
u "launching $ARMN: FIDESLIB_BOOT_UNIFORM_EXT=0 (stock K = 512), TICKS=40, stop file at $STOP_AT"
nohup setsid bash /root/demo/longrun_pod.sh > /root/demo/ctrl512_pod.out 2>&1 < /dev/null &
# stop file at the fixed time: the tick in flight finishes, records complete
( while [ "$(date -u +%s)" -lt "$(date -u -d "$STOP_AT" +%s)" ]; do sleep 5; done; mkdir -p /root/demo/s37/serve_$ARMN/serve; touch /root/demo/s37/serve_$ARMN/serve/stop; echo "$(date -u +%H:%M:%SZ) stop file written" >> $L ) &
# smaps snapshots after requests 1, 2, 3, 4
S=/root/demo/s37/serve_$ARMN; SJ=$S/logs/server.jsonl; last=0
while true; do
  grep -q "${ARMN}_DONE" /root/demo/longrun.log 2>/dev/null && break
  n=$(grep -ac '"serve":"served"' $SJ 2>/dev/null || echo 0)
  if [ "$n" -gt "$last" ] && [ "$n" -le 4 ]; then
    PID=$(pgrep -f "gpu_real_model_x" | head -n 1)
    if [ -n "$PID" ]; then cat /proc/$PID/smaps > $S/smaps_after_$n.txt 2>/dev/null; grep -E "VmRSS|VmHWM|RssAnon|VmData" /proc/$PID/status > $S/status_after_$n.txt; u "smaps snapshot after request $n (pid $PID, $(wc -l < $S/smaps_after_$n.txt) lines)"; fi
    last=$n
  fi
  sleep 2
done
u "${ARMN} finished: $(grep -ac '"serve":"served"' $SJ 2>/dev/null) served ticks"
