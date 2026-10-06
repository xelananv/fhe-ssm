#!/bin/bash
# usage: cap_at.sh "2026-09-20 14:30:00" [ARM=LONG]   -- at that UTC time write the arm's stop file (the server finishes the
# request in flight and exits; the driver exits; the arm's records are complete). Run with nohup setsid.
set -u
T=$(date -u -d "$1" +%s); ARMN=${2:-LONG}; L=/root/demo/cap.log
echo "$(date -u +%H:%M:%SZ) armed: $ARMN stops at $1 UTC" >> $L
while [ "$(date -u +%s)" -lt "$T" ]; do grep -q "${ARMN}_DONE" /root/demo/longrun.log 2>/dev/null && exit 0; sleep 20; done
N=$(grep -ac '"serve":"served"' /root/demo/s37/serve_$ARMN/logs/server.jsonl 2>/dev/null)
touch /root/demo/s37/serve_$ARMN/serve/stop; echo "$(date -u +%H:%M:%SZ) $ARMN capped after $N served ticks" >> $L
