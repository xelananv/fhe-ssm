#!/bin/bash
# The 64-lane lanes-block store along the long run's level trajectory (periodic encoder; ~11 min last session).
set -u
. /root/demo/longrun.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/store_long.log
export EXTRA_FLAGS="$FROZEN_LONG" BIN="$BINX" LANES_BLOCK=1
echo "$(date -u +%H:%M:%SZ) store build start: flags '$EXTRA_FLAGS' bin $BIN -> $STORE_LONG (K768 env $FIDESLIB_BOOT_UNIFORM_EXT)" >> $L
CELL=store_long NO_TRACE=1 PASSES=1 STOREF="$STORE_LONG" bash $D/90_s37_cell.sh >> $L 2>&1
echo "$(date -u +%H:%M:%SZ) store_long rc=$?" >> $L; ls -la "$STORE_LONG" >> $L 2>&1
grep -a "FIELDS\|boots\|top1\|relLogitErr" /root/demo/s37/store_long/FIELDS.txt 2>/dev/null | cut -c1-400 >> $L
echo STORE_LONG_DONE >> $L
