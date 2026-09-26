#!/bin/bash
# X2 (pod-side, teacher-forced): the A11 shape (64 lanes, same prompts, same binary / frozen flags / nr store / pod test
# keys) + the plaintext cache. Tick 0 = the misses (each one verified limb for limb against the dense encode under
# --pt-cache-verify; a mismatch is fatal); ticks 1.. = hits. Compare the warm ticks with A11's warm ticks: same binary,
# same box, same load state (R6). Records: /root/demo/s37/serve_<ARM>/ + this log.
set -u
. /root/demo/s37.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/x2.log
u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
FF=$(sed -n 1p /root/demo/FROZEN_FLAGS.txt); PB=$(sed -n 2p /root/demo/FROZEN_FLAGS.txt); XB=0; [ "$PB" = "$BINX" ] && XB=1
ST=${X2_STORE:-/root/demo/store_pbd430a_r17_xd12_lanesblock_canon_nr.bin}
AF=${X2_FLAGS:---pt-cache --pt-cache-verify --pt-cache-max-gb 24}
ARMN=${X2_ARM:-X2}
u "=== $ARMN: TICKS=${TICKS:-4} lanes 64, flags [$FF] + [$AF], bin $PB (xbin $XB), store $ST ==="
cd /root/src
EXTRA_FLAGS="$FF" LANES_BLOCK=1 FID_LANES=64 ARM=$ARMN XBIN=$XB TICKS=${TICKS:-4} STORE="$ST" \
  PROMPTS=/root/src/results/dense-demo-s31/sessions/a11_prompts64_long.txt ARM_FLAGS="$AF" bash $D/91_s37_serve.sh >> $L 2>&1; u "$ARMN rc=$?"
grep -a "VERDICT" /root/demo/s37/serve_$ARMN/TICKS.md 2>/dev/null | cut -c1-300 >> $L
grep -a '"ptCacheTick"\|"fatal"' /root/demo/s37/serve_$ARMN/logs/server.jsonl 2>/dev/null | cut -c1-400 >> $L
u "${ARMN}_DONE"
