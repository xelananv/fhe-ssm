#!/bin/bash
# A11 (pod-side, teacher-forced): 64 lanes (lane 64 = lane 1) x TICKS ticks on the FROZEN flags + the lanes-block store,
# pod test keys; the per-tick server timing curve is complete even where a reply fails to decrypt (decFailed rows).
set -u
. /root/demo/s37.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/a11.log
u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
FF=$(sed -n 1p /root/demo/FROZEN_FLAGS.txt); PB=$(sed -n 2p /root/demo/FROZEN_FLAGS.txt); XB=0; [ "$PB" = "$BINX" ] && XB=1
ST=${A11_STORE:-/root/demo/store_pbd430a_r17_xd12_lanesblock_canon.bin}
u "=== A11: TICKS=${TICKS:-64} lanes 64, flags [$FF], bin $PB (xbin $XB), store $ST ==="
cd /root/src
EXTRA_FLAGS="$FF" LANES_BLOCK=1 FID_LANES=64 ARM=A11 XBIN=$XB TICKS=${TICKS:-64} STORE="$ST" \
  PROMPTS=/root/src/results/dense-demo-s31/sessions/a11_prompts64_long.txt ARM_FLAGS="" bash $D/91_s37_serve.sh >> $L 2>&1; u "A11 rc=$?"
grep -a "VERDICT" /root/demo/s37/serve_A11/TICKS.md 2>/dev/null | cut -c1-300 >> $L
u "A11_DONE"
