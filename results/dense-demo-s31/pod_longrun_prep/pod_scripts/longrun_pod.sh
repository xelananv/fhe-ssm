#!/bin/bash
# THE LONG RUN: pod-side, STATEFUL, AUTOREGRESSIVE (fidelity_tick.py --free-run): 64 lanes are fed their prompts, then the
# token the encrypted session itself produced, for TICKS ticks; the plaintext reference runs on the same fed sequence, so every
# tick still yields 64 exact FHE-vs-plaintext comparisons. Pod TEST keys (an instrument, not the privacy demo).
# env: TICKS (default 260), ARMN (LONG), BOOT_RECORD=1 to add the bootstrap flight recorder (K3 harness patch + K2 FIDESlib hook).
# Stop early and cleanly: touch /root/demo/s37/serve_$ARMN/serve/stop  (the request in flight finishes; records are complete).
set -u
. /root/demo/longrun.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; L=/root/demo/longrun.log
u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
TICKS=${TICKS:-260}; ARMN=${ARMN:-LONG}; REC=""
[ "${BOOT_RECORD:-0}" = 1 ] && REC="--boot-record --boot-record-secret $KEYS_POD/secret.key"
u "=== $ARMN: free-running [client decode: ${DECODE_ARGS:-greedy}], prompts $PROMPTS_LONG ($(sha256sum $PROMPTS_LONG | cut -c1-16)), TICKS=$TICKS, 64 lanes, flags [$FROZEN_LONG] + [$X2_LONG $REC], FIDESLIB_BOOT_UNIFORM_EXT=$FIDESLIB_BOOT_UNIFORM_EXT, bin $BINX ($(sha256sum $BINX | cut -c1-16)), store $STORE_LONG ==="
cd /root/src
EXTRA_FLAGS="$FROZEN_LONG" LANES_BLOCK=1 FID_LANES=64 ARM=$ARMN XBIN=1 TICKS=$TICKS STORE="$STORE_LONG" PROMPTS="$PROMPTS_LONG" \
  ARM_FLAGS="$X2_LONG $REC" FID_EXTRA_ARGS="--free-run ${DECODE_ARGS:-}" bash $D/91_s37_serve.sh >> $L 2>&1; u "$ARMN rc=$?"
grep -a "VERDICT" /root/demo/s37/serve_$ARMN/TICKS.md 2>/dev/null | cut -c1-300 >> $L
u "${ARMN}_DONE"
