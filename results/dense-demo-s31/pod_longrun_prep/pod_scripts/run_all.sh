#!/bin/bash
# The whole pod-side sequence after bring-up (build with the patches, artifacts staged, /root/demo/longrun.env in place).
# Every step logs to /root/demo/run_all.log and is skipped if its marker exists, so the script can be re-run after a fix.
#   1 keygen (pod test keys)   2 store   3 sampler   4 LONG run
# env: TICKS for the long run; BOOT_RECORD=1 for the flight recorder; CAP="YYYY-MM-DD HH:MM:SS" (UTC) = the stop time
set -u
. /root/demo/longrun.env; D=/root/src/results/dense-demo-s31/pod_tools/demo; S=/root/demo; L=$S/run_all.log; u() { echo "$(date -u +%H:%M:%SZ) $*" >> $L; }
u "=== run_all: K768 env $FIDESLIB_BOOT_UNIFORM_EXT, flags [$FROZEN_LONG], bin $(sha256sum $BINX | cut -c1-16) ==="
if [ ! -s $KEYS_POD/secret.key ]; then u "1 keygen"; BIN=$BINX LANES_BLOCK=1 bash $D/60_fidelity_gate.sh keygen-only >> $L 2>&1; u "keygen rc=$?"; fi
if ! grep -q "store_long rc=0" $S/store_long.log 2>/dev/null; then u "2 store"; bash $S/store_long.sh; u "store: $(grep -a 'store_long rc' $S/store_long.log | tail -1)"; fi
grep -q "store_long rc=0" $S/store_long.log || { u "STORE FAILED: stopping before the long run"; exit 2; }
pgrep -f "mem_sample[r].sh" >/dev/null || { nohup setsid bash $S/mem_sampler.sh > /dev/null 2>&1 < /dev/null & u "3 sampler started"; }
[ -n "${CAP:-}" ] && { nohup setsid bash $S/cap_at.sh "$CAP" LONG > /dev/null 2>&1 < /dev/null & u "cap armed at $CAP UTC"; }
u "4 LONG run"; bash $S/longrun_pod.sh; u "RUN_ALL_DONE"
