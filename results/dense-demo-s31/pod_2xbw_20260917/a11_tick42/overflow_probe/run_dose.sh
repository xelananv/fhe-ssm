#!/bin/bash
# Probe B, soundness control + dose-response with hand-built wide secrets. One process at a time, each under
# tools/memguard.sh 6, ring 2^13, caffeinate so the Mac cannot idle-sleep mid-run.
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_dose.sh
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe
B=harness/build/boot_overflow_probe
{ echo "start $(date -u +%FT%TZ)"; shasum -a 256 $B harness/boot_overflow_probe.cpp; tools/mem_probe.sh; pgrep -fl "clang|ninja|cmake|harness/build|python" || true; uptime; } > $OUT/dose_loadstate.txt 2>&1
run() { name=$1; shift; echo "== $name: $* $(date -u +%FT%TZ)" >> $OUT/dose_runlog.txt
  caffeinate -i tools/memguard.sh 6 $B "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr; echo "   exit=$? $(date -u +%FT%TZ)" >> $OUT/dose_runlog.txt; }
run sound_r13_stock    --log-ring 13 --mode stock --trials 40
run sound_r13_hand_a1  --log-ring 13 --mode hand --widths 1 --trials 40
run dose_r13_F10       --log-ring 13 --mode hand --widths 8,9,7,10,6,5 --trials-list 300,300,300,150,200,100
{ echo "end $(date -u +%FT%TZ)"; tools/mem_probe.sh; uptime; } >> $OUT/dose_loadstate.txt 2>&1
