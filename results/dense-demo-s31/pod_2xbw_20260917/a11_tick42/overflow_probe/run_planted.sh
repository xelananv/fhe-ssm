#!/bin/bash
# Probe B, planted-overflow sweeps (stock library KeyGen, uniform ternary). One process at a time, each under
# tools/memguard.sh 6, rings 2^13 / 2^12, caffeinate so the Mac cannot idle-sleep mid-run.
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_planted.sh
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe
B=harness/build/boot_overflow_probe
{ echo "start $(date -u +%FT%TZ)"; shasum -a 256 $B harness/boot_overflow_probe.cpp; tools/mem_probe.sh; pgrep -fl "clang|ninja|cmake|harness/build|python" || true; uptime; } > $OUT/planted_loadstate.txt 2>&1
run() { name=$1; shift; echo "== $name: $* $(date -u +%FT%TZ)" >> $OUT/planted_runlog.txt
  caffeinate -i tools/memguard.sh 6 $B "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr; echo "   exit=$? $(date -u +%FT%TZ)" >> $OUT/planted_runlog.txt; }
run planted_r13_F10_pos   --log-ring 13 --mode stock --plant 500:560:1 --trials 3 --plant-index 100
run planted_r13_F10_neg   --log-ring 13 --mode stock --plant -560:-500:2 --trials 2 --plant-index 100
run planted_r13_F07_pos   --log-ring 13 --mode stock --corr 7 --plant 500:560:1 --trials 3 --plant-index 100
run planted_r12_F11_pos   --log-ring 12 --mode stock --plant 500:560:1 --trials 3 --plant-index 100
run planted_r13_F10_imag  --log-ring 13 --mode stock --plant 510:545:5 --trials 3 --plant-index 6000
{ echo "end $(date -u +%FT%TZ)"; tools/mem_probe.sh; } >> $OUT/planted_loadstate.txt 2>&1
