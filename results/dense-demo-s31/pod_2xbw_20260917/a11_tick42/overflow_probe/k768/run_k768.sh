#!/bin/bash
# K = 768 (g_coefficientsUniformExt) vs stock K = 512 in the library's own CPU bootstrap: planted-overflow sweeps and
# un-planted baselines. ONE binary per library variant, the arm chosen at run time by environment variables:
#   A = harness/build/boot_overflow_probe_k768   (vendor/install-k768,  openfhe_k768_env.patch)
#         OPENFHE_BOOT_UNIFORM_EXT unset = stock table (K = 512), =1 = Ext table (K = 768)
#   B = harness/build/boot_overflow_probe_k768b  (vendor/install-k768b, + openfhe_k768_split_variantB.patch)
#         additionally OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1 = the factor 3 of K = 768 folded into CoeffsToSlots
# One process at a time, each under tools/memguard.sh 6, rings 2^13 / 2^12, caffeinate -i so the Mac
# cannot idle-sleep mid-run. Same parameters as the previous planted sweeps (../planted_runlog.txt):
# --mode stock, --plant-index 100, --trials 3, FLEXIBLEAUTO, scaling 59, first modulus 60, level budget {3,3}, depth 23.
# (The `main` battery was run with the first version of this script, whose run() had modes 0 and 1 only; the commands it
# issued are in runlog_main.txt.)
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768/run_k768.sh <battery>
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/k768
BA=harness/build/boot_overflow_probe_k768
BB=harness/build/boot_overflow_probe_k768b
BAT=${1:-main}
RUNLOG=$OUT/runlog_$BAT.txt
LOAD=$OUT/loadstate_$BAT.txt
{ echo "start $(date -u +%FT%TZ)"; shasum -a 256 $BA $BB harness/boot_overflow_probe.cpp vendor/install-k768/lib/libOPENFHEpke.1.5.1.dylib vendor/install-k768/lib/libOPENFHEcore.1.5.1.dylib vendor/install-k768b/lib/libOPENFHEpke.1.5.1.dylib vendor/install-k768b/lib/libOPENFHEcore.1.5.1.dylib
  tools/mem_probe.sh; pgrep -fl "clang|ninja|cmake|harness/build|python" || true; uptime; } > $LOAD 2>&1
# run <name> <mode> <probe args...>   mode: 0 = A, switch off | 1 = A, Ext | 2 = B, Ext + split | 3 = B, Ext, no split | 4 = B, switch off
run() { name=$1; mode=$2; shift 2
  case $mode in
    0) echo "== $name: (no env) $BA $* $(date -u +%FT%TZ)" >> $RUNLOG
       env -u OPENFHE_BOOT_UNIFORM_EXT -u OPENFHE_BOOT_UNIFORM_EXT_SPLIT caffeinate -i tools/memguard.sh 6 $BA "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr ;;
    1) echo "== $name: OPENFHE_BOOT_UNIFORM_EXT=1 $BA --k-bound 768 $* $(date -u +%FT%TZ)" >> $RUNLOG
       env -u OPENFHE_BOOT_UNIFORM_EXT_SPLIT OPENFHE_BOOT_UNIFORM_EXT=1 caffeinate -i tools/memguard.sh 6 $BA --k-bound 768 "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr ;;
    2) echo "== $name: OPENFHE_BOOT_UNIFORM_EXT=1 OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1 $BB --k-bound 768 $* $(date -u +%FT%TZ)" >> $RUNLOG
       OPENFHE_BOOT_UNIFORM_EXT=1 OPENFHE_BOOT_UNIFORM_EXT_SPLIT=1 caffeinate -i tools/memguard.sh 6 $BB --k-bound 768 "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr ;;
    3) echo "== $name: OPENFHE_BOOT_UNIFORM_EXT=1 $BB --k-bound 768 $* $(date -u +%FT%TZ)" >> $RUNLOG
       env -u OPENFHE_BOOT_UNIFORM_EXT_SPLIT OPENFHE_BOOT_UNIFORM_EXT=1 caffeinate -i tools/memguard.sh 6 $BB --k-bound 768 "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr ;;
    4) echo "== $name: (no env) $BB $* $(date -u +%FT%TZ)" >> $RUNLOG
       env -u OPENFHE_BOOT_UNIFORM_EXT -u OPENFHE_BOOT_UNIFORM_EXT_SPLIT caffeinate -i tools/memguard.sh 6 $BB "$@" > $OUT/$name.jsonl 2> $OUT/$name.stderr ;;
  esac
  echo "   exit=$? $(date -u +%FT%TZ) lines=$(wc -l < $OUT/$name.jsonl | tr -d ' ')" >> $RUNLOG; }

case $BAT in
main)   # ring 2^13, default correction factor (F = 10): the earlier main configuration
  run ctrl_r13_F10_planted       0 --log-ring 13 --mode stock --plant 500:560:1 --trials 3 --plant-index 100
  run ctrl_r13_F10_base          0 --log-ring 13 --mode stock --trials 40 --err-by-absI 1
  run ext_r13_F10_base           1 --log-ring 13 --mode stock --trials 40 --err-by-absI 1
  run ext_r13_F10_planted_old    1 --log-ring 13 --mode stock --plant 500:560:1 --trials 3 --plant-index 100
  run ext_r13_F10_planted_old2   1 --log-ring 13 --mode stock --plant 565:600:5 --trials 3 --plant-index 100
  run ext_r13_F10_planted_edge   1 --log-ring 13 --mode stock --plant 760:830:1 --trials 3 --plant-index 100
  ;;
splitB) # VARIANT B: does folding the 3 of K = 768 into CoeffsToSlots remove the error that grows with |I|? (ring 2^13 F = 10, ring 2^12 F = 11)
  run B_split_r13_F10_base          2 --log-ring 13 --mode stock --trials 40 --err-by-absI 1
  run B_split_r13_F10_planted_wide  2 --log-ring 13 --mode stock --plant 100:760:20 --trials 3 --plant-index 100
  run B_split_r13_F10_planted_edge  2 --log-ring 13 --mode stock --plant 765:805:1 --trials 3 --plant-index 100
  run B_split_r12_F11_base          2 --log-ring 12 --mode stock --trials 40 --err-by-absI 1
  run B_split_r12_F11_planted_wide  2 --log-ring 12 --mode stock --plant 100:760:60 --trials 3 --plant-index 100
  run B_nosplit_r13_F10_base        3 --log-ring 13 --mode stock --trials 10 --err-by-absI 1
  run B_ctrl_r13_F10_base           4 --log-ring 13 --mode stock --trials 10 --err-by-absI 1
  ;;
extra)  # variant A robustness: the whole fitted range, F = 7 (the library's default at N = 2^17 / 2^16 slots), ring 2^12 (F = 11), the negative side, a second-half coefficient
  run ext_r13_F10_planted_wide   1 --log-ring 13 --mode stock --plant 100:760:20 --trials 3 --plant-index 100
  run ctrl_r13_F10_planted_wide  0 --log-ring 13 --mode stock --plant 100:500:20 --trials 3 --plant-index 100
  run ctrl_r13_F07_base          0 --log-ring 13 --mode stock --corr 7 --trials 40 --err-by-absI 1
  run ext_r13_F07_base           1 --log-ring 13 --mode stock --corr 7 --trials 40 --err-by-absI 1
  run ext_r13_F07_planted_edge   1 --log-ring 13 --mode stock --corr 7 --plant 765:805:1 --trials 3 --plant-index 100
  run ext_r13_F07_planted_old    1 --log-ring 13 --mode stock --corr 7 --plant 510:600:10 --trials 3 --plant-index 100
  run ctrl_r12_F11_base          0 --log-ring 12 --mode stock --trials 40 --err-by-absI 1
  run ext_r12_F11_base           1 --log-ring 12 --mode stock --trials 40 --err-by-absI 1
  run ext_r12_F11_planted_wide   1 --log-ring 12 --mode stock --plant 100:760:60 --trials 3 --plant-index 100
  run ext_r12_F11_planted_edge   1 --log-ring 12 --mode stock --plant 765:800:1 --trials 3 --plant-index 100
  run ext_r13_F10_planted_neg    1 --log-ring 13 --mode stock --plant -806:-766:2 --trials 2 --plant-index 100
  run ext_r13_F10_planted_imag   1 --log-ring 13 --mode stock --plant 765:805:5 --trials 3 --plant-index 6000
  ;;
*) echo "unknown battery $BAT" >&2; exit 2 ;;
esac
{ echo "end $(date -u +%FT%TZ)"; tools/mem_probe.sh; uptime; } >> $LOAD 2>&1
