#!/bin/bash
# Probe C: tower-0 dump of the 36 saved session ciphertexts (ring 2^17, LOAD-ONLY: no evaluation keys, no homomorphic
# evaluation), alone, under tools/memguard.sh 8; then the numpy statistics; then the scratch directory is deleted.
# The kit is opened read-only in place (chmod 444 files; nothing is copied, moved or symlinked).
# Run from the repo root:  bash results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe/run_probe_c.sh dump|stats|clean
set -u
OUT=results/dense-demo-s31/pod_2xbw_20260917/a11_tick42/overflow_probe
KIT="$HOME/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z"
SCR=$SCRATCH/overflow_dist
case "${1:-}" in
dump)
  mkdir -p "$SCR" && chmod 700 "$SCR"
  { echo "start $(date -u +%FT%TZ)"; shasum -a 256 harness/build/overflow_dist_dump harness/overflow_dist_dump.cpp; tools/mem_probe.sh; pgrep -fl "clang|ninja|cmake|harness/build" || true; uptime; } > $OUT/probe_c_loadstate.txt 2>&1
  CTS=$(for t in "$KIT"/session_cts/tick_*; do n=${t##*req}; echo "$t/req.$n.0 $t/resp.$n.0"; done)
  # stdout carries only public metadata (ring, q0, towers, Hamming weight, per-ciphertext level/scale); no coefficient is printed
  caffeinate -i tools/memguard.sh 8 harness/build/overflow_dist_dump --ctx "$KIT/keys/cryptocontext.bin" --sec "$KIT/keys/secret.key" \
      --out-dir "$SCR" --cts $CTS > $OUT/probe_c_dump.jsonl 2> $OUT/probe_c_dump.stderr
  echo "dump exit=$? $(date -u +%FT%TZ)" >> $OUT/probe_c_loadstate.txt
  ls -l "$SCR" | awk '{print $1, $5, $9}' | sed "s#$SCR/##" > $OUT/probe_c_scratch_listing.txt ;;
stats)
  .venv/bin/python $OUT/overflow_dist_stats.py --scratch "$SCR" > $OUT/probe_c_stats.stdout 2> $OUT/probe_c_stats.stderr
  echo "stats exit=$? $(date -u +%FT%TZ)" >> $OUT/probe_c_loadstate.txt ;;
dump-ext)
  # EXTENSION beyond the brief's kit: the two earlier verification kits hold ciphertexts under the SAME keys (their MANIFEST.sha256
  # lines for keys/secret.key, public.key and cryptocontext.bin carry the same sha256 as this kit's). Same rules: read-only, alone, guarded.
  mkdir -p "${SCR}_ext" && chmod 700 "${SCR}_ext"
  { echo "dump-ext start $(date -u +%FT%TZ)"; tools/mem_probe.sh; pgrep -fl "clang|ninja|cmake|harness/build" || true; } >> $OUT/probe_c_loadstate.txt 2>&1
  CTS=$(for kit in demo_verification_demo_20260904T183832Z demo_verification_demo_20260918T025113Z; do for t in "$HOME/Documents/fhe-ssm-backup/$kit"/session_cts/tick_*; do n=${t##*req}; echo "$t/req.$n.0 $t/resp.$n.0"; done; done)
  caffeinate -i tools/memguard.sh 8 harness/build/overflow_dist_dump --ctx "$KIT/keys/cryptocontext.bin" --sec "$KIT/keys/secret.key" \
      --out-dir "${SCR}_ext" --cts $CTS > $OUT/probe_c_dump_ext.jsonl 2> $OUT/probe_c_dump_ext.stderr
  echo "dump-ext exit=$? $(date -u +%FT%TZ)" >> $OUT/probe_c_loadstate.txt ;;
stats-ext)
  .venv/bin/python $OUT/overflow_dist_stats.py --scratch "$SCR" --scratch "${SCR}_ext" --tag _ext > $OUT/probe_c_stats_ext.stdout 2> $OUT/probe_c_stats_ext.stderr
  echo "stats-ext exit=$? $(date -u +%FT%TZ)" >> $OUT/probe_c_loadstate.txt ;;
clean)
  rm -rf "$SCR" "${SCR}_ext"; { echo "clean $(date -u +%FT%TZ)"; ls -ld "$SCR" "${SCR}_ext" 2>&1; } >> $OUT/probe_c_loadstate.txt ;;
*) echo "usage: $0 dump|stats|dump-ext|stats-ext|clean"; exit 2 ;;
esac
