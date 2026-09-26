#!/bin/bash
# Level-1 replay of the 2026-09-18 verification kit on the Mac: (1) manifest integrity, (2) re-decrypt every reply and every
# request with mac_fhe_client and compare with what the client recorded (hidden.f64 / input_rows.f64), (3) hidden rows -> tokens via
# the head matrix vs records/tokens.json. One heavy process at a time, under memguard. Writes JSON lines to $OUT/level1.jsonl.
set -u
K=$HOME/Documents/fhe-ssm-backup/demo_verification_demo_20260918T074952Z
REPO=$REPO; TOOL=$REPO/hpc_gpu_port/mac_fhe_client
OUT=$1; mkdir -p $OUT; L=$OUT/level1.jsonl; : > $L
echo "{\"step\":\"manifest\",\"utc\":\"$(date -u +%FT%TZ)\",\"start\":true}" >> $L
( cd $K && shasum -a 256 -c MANIFEST.sha256 > $OUT/manifest_check.txt 2>&1 ); rc=$?
echo "{\"step\":\"manifest\",\"rc\":$rc,\"ok\":$(grep -c ': OK$' $OUT/manifest_check.txt),\"failed\":$(grep -c 'FAILED' $OUT/manifest_check.txt)}" >> $L
for T in $(ls $K/session_cts | grep '^tick_'); do
  N=${T##*req}; D=$K/session_cts/$T
  for kind in resp req; do
    f=$D/$kind.$N; meta=$D/$kind.$N.meta; o=$OUT/${T}_$kind.f64
    t0=$(date +%s)
    $REPO/tools/memguard.sh 6 $TOOL dec --ctx $K/keys/cryptocontext.bin --sec $K/keys/secret.key --in $f --count 1 --lanes 64 --d 1024 --dpad 1024 --require-meta $meta --out $o > $OUT/${T}_$kind.log 2>&1; rc=$?
    t1=$(date +%s)
    echo "{\"step\":\"dec\",\"tick\":\"$T\",\"kind\":\"$kind\",\"rc\":$rc,\"sec\":$((t1-t0))}" >> $L
  done
done
echo "{\"step\":\"dec\",\"done\":true}" >> $L
