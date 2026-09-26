#!/usr/bin/env bash
# 20_stage_artifacts.sh -- ship the demo model + anchor bundle + prompts to the pod,
#                          and verify every byte against a recorded sha256.
#
# PURPOSE   ml-eval/artifacts is NOT git-tracked (0 files), so the pod clone has
#           no bundle, weights, config or seeds. This script (a) PRINTS the exact
#           Mac-side rsync commands, one stream at a time (three concurrent
#           rsyncs onto one partial path were a 2026-09-01 mistake,
#           CORRECTIONS_20260902.md C), and (b) on the pod VERIFIES what landed
#           against the .VERIFIED files (format: one line, the 64-hex sha256 --
#           e.g. pod_artifacts/bundle_pbd430a.bin.VERIFIED) and against the
#           hashes embedded below.
# INPUTS    `print` (on the Mac): SSH_KEY PORT HOST [MAC_BACKUP REPO]
#           `verify` (on the pod, default): env per _lib.sh
#           `export-lanes N` (on the pod, optional): export bundle_${TAG}lanesN
#           for --lanes-full N cells (40 optional / 80b), precedent queue cell
#           s31_t1c_fullpass_lanes_r15 (train_fhe_native_ssm.py export --tokens 8
#           --lanes N --bundle-suffix lanesN; record s31_t1c_fullpass_lanes_r15_a3/
#           console.log:2-3). Needs the venv and HF WikiText access.
# OUTPUTS   $ART/{bundle_pbd430a.bin,.index.txt,pbd430a_weights.npz,pbd430a_config.json,
#           pbd430a_seeds.json,bundle_agnd768b.bin,.index.txt}, $DEMO/demo_prompts_64.txt,
#           $DEMO/stage_verify.txt
# PASS      every sha256 equals the recorded value AND, where a .VERIFIED file
#           exists, equals it too. Byte sizes as listed.
# RECORD    stage_verify.txt (hash table = R2 provenance for every FHE record).
#
# Expected hashes (all computed on the Mac 2026-09-02 from the files named;
# the bundle hash equals its .VERIFIED; npz/config/seeds prefixes+suffixes match
# results/dense-demo-s31/RECOVERY_STATE_20260901.md:14,17,18; the anchor bundle
# is byte-identical to results/a100-secure-20260730/artifacts/bundle_agnd768b.bin):
#   bundle_pbd430a.bin        3448050888 B  67069799192af9e2447abca0e707c8464576d35f0f06aaae83029bd03b45b5fc
#   bundle_pbd430a.index.txt  18332 B       4fde393f8456ba8c09fcf16945310e965cf0125f40a9281ff11fd25df33c501d
#   pbd430a_weights.npz       1824931038 B  3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b
#   pbd430a_config.json       465 B         1829d34eeddbbfc4ce023a43d828085c6e71a2a941e7da8aa17790544b770ba7
#   pbd430a_seeds.json        9332 B        d63288181655ebbb5370a16be4b59d7bd6745fcf98bf8ca14e1ca15ec51245b7
#   bundle_agnd768b.bin       1313514120 B  12d3bf11063b589f90929855a11691953f10cf892cf15e633c6df559045c1be2
#   bundle_agnd768b.index.txt 8553 B        ae1b5dc4a4db12710470003fda558b0ca74628c0ac328248e96a4ad928089bc8
#   demo_prompts_64.txt       2709 B        0713393b8e61e58c9c9c1360e2c6fc3269b554280e5425c4140af02f95772f49
# Transfer arithmetic at 12.5 MB/s (100 Mbit, THURSDAY_RUNBOOK.md:4): 3.45 + 1.82
# + 1.31 GB = 6.59 GB -> ~9 min. At the 2026-09-01 35 KB/s it would be 2.2 days.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
MODE=${1:-verify}

EXPECT='bundle_pbd430a.bin 3448050888 67069799192af9e2447abca0e707c8464576d35f0f06aaae83029bd03b45b5fc
bundle_pbd430a.index.txt 18332 4fde393f8456ba8c09fcf16945310e965cf0125f40a9281ff11fd25df33c501d
pbd430a_weights.npz 1824931038 3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b
pbd430a_config.json 465 1829d34eeddbbfc4ce023a43d828085c6e71a2a941e7da8aa17790544b770ba7
pbd430a_seeds.json 9332 d63288181655ebbb5370a16be4b59d7bd6745fcf98bf8ca14e1ca15ec51245b7
bundle_agnd768b.bin 1313514120 12d3bf11063b589f90929855a11691953f10cf892cf15e633c6df559045c1be2
bundle_agnd768b.index.txt 8553 ae1b5dc4a4db12710470003fda558b0ca74628c0ac328248e96a4ad928089bc8'
PROMPTS_SHA=0713393b8e61e58c9c9c1360e2c6fc3269b554280e5425c4140af02f95772f49

case "$MODE" in
print)
  : "${SSH_KEY:=KEY}"; : "${PORT:=PORT}"; : "${HOST:=HOST}"
  : "${MAC_BACKUP:=$BACKUP}"
  : "${REPO:=$REPO}"
  RS="rsync -avh --progress --partial --partial-dir=.rsync-partial -e \"ssh -i $SSH_KEY -p $PORT\""
  cat <<EOF
# ---- run ON THE MAC, sequentially (one stream; each && stops on failure) ----
ssh -i $SSH_KEY -p $PORT root@$HOST "mkdir -p $ART $DEMO && (command -v rsync >/dev/null || apt-get install -y -qq rsync)"
# 1. anchor bundle first (10_bringup.sh anchor needs it; 1.31 GB)
$RS $REPO/ml-eval/artifacts/bundle_agnd768b.bin $REPO/ml-eval/artifacts/bundle_agnd768b.index.txt root@$HOST:$ART/ \\
&& \\
# 2. demo bundle + its .VERIFIED + index (3.45 GB)
$RS $MAC_BACKUP/pod_artifacts/bundle_pbd430a.bin $MAC_BACKUP/pod_artifacts/bundle_pbd430a.bin.VERIFIED $MAC_BACKUP/pod_artifacts/bundle_pbd430a.index.txt root@$HOST:$ART/ \\
&& \\
# 3. demo weights (1.82 GB) -- renamed to the loader's name (_native_loader_copy.py:40 reads {tag}_weights.npz)
$RS $MAC_BACKUP/pbd430a_weights_DEMO_MODEL_20260901.npz root@$HOST:$ART/pbd430a_weights.npz \\
&& \\
# 4. config + seeds (the export/loader sidecars) + the 64 demo prompts (D10 pool, tracked in git too)
$RS $REPO/ml-eval/artifacts/pbd430a_config.json $REPO/ml-eval/artifacts/pbd430a_seeds.json root@$HOST:$ART/ \\
&& $RS $REPO/results/dense-demo-s31/sessions/demo_prompts_64.txt root@$HOST:$DEMO/ \\
&& ssh -i $SSH_KEY -p $PORT root@$HOST "bash $DEMO_TOOLS/$SCRIPT verify"
EOF
  ;;
verify)
  mkdir -p "$DEMO"
  OUT="$DEMO/stage_verify.txt"
  FAIL=0
  # brace group + redirection runs in THIS shell (a `| tee` pipeline would fork
  # it and lose FAIL -> rc 0 on a failed verify); printed afterwards.
  {
    echo "=== STAGE VERIFY $(utc) ART=$ART ==="
    printf '%-4s %-28s %-12s %-12s %s\n' ok file bytes want sha256
    while read -r f want_bytes want_sha; do
      p="$ART/$f"
      if [ ! -s "$p" ]; then printf '%-4s %-28s %s\n' MISS "$f" "$p"; FAIL=1; continue; fi
      b=$(file_bytes "$p"); s=$(sha256sum "$p" | cut -c1-64)
      st=OK
      [ "$b" = "$want_bytes" ] || st=SIZE
      [ "$s" = "$want_sha" ] || st=SHA
      if [ -s "$p.VERIFIED" ]; then v=$(tr -dc 'a-f0-9' < "$p.VERIFIED" | cut -c1-64); [ "$v" = "$s" ] || st="SHA(.VERIFIED=$v)"; fi
      [ "$st" = OK ] || FAIL=1
      printf '%-4s %-28s %-12s %-12s %s\n' "$st" "$f" "$b" "$want_bytes" "$s"
    done <<< "$EXPECT"
    pf="$DEMO/demo_prompts_64.txt"
    [ -s "$pf" ] || pf="$POD_SRC/results/dense-demo-s31/sessions/demo_prompts_64.txt"
    if [ -s "$pf" ]; then s=$(sha256sum "$pf" | cut -c1-64); st=OK; [ "$s" = "$PROMPTS_SHA" ] || { st=SHA; FAIL=1; }
      printf '%-4s %-28s %-12s %-12s %s\n' "$st" demo_prompts_64.txt "$(file_bytes "$pf")" 2709 "$s"; echo "prompts path: $pf ($(grep -c . "$pf") lines)"
    else echo "MISS demo_prompts_64.txt"; FAIL=1; fi
    [ "$FAIL" = 0 ] && echo "=== STAGE VERIFY: PASS ===" || echo "=== STAGE VERIFY: FAIL ==="
  } > "$OUT"
  cat "$OUT"
  campaign_log "$([ "$FAIL" = 0 ] && echo PASS || echo FAIL) stage_verify | $OUT"
  exit "$FAIL"
  ;;
export-lanes)
  N=${2:?usage: $SCRIPT export-lanes N}
  busy_acquire
  require_file "$ART/${TAG}_weights.npz" "run verify first"
  L="$DEMO/export_lanes${N}.log"
  log "exporting bundle_${TAG}lanes${N} (precedent: s31_t1c_fullpass_lanes_r15 cmd; needs HF WikiText)"
  ( cd "$POD_SRC/ml-eval" && "$VENV/bin/python" train_fhe_native_ssm.py export --tag "$TAG" --tokens 8 --lanes "$N" --bundle-suffix "lanes$N" ) > "$L" 2>&1
  echo "RC=$?" >> "$L"
  if grep -q "\"exported\": \"bundle_${TAG}lanes${N}.bin\"" "$L" && [ -s "$ART/bundle_${TAG}lanes${N}.bin" ]; then
    log "EXPORT OK: $(grep -o '{"exported"[^}]*}' "$L")"; campaign_log "PASS export-lanes$N | $(grep -o '{"exported"[^}]*}' "$L") | $L"
  else
    log "EXPORT FAILED -- see $L"; campaign_log "FAIL export-lanes$N | $L"; busy_release; exit 1
  fi
  busy_release
  ;;
*) echo "usage: $SCRIPT print|verify|export-lanes N"; exit 2 ;;
esac
