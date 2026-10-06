#!/usr/bin/env bash
# 30_dump_indices.sh -- write the rotation/automorphism CONTRACT the server will
#                       need, so the Mac mints exactly that key set.
#
# PURPOSE   The 2026-09-01 sovereignty failure was a KEY-SET mismatch: the Mac
#           keygen replicated all rotation keys but OpenFHE's bootstrap giant-
#           step selection differs between builds, so the Mac set lacked
#           indices the server required and the harness died with map::at
#           (F62_CLASS_SWEEP_20260901.md "Key-sovereignty diagnosis";
#           mac_fhe_client.cpp:118-127). --dump-rot-indices makes the server
#           STATE its requirement up front: it builds the context from the same
#           flags, mints throwaway keys, writes the JSON and exits before
#           touching the GPU. The Mac then mints against that file.
# INPUTS    env per _lib.sh; $BIN with --dump-rot-indices; the pbd430a bundle
#           (the bundle is NOT read in dump mode: haveBundle is forced false and
#           d falls back to Dpad, so the contract depends only on the flags).
# OUTPUTS   $DEMO/contract_r17.json, $DEMO/dump_indices.log, printed Mac commands.
# PASS      contract parses; ring 131072, slots 65536, depth EXPECT_DEPTH (41 at
#           xd12), non-empty rotationAmounts/autoIndices; no CUDA text.
# RECORD    contract_r17.json (cite it beside the key upload).
#
# EXPECTED KEY BYTES: on disk 46,507,938,432 B for the 132-key set at 2^17/xd12
#   (s31_t31_keys_r17/console.log:1 evalRotKeyBytes; :6 "Rotation keys loaded:
#   132") = 352,333,321 B/key; the lanes-block set is 143 keys
#   (demo_smoke/lanes17.log:7 "143 ~ 48048MB" = 336 MB/key DEVICE-resident) ->
#   143 x 352.3 MB = 50.38 GB evalrot + evalmult 352,339,004 + public 88,086,142
#   + cryptocontext 2,866 + .dev 316 (keys_r17 console.log:20-25) = ~50.8 GB.
#   With --indices the Mac mints the contract exactly, so the 2^15-era surplus
#   (Mac 152-158 keys vs pod 102, F62 sweep) should not recur -- the keygen's
#   printed count is the check.
set -uo pipefail
SCRIPT=$(basename "$0")
. "$(dirname "$(readlink -f "$0")")/_lib.sh"
require_bin
mkdir -p "$DEMO"
OUT="$DEMO/contract_r17.json"; L="$DEMO/dump_indices.log"
# --dump-rot-indices never opens the bundle (gpu_real_model.cu: haveBundle is
# false in dump mode and d falls back to Dpad), and it REFUSES --selftest /
# --keys-load / --serve / --ct-in (standalone-mode guard). So the model flags
# are passed for the record only; a missing bundle is not an error here.
MF=("${MODEL_FLAGS[@]}")
[ -s "$ART/bundle_${TAG}.bin" ] || log "note: $ART/bundle_${TAG}.bin not staged yet -- fine for the dump (bundle is not read in this mode)"
log "dumping the rotation contract with the SERVER's flags"
echo "CMD: $BIN ${MF[*]} ${DEMO_FLAGS[*]} ${BLOCK_FLAGS[*]} --dump-rot-indices $OUT" | tee "$L"
"$BIN" "${MF[@]}" "${DEMO_FLAGS[@]}" "${BLOCK_FLAGS[@]}" --dump-rot-indices "$OUT" >> "$L" 2>&1
RC=$?
echo "RC=$RC" >> "$L"
P=()
# review 2026-09-03: a non-zero harness exit (3 = an index the harness's own keygen
# lacks, i.e. missingFromOwnKeygen non-empty) must never log PASS.
[ "$RC" = 0 ] || P+=("harness exit $RC (3 = missingFromOwnKeygen non-empty: see $L)")
cuda_ok "$L" || P+=("harness text: $(cuda_problems "$L" | tr '\n' ';')")
[ -s "$OUT" ] || P+=("contract not written: $OUT (is --dump-rot-indices in this binary? 10_bringup.sh checks)")
if [ -s "$OUT" ]; then
  INFO=$(python3 - "$OUT" "$EXPECT_DEPTH" "$LOG_RING" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); want_depth = int(sys.argv[2]); ring = 1 << int(sys.argv[3])   # 2026-09-17: ring from LOG_RING
rot = d.get("rotationAmounts") or []; auto = d.get("autoIndices") or []
n = len(set(auto)) if auto else len(set(rot))
probs = []
if d.get("ring") != ring: probs.append("ring %s != %d" % (d.get("ring"), ring))
if d.get("slots") != ring // 2: probs.append("slots %s != %d" % (d.get("slots"), ring // 2))
if d.get("depth") != want_depth: probs.append("depth %s != %d" % (d.get("depth"), want_depth))
if n == 0: probs.append("no indices")
per_key = 46507938432 / 132.0 * (ring / 131072.0) * (min(want_depth, 41) / 41.0)   # 2^17/depth-41 record, scaled (estimate)
tot = n * per_key + (352339004 + 88086142) * (ring / 131072.0) + 2866 + 316
print(json.dumps({"ring": d.get("ring"), "slots": d.get("slots"), "depth": d.get("depth"),
                  "bootstrapSlots": d.get("bootstrapSlots"), "nRotationAmounts": len(rot),
                  "nAutoIndices": len(auto), "nKeysToMint": n,
                  "expectedUploadBytes": int(tot), "expectedUploadGB": round(tot / 1e9, 2),
                  "minutesAt12.5MBps": round(tot / 12.5e6 / 60, 1), "problems": probs}))
PY
)
  echo "CONTRACT: $INFO" | tee -a "$L"
  echo "$INFO" | grep -q '"problems": \[\]' || P+=("contract check: $(echo "$INFO" | grep -o '"problems": \[[^]]*\]')")
fi
if [ ${#P[@]} -gt 0 ]; then for p in "${P[@]}"; do log "PROBLEM: $p"; done; campaign_log "FAIL dump_indices | ${P[*]} | $L"; exit 1; fi
NK=$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["nKeysToMint"])')
GB=$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["expectedUploadGB"])')
campaign_log "PASS dump_indices | keys=$NK expectedUpload=${GB}GB | $OUT"
cat <<EOF

# ===================== MAC SIDE (fill KEY/PORT/HOST) =====================
# 1. fetch the contract
scp -i KEY -P PORT root@HOST:$OUT ~/mac_keys_r17/contract_r17.json
# 2. mint the FULL key set on the Mac (secret.key never leaves it); r15 took 1m47s
#    (THURSDAY_RUNBOOK.md:11, transcript-class -- no log); r17 is UNMEASURED: time it.
mkdir -p ~/mac_keys_r17 && cd $REPO && \\
  ( time ./hpc_gpu_port/mac_fhe_client keygen --out ~/mac_keys_r17 --log-ring $LOG_RING --secure $SECURE \\
      --extra-depth $EXTRA_DEPTH --lanes-block 1 --indices ~/mac_keys_r17/contract_r17.json --batch 16 ) \\
  2>&1 | tee ~/mac_keys_r17/keygen.log
#    expect $NK keys across evalrot.part*.bin (16 per part); check the printed count against $NK
# 3. manifest of the EVAL material only (secret.key excluded by construction)
cd ~/mac_keys_r17 && shasum -a 256 cryptocontext.bin cryptocontext.bin.dev public.key evalmult.bin evalrot*.bin > MANIFEST.sha256
# 4. upload EVERYTHING EXCEPT secret.key -- small files first, parts in order, ONE stream
rsync -avh --progress --partial --partial-dir=.rsync-partial -e "ssh -i KEY -p PORT" \\
  --exclude secret.key --exclude keygen.log ~/mac_keys_r17/ root@HOST:$KEYS/
#    expected ~${GB} GB; ~$(echo "$INFO" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["minutesAt12.5MBps"])') min at 12.5 MB/s (100 Mbit)
# 5. then on the pod: bash $DEMO_TOOLS/50_sovereignty.sh
EOF
